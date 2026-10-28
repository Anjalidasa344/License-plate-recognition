import cv2
import numpy as np
import onnxruntime as ort

# ==========================================
# CONFIGURATION
# ==========================================

PLATE_MODEL = "networks/az_plate/az_plate_ssdmobilenetv1.onnx"
OCR_MODEL = "networks/az_ocr/az_ocr_ssdmobilenetv1_2.onnx"
IMAGE_PATH = "test_car.png"
LABELS_PATH = "networks/az_ocr/labels.txt"

PLATE_THRESHOLD = 0.50
OCR_THRESHOLD = 0.15
NMS_THRESHOLD = 0.35


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess(image):
    resized = cv2.resize(image, (300, 300))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    data = rgb.astype(np.float32) / 255.0
    data = np.transpose(data, (2, 0, 1))

    return np.expand_dims(data, axis=0)


# ==========================================
# LOAD IMAGE AND MODELS
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Cannot load {IMAGE_PATH}. Check the image path."
    )

height, width = image.shape[:2]

print("Image loaded successfully!")
print("Image dimensions:", width, "x", height)

plate_session = ort.InferenceSession(
    PLATE_MODEL,
    providers=["CPUExecutionProvider"]
)

ocr_session = ort.InferenceSession(
    OCR_MODEL,
    providers=["CPUExecutionProvider"]
)

with open(LABELS_PATH, encoding="utf-8") as file:
    labels = file.read().splitlines()

print("Plate model loaded.")
print("OCR model loaded.")
print("Number of labels:", len(labels))


# ==========================================
# STEP 1: DETECT THE NUMBER PLATE
# ==========================================

plate_input = preprocess(image)

plate_outputs = plate_session.run(
    None,
    {"input_0": plate_input}
)

plate_scores, plate_boxes = plate_outputs

best_index = int(np.argmax(plate_scores[0, :, 1]))
plate_confidence = float(
    plate_scores[0, best_index, 1]
)

if plate_confidence < PLATE_THRESHOLD:
    raise RuntimeError(
        "No confident number plate detected."
    )

# The detector boxes are assumed to be normalized
# [y1, x1, y2, x2], matching the original script.

y1, x1, y2, x2 = plate_boxes[0, best_index]

left = int(x1 * width)
right = int(x2 * width)
top = int(y1 * height)
bottom = int(y2 * height)

# Expand the detected region slightly.
margin_x = int(width * 0.02)
margin_y = int(height * 0.02)

left = max(0, left - margin_x)
right = min(width, right + margin_x)
top = max(0, top - margin_y)
bottom = min(height, bottom + margin_y)

plate_crop = image[top:bottom, left:right]

if plate_crop.size == 0:
    raise RuntimeError("The detected plate crop is empty.")

cv2.imwrite("plate_crop.jpg", plate_crop)

print("\nPLATE DETECTION")
print("Confidence:", round(plate_confidence, 4))
print("Crop dimensions:", plate_crop.shape)
print("Saved: plate_crop.jpg")


# ==========================================
# STEP 2: RUN THE OCR MODEL
# ==========================================

ocr_input = preprocess(plate_crop)

scores, boxes = ocr_session.run(
    None,
    {"input_0": ocr_input}
)

print("\nOCR DEBUG INFORMATION")
print("Scores shape:", scores.shape)
print("Boxes shape:", boxes.shape)
print("Maximum score:", float(np.max(scores)))

if scores.ndim != 3 or boxes.ndim != 3:
    raise RuntimeError(
        "Unexpected OCR output dimensions. "
        "Inspect the model output format."
    )

if scores.shape[1] != boxes.shape[1]:
    raise RuntimeError(
        "OCR scores and boxes have different detection counts."
    )


# ==========================================
# STEP 3: EXTRACT CHARACTER DETECTIONS
# ==========================================

crop_height, crop_width = plate_crop.shape[:2]

detections = []

for i in range(scores.shape[1]):

    # Assumes score column 0 is background and
    # columns 1 onward correspond to labels 1 onward.
    class_id = int(
        np.argmax(scores[0, i, 1:])
    ) + 1

    confidence = float(
        scores[0, i, class_id]
    )

    if confidence < OCR_THRESHOLD:
        continue

    if class_id >= len(labels):
        continue

    # Assumes normalized box format [x1, y1, x2, y2].
    x1, y1, x2, y2 = boxes[0, i]

    x1 = int(x1 * crop_width)
    y1 = int(y1 * crop_height)
    x2 = int(x2 * crop_width)
    y2 = int(y2 * crop_height)

    x1 = max(0, min(crop_width, x1))
    x2 = max(0, min(crop_width, x2))
    y1 = max(0, min(crop_height, y1))
    y2 = max(0, min(crop_height, y2))

    if x2 <= x1 or y2 <= y1:
        continue

    label = labels[class_id]

    if label.upper() == "BACKGROUND":
        continue

    detections.append({
        "label": label,
        "confidence": confidence,
        "box": [x1, y1, x2, y2]
    })

print("\nOCR DEBUG")
print("Candidates above threshold:", len(detections))


# ==========================================
# STEP 4: REMOVE DUPLICATE CHARACTER BOXES
# ==========================================

final_detections = []

unique_labels = sorted(
    set(item["label"] for item in detections)
)

for label in unique_labels:

    group = [
        item for item in detections
        if item["label"] == label
    ]

    # OpenCV NMSBoxes expects [x, y, width, height],
    # not [x1, y1, x2, y2].

    nms_boxes = []

    for item in group:
        x1, y1, x2, y2 = item["box"]

        nms_boxes.append([
            x1,
            y1,
            max(1, x2 - x1),
            max(1, y2 - y1)
        ])

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        [item["confidence"] for item in group],
        OCR_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(indices) > 0:
        for index in np.array(indices).flatten():
            final_detections.append(group[int(index)])


# ==========================================
# STEP 5: SORT CHARACTERS LEFT TO RIGHT
# ==========================================

final_detections.sort(
    key=lambda item: item["box"][0]
)

recognized_text = "".join(
    item["label"] for item in final_detections
)

print("\nOCR RESULTS")
print("Characters detected:", len(final_detections))
print("Recognized text:", recognized_text)

for item in final_detections:
    print(
        f"Character: {item['label']} | "
        f"Confidence: {item['confidence']:.3f} | "
        f"Box: {item['box']}"
    )


# ==========================================
# STEP 6: SAVE OCR VISUALIZATION
# ==========================================

result = plate_crop.copy()

for item in final_detections:

    x1, y1, x2, y2 = item["box"]

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        item["label"],
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

cv2.imwrite("ocr_result.jpg", result)

print("\nOCR visualization saved: ocr_result.jpg")
print("Open the project folder to view the images.")
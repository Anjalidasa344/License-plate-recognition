import cv2
import numpy as np
import onnxruntime as ort

# ==========================================
# MODEL PATHS
# ==========================================

PLATE_MODEL = "networks/az_plate/az_plate_ssdmobilenetv1.onnx"
OCR_MODEL = "networks/az_ocr/az_ocr_ssdmobilenetv1_2.onnx"
LABELS_PATH = "networks/az_ocr/labels.txt"
IMAGE_PATH = "test_car.png"

PLATE_THRESHOLD = 0.40
OCR_THRESHOLD = 0.15


# ==========================================
# PREPROCESS IMAGE
# ==========================================

def preprocess(image):
    resized = cv2.resize(image, (300, 300))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    data = rgb.astype(np.float32) / 255.0
    data = np.transpose(data, (2, 0, 1))

    return np.expand_dims(data, axis=0)


# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load {IMAGE_PATH}. "
        "Check that the image exists in the project folder."
    )

height, width = image.shape[:2]

print("Image loaded successfully!")
print("Image dimensions:", width, "x", height)


# ==========================================
# LOAD MODELS
# ==========================================

plate_session = ort.InferenceSession(
    PLATE_MODEL,
    providers=["CPUExecutionProvider"]
)

ocr_session = ort.InferenceSession(
    OCR_MODEL,
    providers=["CPUExecutionProvider"]
)


# ==========================================
# LOAD CHARACTER LABELS
# ==========================================

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    labels = file.read().splitlines()


# ==========================================
# STEP 1: DETECT NUMBER PLATE
# ==========================================

plate_input = preprocess(image)

plate_scores, plate_boxes = plate_session.run(
    None,
    {"input_0": plate_input}
)

plate_scores = plate_scores[0]
plate_boxes = plate_boxes[0]

if plate_scores.ndim != 2 or plate_scores.shape[1] < 2:
    raise RuntimeError(
        f"Unexpected plate scores shape: {plate_scores.shape}"
    )

best_index = int(np.argmax(plate_scores[:, 1]))
plate_confidence = float(plate_scores[best_index, 1])

if plate_confidence < PLATE_THRESHOLD:
    raise RuntimeError(
        f"No confident plate detected. "
        f"Best confidence: {plate_confidence:.4f}"
    )

# The earlier test_model.py result used [x1, y1, x2, y2].
x1, y1, x2, y2 = plate_boxes[best_index]

left = int(x1 * width)
top = int(y1 * height)
right = int(x2 * width)
bottom = int(y2 * height)

# Keep coordinates inside the image.
left = max(0, min(width, left))
right = max(0, min(width, right))
top = max(0, min(height, top))
bottom = max(0, min(height, bottom))

if right <= left or bottom <= top:
    raise RuntimeError(
        "Invalid plate coordinates. "
        "The model's bounding-box format needs checking."
    )

plate_crop = image[top:bottom, left:right].copy()

if plate_crop.size == 0:
    raise RuntimeError("The detected plate crop is empty.")

cv2.imwrite("plate_crop.jpg", plate_crop)

print("\nPlate detected!")
print("Plate confidence:", round(plate_confidence, 4))
print("Plate coordinates:", left, top, right, bottom)
print("Crop dimensions:", plate_crop.shape)
print("Saved: plate_crop.jpg")


# ==========================================
# STEP 2: RUN OCR
# ==========================================

# Stretching to 300 x 300 is retained because our
# earlier comparison gave stronger candidate scores
# with this preprocessing method.
ocr_input = preprocess(plate_crop)

scores, boxes = ocr_session.run(
    None,
    {"input_0": ocr_input}
)

scores = scores[0]
boxes = boxes[0]

print("\nOCR DEBUG INFORMATION")
print("Scores shape:", scores.shape)
print("Boxes shape:", boxes.shape)
print("Maximum score:", float(np.max(scores)))


# ==========================================
# STEP 3: EXTRACT CHARACTER PREDICTIONS
# ==========================================

crop_height, crop_width = plate_crop.shape[:2]

# Keep the best prediction for each overlapping box
# of the same character class.
candidates = []

for i in range(scores.shape[0]):

    # Column 0 represents background.
    class_id = int(np.argmax(scores[i, 1:])) + 1
    confidence = float(scores[i, class_id])

    if confidence < OCR_THRESHOLD:
        continue

    if class_id >= len(labels):
        continue

    # Character boxes use [x1, y1, x2, y2].
    bx1, by1, bx2, by2 = boxes[i]

    bx1 = int(bx1 * crop_width)
    by1 = int(by1 * crop_height)
    bx2 = int(bx2 * crop_width)
    by2 = int(by2 * crop_height)

    bx1 = max(0, min(crop_width, bx1))
    bx2 = max(0, min(crop_width, bx2))
    by1 = max(0, min(crop_height, by1))
    by2 = max(0, min(crop_height, by2))

    if bx2 <= bx1 or by2 <= by1:
        continue

    candidates.append({
        "box": [bx1, by1, bx2 - bx1, by2 - by1],
        "label": labels[class_id],
        "class_id": class_id,
        "confidence": confidence
    })


# ==========================================
# STEP 4: NON-MAXIMUM SUPPRESSION PER CLASS
# ==========================================

detections = []

class_ids = sorted(set(
    item["class_id"] for item in candidates
))

for class_id in class_ids:

    same_class = [
        item for item in candidates
        if item["class_id"] == class_id
    ]

    nms_boxes = [item["box"] for item in same_class]
    nms_scores = [
        item["confidence"] for item in same_class
    ]

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        nms_scores,
        OCR_THRESHOLD,
        0.35
    )

    if len(indices) > 0:
        indices = np.array(indices).flatten()

        for index in indices:
            detections.append(same_class[int(index)])


# ==========================================
# STEP 5: SORT CHARACTERS LEFT TO RIGHT
# ==========================================

detections.sort(
    key=lambda item: item["box"][0]
)

recognized_text = "".join(
    item["label"] for item in detections
)

print("\nOCR RESULTS")
print("Characters detected:", len(detections))
print("Recognized text:", recognized_text)

for item in detections:
    print(
        f"Character: {item['label']} | "
        f"Confidence: {item['confidence']:.3f} | "
        f"Position: {item['box'][0]}"
    )


# ==========================================
# STEP 6: SAVE VISUALIZATION
# ==========================================

result = plate_crop.copy()

for item in detections:

    x, y, w, h = item["box"]

    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        item["label"],
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

cv2.imwrite("ocr_result.jpg", result)

print("\nOCR visualization saved: ocr_result.jpg")

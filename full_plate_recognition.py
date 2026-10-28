import cv2
import numpy as np
import onnxruntime as ort
import easyocr
import re

# ==========================================
# CONFIGURATION
# ==========================================

PLATE_MODEL = "networks/az_plate/az_plate_ssdmobilenetv1.onnx"
IMAGE_PATH = "test_car.png"

PLATE_THRESHOLD = 0.40

# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Cannot find {IMAGE_PATH}. "
        "Place the image in the project folder."
    )

height, width = image.shape[:2]

print("Image loaded successfully!")
print("Image dimensions:", width, "x", height)

# ==========================================
# PREPROCESS FOR ONNX MODEL
# ==========================================

def preprocess(image):
    resized = cv2.resize(image, (300, 300))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    data = rgb.astype(np.float32) / 255.0
    data = np.transpose(data, (2, 0, 1))

    return np.expand_dims(data, axis=0)


# ==========================================
# LOAD PLATE DETECTOR
# ==========================================

print("\nLoading plate detection model...")

plate_session = ort.InferenceSession(
    PLATE_MODEL,
    providers=["CPUExecutionProvider"]
)

# ==========================================
# DETECT PLATE
# ==========================================

scores, boxes = plate_session.run(
    None,
    {"input_0": preprocess(image)}
)

scores = scores[0]
boxes = boxes[0]

best_index = int(np.argmax(scores[:, 1]))
plate_confidence = float(scores[best_index, 1])

print("Plate detection confidence:", round(plate_confidence, 4))

if plate_confidence < PLATE_THRESHOLD:
    raise RuntimeError(
        "No confident number plate detected."
    )

# Coordinates are interpreted as normalized
# [x1, y1, x2, y2], as in the earlier test_model.py.
x1, y1, x2, y2 = boxes[best_index]

left = max(0, min(width, int(x1 * width)))
right = max(0, min(width, int(x2 * width)))
top = max(0, min(height, int(y1 * height)))
bottom = max(0, min(height, int(y2 * height)))

if right <= left or bottom <= top:
    raise RuntimeError(
        "Invalid plate coordinates. Check the model's box format."
    )

# Add a small margin around the detected plate.
margin_x = int((right - left) * 0.04)
margin_y = int((bottom - top) * 0.08)

left = max(0, left - margin_x)
right = min(width, right + margin_x)
top = max(0, top - margin_y)
bottom = min(height, bottom + margin_y)

plate_crop = image[top:bottom, left:right].copy()

if plate_crop.size == 0:
    raise RuntimeError("Plate crop is empty.")

cv2.imwrite("automatic_plate_crop.jpg", plate_crop)

print("Plate crop saved: automatic_plate_crop.jpg")

# ==========================================
# IMPROVE PLATE IMAGE
# ==========================================

scale = 3

enlarged = cv2.resize(
    plate_crop,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

gray = cv2.cvtColor(
    enlarged,
    cv2.COLOR_BGR2GRAY
)

gray = cv2.equalizeHist(gray)

cv2.imwrite("automatic_enhanced_plate.jpg", gray)

# ==========================================
# LOAD EASYOCR
# ==========================================

print("\nLoading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

# ==========================================
# RECOGNIZE CHARACTERS
# ==========================================

print("Recognizing number plate...")

results = reader.readtext(
    gray,
    detail=1,
    paragraph=False,
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
)

# ==========================================
# DISPLAY OCR RESULT
# ==========================================

print("\n========== RECOGNITION RESULT ==========")

if not results:
    print("No characters recognized.")
    recognized_text = ""
    average_confidence = 0.0

else:
    recognized_parts = []

    for box, text, confidence in results:
        text = text.strip()

        if text:
            recognized_parts.append(text)

            print(
                f"Detected: {text} | "
                f"Confidence: {confidence:.3f}"
            )

    recognized_text = "".join(recognized_parts)

    if recognized_parts:
        average_confidence = (
            sum(float(item[2]) for item in results if item[1].strip())
            / len(recognized_parts)
        )
    else:
        average_confidence = 0.0

print("\nRecognized plate:", recognized_text)
print("Average OCR confidence:", round(average_confidence, 3))

# ==========================================
# DRAW DETECTION ON ORIGINAL IMAGE
# ==========================================

output = image.copy()

cv2.rectangle(
    output,
    (left, top),
    (right, bottom),
    (0, 255, 0),
    3
)

display_text = recognized_text if recognized_text else "Not recognized"

cv2.putText(
    output,
    display_text,
    (left, max(top - 12, 30)),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 255),
    2
)

cv2.imwrite("automatic_recognition_result.jpg", output)

print("\nSaved output files:")
print("1. automatic_plate_crop.jpg")
print("2. automatic_enhanced_plate.jpg")
print("3. automatic_recognition_result.jpg")

print("\nProcessing completed!")

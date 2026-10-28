import cv2
import easyocr

# =====================================
# LOAD NUMBER PLATE IMAGE
# =====================================

IMAGE_PATH = "plate_crop.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        "plate_crop.jpg not found! Run test_ocr.py first."
    )

print("Plate image loaded successfully!")

# =====================================
# ENLARGE THE NUMBER PLATE
# =====================================

scale = 3

enlarged = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

# =====================================
# IMPROVE IMAGE QUALITY
# =====================================

gray = cv2.cvtColor(
    enlarged,
    cv2.COLOR_BGR2GRAY
)

gray = cv2.equalizeHist(gray)

# Save enhanced image for checking
cv2.imwrite("enhanced_plate.jpg", gray)

print("Enhanced plate image saved!")

# =====================================
# LOAD EASYOCR
# =====================================

print("\nLoading EasyOCR...")
print("The first run may download its model.")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

# =====================================
# RECOGNIZE CHARACTERS
# =====================================

print("\nReading number plate...")

results = reader.readtext(
    gray,
    detail=1,
    paragraph=False,
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
)

# =====================================
# DISPLAY RESULTS
# =====================================

print("\n========== OCR RESULTS ==========")

recognized_parts = []

for box, text, confidence in results:

    text = text.strip()

    if text:
        recognized_parts.append(text)

        print(
            "Text:",
            text,
            "| Confidence:",
            round(float(confidence), 3)
        )

recognized_text = "".join(recognized_parts)

print("\nRecognized plate:", recognized_text)

# =====================================
# DRAW DETECTION BOXES
# =====================================

output = cv2.cvtColor(
    gray,
    cv2.COLOR_GRAY2BGR
)

for box, text, confidence in results:

    points = [
        tuple(map(int, point))
        for point in box
    ]

    for i in range(4):
        cv2.line(
            output,
            points[i],
            points[(i + 1) % 4],
            (0, 255, 0),
            3
        )

    cv2.putText(
        output,
        text,
        points[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

# =====================================
# SAVE FINAL RESULT
# =====================================

cv2.imwrite(
    "easy_ocr_result.jpg",
    output
)

print("\nSaved: easy_ocr_result.jpg")
print("Saved: enhanced_plate.jpg")
print("\nProcessing completed!")

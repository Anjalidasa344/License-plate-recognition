
import cv2
import onnxruntime as ort
import numpy as np

# Load the new Toyota image
image = cv2.imread("test_car.png")

if image is None:
    raise FileNotFoundError("test_car.png not found!")

height, width = image.shape[:2]

print("Image loaded successfully!")
print("Image dimensions:", width, "x", height)

# Preserve aspect ratio while resizing
scale = min(300 / width, 300 / height)
new_width = int(width * scale)
new_height = int(height * scale)

resized = cv2.resize(image, (new_width, new_height))

# Create a 300x300 canvas
canvas = np.zeros((300, 300, 3), dtype=np.uint8)

x_offset = (300 - new_width) // 2
y_offset = (300 - new_height) // 2

canvas[
    y_offset:y_offset + new_height,
    x_offset:x_offset + new_width
] = resized

# Prepare model input
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
input_data = rgb.astype(np.float32) / 255.0
input_data = np.transpose(input_data, (2, 0, 1))
input_data = np.expand_dims(input_data, axis=0)

# Load the license plate detection model
session = ort.InferenceSession(
    "networks/az_plate/az_plate_ssdmobilenetv1.onnx",
    providers=["CPUExecutionProvider"]
)

# Run detection
scores, boxes = session.run(
    None,
    {"input_0": input_data}
)

# Find the highest-confidence plate prediction
best_index = int(np.argmax(scores[0, :, 1]))
confidence = float(scores[0, best_index, 1])

# Get predicted bounding box
x1, y1, x2, y2 = boxes[0, best_index]

# Convert coordinates to 300x300 pixels
x1 = x1 * 300
x2 = x2 * 300
y1 = y1 * 300
y2 = y2 * 300

# Map coordinates back to original image
x1 = (x1 - x_offset) / scale
x2 = (x2 - x_offset) / scale
y1 = (y1 - y_offset) / scale
y2 = (y2 - y_offset) / scale

# Clip coordinates to image boundaries
x1 = max(0, min(width, int(x1)))
x2 = max(0, min(width, int(x2)))
y1 = max(0, min(height, int(y1)))
y2 = max(0, min(height, int(y2)))

# Draw detection and save plate crop
output = image.copy()

if x2 > x1 and y2 > y1:
    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        output,
        f"Plate: {confidence:.3f}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    crop = image[y1:y2, x1:x2]
    cv2.imwrite("plate_crop.jpg", crop)
    print("Plate crop saved: plate_crop.jpg")
else:
    print("Invalid bounding box. No crop saved.")

# Save detection result
cv2.imwrite("detected_plate.jpg", output)

print("Detection visualization saved: detected_plate.jpg")
print("Confidence:", round(confidence, 4))
print("Bounding box:", x1, y1, x2, y2)

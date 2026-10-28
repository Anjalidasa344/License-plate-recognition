import cv2
import numpy as np
import onnxruntime as ort

MODEL = "networks/az_ocr/az_ocr_ssdmobilenetv1_2.onnx"
IMAGE = "plate_crop.jpg"
INPUT_SIZE = 300

# Load the plate image
image = cv2.imread(IMAGE)

if image is None:
    raise FileNotFoundError(
        f"Cannot open {IMAGE}. Check your project folder."
    )

print("Original crop size:", image.shape)

# Load the OCR model
session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

print("\nModel inputs:")
for item in session.get_inputs():
    print(item.name, item.shape, item.type)

print("\nModel outputs:")
for item in session.get_outputs():
    print(item.name, item.shape, item.type)

# Load character labels
with open(
    "networks/az_ocr/labels.txt",
    "r",
    encoding="utf-8"
) as file:
    labels = file.read().splitlines()

print("\nNumber of labels:", len(labels))

# Resize the image while preserving proportions
height, width = image.shape[:2]

scale = min(
    INPUT_SIZE / width,
    INPUT_SIZE / height
)

new_width = max(1, int(width * scale))
new_height = max(1, int(height * scale))

resized = cv2.resize(
    image,
    (new_width, new_height),
    interpolation=cv2.INTER_AREA
)

# Add black padding
padded = np.zeros(
    (INPUT_SIZE, INPUT_SIZE, 3),
    dtype=np.uint8
)

x_offset = (INPUT_SIZE - new_width) // 2
y_offset = (INPUT_SIZE - new_height) // 2

padded[
    y_offset:y_offset + new_height,
    x_offset:x_offset + new_width
] = resized

cv2.imwrite("debug_ocr_input.jpg", padded)

print("\nProcessed image saved: debug_ocr_input.jpg")
print("Processed image size:", padded.shape)

# Prepare the model input
rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

data = rgb.astype(np.float32) / 255.0

data = np.expand_dims(
    data.transpose(2, 0, 1),
    axis=0
)

print("Model input shape:", data.shape)

# Run the OCR model
input_name = session.get_inputs()[0].name

outputs = session.run(
    None,
    {input_name: data}
)

print("\nRaw output count:", len(outputs))

for i, output in enumerate(outputs):
    print(
        f"Output {i}: shape={output.shape}, "
        f"min={output.min():.6f}, "
        f"max={output.max():.6f}"
    )

if len(outputs) < 2:
    raise RuntimeError("Expected scores and boxes from the model.")

scores = outputs[0]
boxes = outputs[1]

if scores.ndim != 3 or boxes.ndim != 3:
    raise RuntimeError(
        f"Unexpected output shapes: {scores.shape}, {boxes.shape}"
    )

if scores.shape[1] != boxes.shape[1]:
    raise RuntimeError("The scores and boxes counts do not match.")

# Collect character predictions
print("\nTop 15 character predictions:")

candidates = []

for i in range(scores.shape[1]):
    class_id = int(np.argmax(scores[0, i, 1:])) + 1
    confidence = float(scores[0, i, class_id])

    if class_id < len(labels):
        candidates.append(
            (
                confidence,
                labels[class_id],
                boxes[0, i].copy(),
                i
            )
        )

candidates.sort(
    key=lambda item: item[0],
    reverse=True
)

for confidence, label, box, index in candidates[:15]:
    print(
        f"Index={index}, label={label}, "
        f"confidence={confidence:.4f}, "
        f"box={np.round(box, 4)}"
    )

print("\nDebugging completed.")
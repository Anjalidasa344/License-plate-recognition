import cv2
import numpy as np
import onnxruntime as ort

MODEL = "networks/az_ocr/az_ocr_ssdmobilenetv1_2.onnx"
IMAGE = "plate_crop.jpg"
SIZE = 300

# Load the plate crop
image = cv2.imread(IMAGE)

if image is None:
    raise FileNotFoundError(
        "Cannot open plate_crop.jpg. Run python test_model.py first."
    )

print("Plate image size:", image.shape)

# Load OCR model
session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# Method 1: Stretch image to 300 x 300
def prepare_stretched(image):
    resized = cv2.resize(image, (SIZE, SIZE))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    data = rgb.astype(np.float32) / 255.0

    return np.expand_dims(
        data.transpose(2, 0, 1),
        axis=0
    )


# Method 2: Preserve proportions with black padding
def prepare_padded(image):
    height, width = image.shape[:2]

    scale = min(SIZE / width, SIZE / height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (SIZE, SIZE, 3),
        dtype=np.uint8
    )

    x_offset = (SIZE - new_width) // 2
    y_offset = (SIZE - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    cv2.imwrite("compare_padded.jpg", canvas)

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    data = rgb.astype(np.float32) / 255.0

    return np.expand_dims(
        data.transpose(2, 0, 1),
        axis=0
    )


# Display the strongest predictions for each method
def inspect_predictions(name, data):
    scores, boxes = session.run(
        None,
        {input_name: data}
    )

    print("\n==========", name, "==========")
    print("Scores shape:", scores.shape)
    print("Boxes shape:", boxes.shape)

    candidates = []

    for i in range(scores.shape[1]):
        class_id = int(
            np.argmax(scores[0, i, 1:])
        ) + 1

        confidence = float(
            scores[0, i, class_id]
        )

        candidates.append(
            (
                confidence,
                class_id,
                i,
                boxes[0, i]
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    print("\nTop 10 predictions:")

    for confidence, class_id, index, box in candidates[:10]:
        print(
            f"Index={index}, "
            f"Class={class_id}, "
            f"Confidence={confidence:.4f}, "
            f"Box={np.round(box, 4)}"
        )

    print(
        "\nMaximum score:",
        round(float(scores.max()), 6)
    )


# Prepare both inputs
stretched = prepare_stretched(image)
padded = prepare_padded(image)

cv2.imwrite(
    "compare_stretched.jpg",
    cv2.resize(image, (SIZE, SIZE))
)

# Compare model outputs
inspect_predictions(
    "STRETCHED IMAGE",
    stretched
)

inspect_predictions(
    "PADDED IMAGE",
    padded
)

print("\nComparison completed!")
print("Images saved:")
print("1. compare_stretched.jpg")
print("2. compare_padded.jpg")
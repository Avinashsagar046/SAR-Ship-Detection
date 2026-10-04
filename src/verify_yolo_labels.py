import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_DIR = BASE_DIR / "yolo_dataset"

IMAGE_DIR = YOLO_DIR / "images" / "train"
LABEL_DIR = YOLO_DIR / "labels" / "train"


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_IMAGES = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = {".png", ".jpg", ".jpeg"}

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

print("=" * 60)
print("YOLO LABEL VERIFICATION")
print("=" * 60)

print(f"Training images available: {len(images)}")


# ============================================================
# SELECT RANDOM IMAGES
# ============================================================

selected_images = random.sample(
    images,
    min(NUMBER_OF_IMAGES, len(images))
)


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for image_path in selected_images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    print("\n" + "-" * 60)
    print(f"Image: {image_path.name}")

    if not label_path.exists():
        print("WARNING: Label file not found!")
        continue

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERROR: Could not load image")
        continue

    height, width = image.shape[:2]

    # Convert BGR → RGB for Matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # Read YOLO labels
    # --------------------------------------------------------

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Image size: {width} × {height}")
    print(f"Number of labels: {len(lines)}")

    # --------------------------------------------------------
    # Draw bounding boxes
    # --------------------------------------------------------

    for line in lines:

        values = line.strip().split()

        if len(values) != 5:
            print("WARNING: Invalid label:", line)
            continue

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # ----------------------------------------------------
        # Convert normalized YOLO coordinates
        # back to pixel coordinates
        # ----------------------------------------------------

        x_center *= width
        y_center *= height

        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)

        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Draw class name
        cv2.putText(
            image,
            "ship",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

        print(
            f"  Ship: "
            f"x1={x1}, y1={y1}, "
            f"x2={x2}, y2={y2}"
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"YOLO Labels: {image_path.name}")
    plt.axis("off")
    plt.show()


print("\n" + "=" * 60)
print("YOLO LABEL VERIFICATION COMPLETE")
print("=" * 60)
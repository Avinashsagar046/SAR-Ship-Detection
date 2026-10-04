import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "DATASET"

IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "annotations" / "test2017.json"


# ---------------------------------------------------------
# 2. Load the annotation JSON
# ---------------------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON loaded successfully!")


# ---------------------------------------------------------
# 3. Select the first image
# ---------------------------------------------------------

image_info = data["images"][0]

image_id = image_info["id"]
file_name = image_info["file_name"]

print("Image ID:", image_id)
print("Image file:", file_name)


# ---------------------------------------------------------
# 4. Find annotations belonging to this image
# ---------------------------------------------------------

image_annotations = []

for annotation in data["annotations"]:
    if annotation["image_id"] == image_id:
        image_annotations.append(annotation)

print("Number of ships:", len(image_annotations))


# ---------------------------------------------------------
# 5. Load the actual SAR image
# ---------------------------------------------------------

image_path = IMAGE_DIR / file_name

image = cv2.imread(str(image_path))

if image is None:
    print("ERROR: Image could not be loaded.")
    print("Path:", image_path)
    exit()

# OpenCV loads images as BGR.
# Matplotlib expects RGB.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# 6. Draw the ground-truth bounding boxes
# ---------------------------------------------------------

for annotation in image_annotations:

    x, y, width, height = annotation["bbox"]

    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2
    )


# ---------------------------------------------------------
# 7. Display the image
# ---------------------------------------------------------

plt.figure(figsize=(10, 10))

plt.imshow(image)
plt.title(f"HRSID Ground Truth - {file_name}")
plt.axis("off")

plt.show()
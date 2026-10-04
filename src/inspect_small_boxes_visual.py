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
# 2. Load JSON
# ---------------------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON loaded successfully!")


# ---------------------------------------------------------
# 3. Create image lookup
# ---------------------------------------------------------

image_lookup = {}

for image in data["images"]:
    image_lookup[image["id"]] = image


# ---------------------------------------------------------
# 4. Collect bounding-box information
# ---------------------------------------------------------

box_information = []

for annotation in data["annotations"]:

    x, y, width, height = annotation["bbox"]

    area = width * height

    image_id = annotation["image_id"]

    box_information.append({
        "image_id": image_id,
        "file_name": image_lookup[image_id]["file_name"],
        "bbox": (x, y, width, height),
        "area": area
    })


# ---------------------------------------------------------
# 5. Sort by bounding-box area
# ---------------------------------------------------------

box_information.sort(key=lambda item: item["area"])


# ---------------------------------------------------------
# 6. Display the 10 smallest boxes
# ---------------------------------------------------------

for i, box_info in enumerate(box_information[:10], start=1):

    file_name = box_info["file_name"]
    x, y, width, height = box_info["bbox"]

    image_path = IMAGE_DIR / file_name

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load:", image_path)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert coordinates to integers
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    # Draw bounding box
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(image)

    plt.title(
        f"Small Box #{i}\n"
        f"{file_name}\n"
        f"Size: {width} × {height} pixels"
    )

    plt.axis("off")
    plt.show()
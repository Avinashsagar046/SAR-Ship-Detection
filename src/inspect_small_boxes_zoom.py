import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "DATASET"
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "annotations" / "test2017.json"


# --------------------------------------------------
# 2. Load annotation JSON
# --------------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON loaded successfully!")


# --------------------------------------------------
# 3. Create image ID -> image information lookup
# --------------------------------------------------

image_lookup = {}

for image in data["images"]:
    image_lookup[image["id"]] = image


# --------------------------------------------------
# 4. Collect bounding-box information
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Sort boxes from smallest to largest
# --------------------------------------------------

box_information.sort(key=lambda item: item["area"])


# --------------------------------------------------
# 6. Inspect the 10 smallest boxes
# --------------------------------------------------

for i, box_info in enumerate(box_information[:10], start=1):

    file_name = box_info["file_name"]

    x, y, width, height = box_info["bbox"]

    image_path = IMAGE_DIR / file_name

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load:", image_path)
        continue

    # Convert OpenCV BGR format to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_height, image_width = image.shape[:2]

    # Convert coordinates to integers
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    # --------------------------------------------------
    # 7. Create a crop around the bounding box
    # --------------------------------------------------

    padding = 40

    crop_x1 = max(0, x - padding)
    crop_y1 = max(0, y - padding)

    crop_x2 = min(image_width, x + width + padding)
    crop_y2 = min(image_height, y + height + padding)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()


    # --------------------------------------------------
    # 8. Adjust bounding-box coordinates for the crop
    # --------------------------------------------------

    box_x1 = x - crop_x1
    box_y1 = y - crop_y1

    box_x2 = box_x1 + width
    box_y2 = box_y1 + height


    # --------------------------------------------------
    # 9. Draw the bounding box
    # --------------------------------------------------

    cv2.rectangle(
        crop,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (255, 0, 0),
        2
    )


    # --------------------------------------------------
    # 10. Display the zoomed crop
    # --------------------------------------------------

    plt.figure(figsize=(8, 8))

    plt.imshow(crop)

    plt.title(
        f"Small Box #{i}\n"
        f"{file_name}\n"
        f"Bounding Box: {width} × {height} pixels\n"
        f"Position: ({x}, {y})"
    )

    plt.axis("off")

    plt.show()


    # --------------------------------------------------
    # 11. Print information in terminal
    # --------------------------------------------------

    print("\n----------------------------------------")
    print(f"Small Box #{i}")
    print("Image:", file_name)
    print("Position:", (x, y))
    print("Width:", width)
    print("Height:", height)
    print("Area:", width * height)
    print("----------------------------------------")
import json
from pathlib import Path


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "DATASET"

ANNOTATION_FILE = DATASET_DIR / "annotations" / "test2017.json"


# ---------------------------------------------------------
# 2. Load the JSON file
# ---------------------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON loaded successfully!")


# ---------------------------------------------------------
# 3. Create a lookup table for image information
# ---------------------------------------------------------

image_lookup = {}

for image in data["images"]:
    image_lookup[image["id"]] = image


# ---------------------------------------------------------
# 4. Calculate the area of every bounding box
# ---------------------------------------------------------

box_information = []

for annotation in data["annotations"]:

    x, y, width, height = annotation["bbox"]

    area = width * height

    image_id = annotation["image_id"]

    box_information.append({
        "image_id": image_id,
        "file_name": image_lookup[image_id]["file_name"],
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "area": area
    })


# ---------------------------------------------------------
# 5. Sort boxes from smallest to largest
# ---------------------------------------------------------

box_information.sort(key=lambda item: item["area"])


# ---------------------------------------------------------
# 6. Display the 10 smallest boxes
# ---------------------------------------------------------

print("\n===== 10 SMALLEST BOUNDING BOXES =====")

for i, box in enumerate(box_information[:10], start=1):

    print(f"\n{i}.")
    print("Image:", box["file_name"])
    print("Image ID:", box["image_id"])
    print("Position:", (box["x"], box["y"]))
    print("Width:", box["width"])
    print("Height:", box["height"])
    print("Area:", box["area"])
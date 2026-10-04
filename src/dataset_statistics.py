import json
from pathlib import Path
from collections import Counter


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
# 3. Get images and annotations
# ---------------------------------------------------------

images = data["images"]
annotations = data["annotations"]


print("\n===== BASIC DATASET INFORMATION =====")

print("Number of images:", len(images))
print("Number of annotations:", len(annotations))


# ---------------------------------------------------------
# 4. Count ships in each image
# ---------------------------------------------------------

ship_count_per_image = Counter()

for annotation in annotations:

    image_id = annotation["image_id"]

    ship_count_per_image[image_id] += 1


# ---------------------------------------------------------
# 5. Calculate ship-per-image statistics
# ---------------------------------------------------------

ship_counts = list(ship_count_per_image.values())

print("\n===== SHIPS PER IMAGE =====")

print("Images containing ships:", len(ship_counts))
print("Minimum ships in an image:", min(ship_counts))
print("Maximum ships in an image:", max(ship_counts))

average_ships = sum(ship_counts) / len(ship_counts)

print("Average ships per image:", round(average_ships, 2))


# ---------------------------------------------------------
# 6. Count how many images have 1, 2, 3... ships
# ---------------------------------------------------------

distribution = Counter(ship_counts)

print("\n===== SHIP COUNT DISTRIBUTION =====")

for number_of_ships in sorted(distribution):

    number_of_images = distribution[number_of_ships]

    print(
        f"{number_of_ships} ship(s): "
        f"{number_of_images} image(s)"
    )


# ---------------------------------------------------------
# 7. Analyze bounding-box sizes
# ---------------------------------------------------------

widths = []
heights = []
areas = []

for annotation in annotations:

    x, y, width, height = annotation["bbox"]

    widths.append(width)
    heights.append(height)

    areas.append(width * height)


print("\n===== BOUNDING BOX STATISTICS =====")

print("Minimum width:", round(min(widths), 2))
print("Maximum width:", round(max(widths), 2))
print("Average width:", round(sum(widths) / len(widths), 2))

print("\nMinimum height:", round(min(heights), 2))
print("Maximum height:", round(max(heights), 2))
print("Average height:", round(sum(heights) / len(heights), 2))

print("\nMinimum bounding-box area:", round(min(areas), 2))
print("Maximum bounding-box area:", round(max(areas), 2))
print("Average bounding-box area:", round(sum(areas) / len(areas), 2))
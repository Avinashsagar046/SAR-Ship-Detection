import json
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "DATASET"

# HRSID test annotation file
ANNOTATION_FILE = DATASET_DIR / "annotations" / "test2017.json"

print("Project directory:", PROJECT_DIR)
print("Dataset directory:", DATASET_DIR)
print("Annotation file:", ANNOTATION_FILE)

# Check whether the annotation file exists
if not ANNOTATION_FILE.exists():
    print("\nERROR: Annotation file was not found.")
    print("Please check the path:", ANNOTATION_FILE)
    exit()

print("\nAnnotation file found!")

# Load JSON
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nJSON loaded successfully!")

# Display the main sections
print("\nAvailable sections:")
for key in data.keys():
    print(" -", key)

# Basic information
print("\nNumber of images:", len(data["images"]))
print("Number of annotations:", len(data["annotations"]))
print("Number of categories:", len(data["categories"]))

# Display categories
print("\nCategories:")
for category in data["categories"]:
    print(category)
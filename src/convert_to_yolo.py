import json
import shutil
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "DATASET"
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_DIR = DATASET_DIR / "annotations"

OUTPUT_DIR = BASE_DIR / "yolo_dataset"

TRAIN_JSON = ANNOTATION_DIR / "train2017.json"
TEST_JSON = ANNOTATION_DIR / "test2017.json"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for folder in [
    OUTPUT_DIR / "images" / "train",
    OUTPUT_DIR / "images" / "val",
    OUTPUT_DIR / "images" / "test",
    OUTPUT_DIR / "labels" / "train",
    OUTPUT_DIR / "labels" / "val",
    OUTPUT_DIR / "labels" / "test",
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# COCO → YOLO CONVERSION
# ============================================================

def convert_annotations(json_file, image_output_dir, label_output_dir):
    print(f"\nProcessing: {json_file.name}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Map image_id → annotations
    annotations_by_image = {}

    for ann in annotations:
        image_id = ann["image_id"]
        annotations_by_image.setdefault(image_id, []).append(ann)

    image_count = 0
    annotation_count = 0

    for image_info in images:

        image_id = image_info["id"]
        file_name = image_info["file_name"]

        width = image_info["width"]
        height = image_info["height"]

        source_image = IMAGE_DIR / file_name

        if not source_image.exists():
            print(f"WARNING: Image not found: {file_name}")
            continue

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = image_output_dir / file_name
        shutil.copy2(source_image, destination_image)

        # ----------------------------------------------------
        # Create YOLO label file
        # ----------------------------------------------------

        label_file = label_output_dir / f"{Path(file_name).stem}.txt"

        with open(label_file, "w", encoding="utf-8") as f:

            for ann in annotations_by_image.get(image_id, []):

                bbox = ann["bbox"]

                x = bbox[0]
                y = bbox[1]
                box_width = bbox[2]
                box_height = bbox[3]

                # COCO:
                # x, y = top-left corner
                #
                # YOLO:
                # x_center, y_center, width, height

                x_center = x + box_width / 2
                y_center = y + box_height / 2

                # Normalize to 0–1
                x_center /= width
                y_center /= height
                box_width /= width
                box_height /= height

                # HRSID has only one class:
                # ship = class 0
                class_id = 0

                f.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}\n"
                )

                annotation_count += 1

        image_count += 1

    print(f"Images processed: {image_count}")
    print(f"Annotations converted: {annotation_count}")


# ============================================================
# RUN CONVERSION
# ============================================================

# Train dataset
convert_annotations(
    TRAIN_JSON,
    OUTPUT_DIR / "images" / "train",
    OUTPUT_DIR / "labels" / "train"
)

# Test dataset
convert_annotations(
    TEST_JSON,
    OUTPUT_DIR / "images" / "test",
    OUTPUT_DIR / "labels" / "test"
)


# ============================================================
# CREATE DATASET YAML
# ============================================================

yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/train
test: images/test

names:
  0: ship
"""

yaml_file = OUTPUT_DIR / "dataset.yaml"

with open(yaml_file, "w", encoding="utf-8") as f:
    f.write(yaml_content)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("YOLO DATASET CONVERSION COMPLETE")
print("=" * 60)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Dataset YAML:     {yaml_file}")

print("\nDataset structure:")
print("yolo_dataset/")
print("├── images/")
print("│   ├── train/")
print("│   ├── val/")
print("│   └── test/")
print("├── labels/")
print("│   ├── train/")
print("│   ├── val/")
print("│   └── test/")
print("└── dataset.yaml")
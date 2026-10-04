import random
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_DIR = BASE_DIR / "yolo_dataset"

TRAIN_IMAGE_DIR = YOLO_DIR / "images" / "train"
TRAIN_LABEL_DIR = YOLO_DIR / "labels" / "train"

VAL_IMAGE_DIR = YOLO_DIR / "images" / "val"
VAL_LABEL_DIR = YOLO_DIR / "labels" / "val"

TEST_IMAGE_DIR = YOLO_DIR / "images" / "test"

RANDOM_SEED = 42
VALIDATION_RATIO = 0.20


# ============================================================
# MOVE CURRENT VALIDATION DATA BACK TO TRAIN
# ============================================================

print("=" * 60)
print("RESETTING TRAIN / VALIDATION SPLIT")
print("=" * 60)

val_images = list(VAL_IMAGE_DIR.glob("*"))

moved_back = 0

for image_path in val_images:

    if not image_path.is_file():
        continue

    destination = TRAIN_IMAGE_DIR / image_path.name

    shutil.move(str(image_path), str(destination))

    label_path = VAL_LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():

        shutil.move(
            str(label_path),
            str(TRAIN_LABEL_DIR / label_path.name)
        )

    moved_back += 1


print(f"Validation images moved back to train: {moved_back}")


# ============================================================
# COLLECT ALL TRAINING IMAGES
# ============================================================

image_extensions = {".png", ".jpg", ".jpeg"}

images = [
    p for p in TRAIN_IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

print(f"Total images available for split: {len(images)}")


# ============================================================
# SHUFFLE
# ============================================================

random.seed(RANDOM_SEED)
random.shuffle(images)


# ============================================================
# CALCULATE VALIDATION SIZE
# ============================================================

validation_count = int(len(images) * VALIDATION_RATIO)

validation_images = images[:validation_count]

print(f"Validation images: {validation_count}")
print(f"Training images:   {len(images) - validation_count}")


# ============================================================
# CREATE VALIDATION SPLIT
# ============================================================

moved_images = 0
moved_labels = 0

for image_path in validation_images:

    label_path = TRAIN_LABEL_DIR / f"{image_path.stem}.txt"

    shutil.move(
        str(image_path),
        str(VAL_IMAGE_DIR / image_path.name)
    )

    moved_images += 1

    if label_path.exists():

        shutil.move(
            str(label_path),
            str(VAL_LABEL_DIR / label_path.name)
        )

        moved_labels += 1


# ============================================================
# UPDATE YAML
# ============================================================

yaml_file = YOLO_DIR / "dataset.yaml"

yaml_content = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: ship
"""

with open(yaml_file, "w", encoding="utf-8") as f:
    f.write(yaml_content)


# ============================================================
# FINAL VERIFICATION
# ============================================================

train_count = len([
    p for p in TRAIN_IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
])

val_count = len([
    p for p in VAL_IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
])

test_count = len([
    p for p in TEST_IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
])


print("\n" + "=" * 60)
print("FINAL DATASET SPLIT")
print("=" * 60)

print(f"Training images:   {train_count}")
print(f"Validation images: {val_count}")
print(f"Test images:       {test_count}")

print("\nLabels moved:", moved_labels)

print("\nExpected:")
print("Training:   2914")
print("Validation:  728")
print("Test:       1962")

print("\nDataset YAML updated:")
print(yaml_file)
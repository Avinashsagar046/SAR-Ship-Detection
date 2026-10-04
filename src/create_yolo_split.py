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


# ============================================================
# SETTINGS
# ============================================================

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42


# ============================================================
# CREATE DIRECTORIES
# ============================================================

VAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
VAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND TRAINING IMAGES
# ============================================================

image_extensions = {".png", ".jpg", ".jpeg"}

images = [
    image for image in TRAIN_IMAGE_DIR.iterdir()
    if image.is_file() and image.suffix.lower() in image_extensions
]


print("=" * 60)
print("YOLO TRAIN / VALIDATION SPLIT")
print("=" * 60)

print(f"Total training images found: {len(images)}")


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


print(f"Validation images: {len(validation_images)}")
print(f"Training images remaining: {len(images) - len(validation_images)}")


# ============================================================
# MOVE VALIDATION IMAGES + LABELS
# ============================================================

moved_images = 0
moved_labels = 0

for image_path in validation_images:

    label_path = TRAIN_LABEL_DIR / f"{image_path.stem}.txt"

    # Move image
    shutil.move(
        str(image_path),
        str(VAL_IMAGE_DIR / image_path.name)
    )

    moved_images += 1

    # Move corresponding label
    if label_path.exists():

        shutil.move(
            str(label_path),
            str(VAL_LABEL_DIR / label_path.name)
        )

        moved_labels += 1

    else:
        print(f"WARNING: Label not found for {image_path.name}")


# ============================================================
# UPDATE DATASET YAML
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
# VERIFY COUNTS
# ============================================================

remaining_train_images = [
    p for p in TRAIN_IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

remaining_val_images = [
    p for p in VAL_IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]


print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)

print(f"Training images:   {len(remaining_train_images)}")
print(f"Validation images: {len(remaining_val_images)}")
print(f"Test images:       {len(list((YOLO_DIR / 'images' / 'test').glob('*')))}")

print(f"\nValidation images moved: {moved_images}")
print(f"Validation labels moved: {moved_labels}")

print(f"\nDataset YAML updated:")
print(yaml_file)

print("\nRandom seed:", RANDOM_SEED)
print("Validation ratio:", VALIDATION_RATIO)
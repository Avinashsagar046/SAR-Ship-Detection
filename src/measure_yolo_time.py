from pathlib import Path
from ultralytics import YOLO
import cv2
import time
import statistics

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\SAR_SHIP")

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo11n_HRSID"
    / "best.pt"
)

IMAGE_DIR = (
    PROJECT_DIR
    / "yolo_dataset"
    / "images"
    / "test"
)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 800
CONF_THRESHOLD = 0.40

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO11n INFERENCE-TIME MEASUREMENT")
print("=" * 70)

model = YOLO(str(MODEL_PATH))

# ============================================================
# IMAGES
# ============================================================

image_paths = sorted(
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg"))
)

print(f"\nImages: {len(image_paths)}")

if not image_paths:
    raise RuntimeError("No test images found.")

# ============================================================
# WARM-UP
# ============================================================

print("\nWarming up model...")

warmup_image = cv2.imread(
    str(image_paths[0])
)

model.predict(
    source=warmup_image,
    imgsz=IMG_SIZE,
    conf=CONF_THRESHOLD,
    verbose=False
)

print("Warm-up complete.")

# ============================================================
# TIMING
# ============================================================

print("\nStarting timing...")

times = []

total_start = time.perf_counter()

for index, image_path in enumerate(
    image_paths,
    start=1
):

    image = cv2.imread(
        str(image_path)
    )

    start = time.perf_counter()

    model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    end = time.perf_counter()

    times.append(
        end - start
    )

    if (
        index % 200 == 0
        or index == len(image_paths)
    ):

        print(
            f"Processed "
            f"{index}/{len(image_paths)}"
        )

total_end = time.perf_counter()

# ============================================================
# RESULTS
# ============================================================

total_time = total_end - total_start

average_time = statistics.mean(times)

median_time = statistics.median(times)

images_per_second = (
    len(image_paths) / total_time
)

print("\n" + "=" * 70)
print("YOLO11n TIMING RESULTS")
print("=" * 70)

print(
    f"Images              : {len(image_paths)}"
)

print(
    f"Total time          : {total_time:.2f} seconds"
)

print(
    f"Average/image       : {average_time:.4f} seconds"
)

print(
    f"Median/image        : {median_time:.4f} seconds"
)

print(
    f"Images/second       : {images_per_second:.2f}"
)

print(
    f"Average milliseconds : "
    f"{average_time * 1000:.2f} ms"
)

print("=" * 70)
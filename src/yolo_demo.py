from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n_HRSID" / "best.pt"
INPUT_DIR = PROJECT_ROOT / "demo" / "input"
OUTPUT_DIR = PROJECT_ROOT / "demo" / "output"


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load trained YOLO model
# --------------------------------------------------

print("Loading YOLO11n model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print()


# --------------------------------------------------
# Find input images
# --------------------------------------------------

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

images = [
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]


if not images:
    print("No images found in:")
    print(INPUT_DIR)
    print()
    print("Place a SAR image inside demo/input and run the program again.")
    raise SystemExit


# --------------------------------------------------
# Run detection
# --------------------------------------------------

for image_path in images:

    print(f"Processing: {image_path.name}")

    results = model.predict(
        source=str(image_path),
        imgsz=800,
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="detections",
        exist_ok=True,
        verbose=False
    )

    result = results[0]

    detected = 0 if result.boxes is None else len(result.boxes)

    print(f"Ships detected: {detected}")
    print()


print("========================================")
print("Detection completed.")
print(f"Results saved to: {OUTPUT_DIR / 'detections'}")
print("========================================")
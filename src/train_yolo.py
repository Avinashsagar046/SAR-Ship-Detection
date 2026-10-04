from ultralytics import YOLO
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_YAML = BASE_DIR / "yolo_dataset" / "dataset.yaml"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "yolo11n.pt"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("YOLO TRAINING")
print("=" * 60)

print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_YAML}")


model = YOLO(MODEL_NAME)


# ============================================================
# TRAIN
# ============================================================

results = model.train(

    data=str(DATASET_YAML),

    # Image size
    imgsz=800,

    # Training duration
    epochs=50,

    # Batch size
    batch=16,

    # Number of workers
    workers=4,

    # Reproducibility
    seed=42,

    # Save training results
    project=str(BASE_DIR / "results" / "yolo"),

    name="yolo11n_hrsID",

    # Save checkpoints
    save=True,

    # Use best model for final evaluation
    pretrained=True,

    # Early stopping
    patience=10
)


print("\n" + "=" * 60)
print("YOLO TRAINING COMPLETE")
print("=" * 60)
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "DATASET"

IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "annotations" / "test2017.json"


# --------------------------------------------------
# CA-CFAR parameters
# --------------------------------------------------

TRAINING_WINDOW_SIZE = 21
GUARD_WINDOW_SIZE = 5
PFA = 1e-2

MIN_COMPONENT_AREA = 3


# --------------------------------------------------
# Load annotations
# --------------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

image_info = data["images"][0]

image_id = image_info["id"]
file_name = image_info["file_name"]

image_path = IMAGE_DIR / file_name


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(image_path)

image_float = image.astype(np.float32)


# --------------------------------------------------
# CA-CFAR
# --------------------------------------------------

training_area = TRAINING_WINDOW_SIZE ** 2
guard_area = GUARD_WINDOW_SIZE ** 2

number_of_training_cells = training_area - guard_area

alpha = number_of_training_cells * (
    PFA ** (-1 / number_of_training_cells) - 1
)


outer_sum = cv2.boxFilter(
    image_float,
    ddepth=-1,
    ksize=(TRAINING_WINDOW_SIZE, TRAINING_WINDOW_SIZE),
    normalize=False
)

inner_sum = cv2.boxFilter(
    image_float,
    ddepth=-1,
    ksize=(GUARD_WINDOW_SIZE, GUARD_WINDOW_SIZE),
    normalize=False
)


ones = np.ones_like(image_float)

outer_count = cv2.boxFilter(
    ones,
    ddepth=-1,
    ksize=(TRAINING_WINDOW_SIZE, TRAINING_WINDOW_SIZE),
    normalize=False
)

inner_count = cv2.boxFilter(
    ones,
    ddepth=-1,
    ksize=(GUARD_WINDOW_SIZE, GUARD_WINDOW_SIZE),
    normalize=False
)

training_sum = outer_sum - inner_sum
training_count = outer_count - inner_count

background = np.divide(
    training_sum,
    training_count,
    out=np.zeros_like(training_sum),
    where=training_count > 0
)

threshold = alpha * background

detections = (image_float > threshold).astype(np.uint8)


# --------------------------------------------------
# Remove border
# --------------------------------------------------

half_window = TRAINING_WINDOW_SIZE // 2

detections[:half_window, :] = 0
detections[-half_window:, :] = 0
detections[:, :half_window] = 0
detections[:, -half_window:] = 0


# --------------------------------------------------
# Find connected components
# --------------------------------------------------

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    detections,
    connectivity=8
)


# --------------------------------------------------
# Create output image
# --------------------------------------------------

output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


# --------------------------------------------------
# Draw CFAR candidate boxes
# --------------------------------------------------

candidate_count = 0

for label in range(1, num_labels):

    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]

    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]

    area = stats[label, cv2.CC_STAT_AREA]

    if area < MIN_COMPONENT_AREA:
        continue

    candidate_count += 1

    cv2.rectangle(
        output,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2
    )


# --------------------------------------------------
# Draw ground-truth boxes
# --------------------------------------------------

ground_truth_count = 0

for annotation in data["annotations"]:

    if annotation["image_id"] != image_id:
        continue

    x, y, width, height = annotation["bbox"]

    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    cv2.rectangle(
        output,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    ground_truth_count += 1


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("Image:", file_name)

print("Ground-truth ships:", ground_truth_count)

print("CFAR candidate regions:", candidate_count)


# --------------------------------------------------
# Display
# --------------------------------------------------

plt.figure(figsize=(10, 10))

plt.imshow(output)

plt.title(
    "CA-CFAR Candidates + Ground Truth\n"
    "Red = CFAR | Green = Ground Truth"
)

plt.axis("off")

plt.show()
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "DATASET"
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "annotations" / "test2017.json"


# --------------------------------------------------
# 2. CA-CFAR parameters
# --------------------------------------------------

TRAINING_WINDOW_SIZE = 21
GUARD_WINDOW_SIZE = 5

PFA = 1e-2


# --------------------------------------------------
# 3. Load dataset annotation file
# --------------------------------------------------

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON loaded successfully!")


# --------------------------------------------------
# 4. Select one test image
# --------------------------------------------------

image_info = data["images"][0]

file_name = image_info["file_name"]

image_path = IMAGE_DIR / file_name

print("Image selected:", file_name)


# --------------------------------------------------
# 5. Load SAR image
# --------------------------------------------------

image = cv2.imread(
    str(image_path),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("Could not load image!")
    exit()

print("Image loaded successfully!")


# --------------------------------------------------
# 6. Convert image to floating point
# --------------------------------------------------

image_float = image.astype(np.float32)


# --------------------------------------------------
# 7. Calculate number of training cells
# --------------------------------------------------

training_area = TRAINING_WINDOW_SIZE ** 2

guard_area = GUARD_WINDOW_SIZE ** 2

number_of_training_cells = training_area - guard_area

print("\n===== CFAR INFORMATION =====")
print("Training window:", TRAINING_WINDOW_SIZE, "x",
      TRAINING_WINDOW_SIZE)

print("Guard window:", GUARD_WINDOW_SIZE, "x",
      GUARD_WINDOW_SIZE)

print("Training cells:", number_of_training_cells)

print("Desired Pfa:", PFA)


# --------------------------------------------------
# 8. Calculate CA-CFAR threshold factor
# --------------------------------------------------

alpha = number_of_training_cells * (
    PFA ** (-1 / number_of_training_cells) - 1
)

print("Threshold factor:", alpha)


# --------------------------------------------------
# 9. Calculate local background
# --------------------------------------------------

outer_sum = cv2.boxFilter(
    image_float,
    ddepth=-1,
    ksize=(TRAINING_WINDOW_SIZE, TRAINING_WINDOW_SIZE),
    normalize=False,
    borderType=cv2.BORDER_CONSTANT
)

inner_sum = cv2.boxFilter(
    image_float,
    ddepth=-1,
    ksize=(GUARD_WINDOW_SIZE, GUARD_WINDOW_SIZE),
    normalize=False,
    borderType=cv2.BORDER_CONSTANT
)


# --------------------------------------------------
# 10. Calculate how many valid pixels exist
# --------------------------------------------------

ones = np.ones_like(image_float)

outer_count = cv2.boxFilter(
    ones,
    ddepth=-1,
    ksize=(TRAINING_WINDOW_SIZE, TRAINING_WINDOW_SIZE),
    normalize=False,
    borderType=cv2.BORDER_CONSTANT
)

inner_count = cv2.boxFilter(
    ones,
    ddepth=-1,
    ksize=(GUARD_WINDOW_SIZE, GUARD_WINDOW_SIZE),
    normalize=False,
    borderType=cv2.BORDER_CONSTANT
)

training_count = outer_count - inner_count


# --------------------------------------------------
# 11. Estimate local background
# --------------------------------------------------

training_sum = outer_sum - inner_sum

background_estimate = np.divide(
    training_sum,
    training_count,
    out=np.zeros_like(training_sum),
    where=training_count > 0
)


# --------------------------------------------------
# 12. Calculate adaptive threshold
# --------------------------------------------------

threshold = alpha * background_estimate


# --------------------------------------------------
# 13. Compare CUT against threshold
# --------------------------------------------------
# --------------------------------------------------
# 13. Debug CFAR threshold
# --------------------------------------------------

print("\n===== CFAR THRESHOLD DEBUG =====")

print("Minimum background estimate:",
      round(float(np.min(background_estimate)), 2))

print("Maximum background estimate:",
      round(float(np.max(background_estimate)), 2))

print("Minimum threshold:",
      round(float(np.min(threshold)), 2))

print("Maximum threshold:",
      round(float(np.max(threshold)), 2))

ratio = np.divide(
    image_float,
    background_estimate,
    out=np.zeros_like(image_float),
    where=background_estimate > 0
)

print("Maximum image/background ratio:",
      round(float(np.max(ratio)), 2))

max_position = np.unravel_index(
    np.argmax(ratio),
    ratio.shape
)

print("Position of maximum ratio:",
      max_position)

print("Pixel value at that position:",
      float(image_float[max_position]))

print("Background at that position:",
      round(float(background_estimate[max_position]), 2))

print("Threshold at that position:",
      round(float(threshold[max_position]), 2))




detections = image_float > threshold


# --------------------------------------------------
# 14. Remove invalid border regions
# --------------------------------------------------

half_window = TRAINING_WINDOW_SIZE // 2

detections[:half_window, :] = False
detections[-half_window:, :] = False
detections[:, :half_window] = False
detections[:, -half_window:] = False


# --------------------------------------------------
# 15. Count detected pixels
# --------------------------------------------------

number_of_detected_pixels = np.count_nonzero(detections)

print("\n===== DETECTION RESULT =====")
print("Detected pixels:", number_of_detected_pixels)


# --------------------------------------------------
# 16. Display original SAR image
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(image, cmap="gray")

plt.title("Original SAR Image")

plt.axis("off")

plt.show()


# --------------------------------------------------
# 17. Display CA-CFAR detection map
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(detections, cmap="gray")

plt.title("CA-CFAR Detection Map")

plt.axis("off")

plt.show()
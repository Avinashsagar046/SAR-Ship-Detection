from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "DATASET"
IMAGE_DIR = DATASET_DIR / "images"


# --------------------------------------------------
# 2. Select one SAR image
# --------------------------------------------------

image_files = list(IMAGE_DIR.glob("*.png"))

if len(image_files) == 0:
    print("No PNG images found!")
    exit()

image_path = image_files[0]

print("Image selected:", image_path.name)


# --------------------------------------------------
# 3. Read the image as grayscale
# --------------------------------------------------

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Could not load image!")
    exit()

print("Image loaded successfully!")


# --------------------------------------------------
# 4. Get image information
# --------------------------------------------------

height, width = image.shape

print("\n===== IMAGE INFORMATION =====")
print("Width:", width)
print("Height:", height)
print("Data type:", image.dtype)
print("Number of pixels:", image.size)


# --------------------------------------------------
# 5. Calculate pixel statistics
# --------------------------------------------------

minimum = np.min(image)
maximum = np.max(image)
average = np.mean(image)
median = np.median(image)
standard_deviation = np.std(image)

print("\n===== PIXEL STATISTICS =====")
print("Minimum pixel value:", minimum)
print("Maximum pixel value:", maximum)
print("Average pixel value:", round(average, 2))
print("Median pixel value:", round(median, 2))
print("Standard deviation:", round(standard_deviation, 2))


# --------------------------------------------------
# 6. Display the SAR image
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(image, cmap="gray")

plt.title(f"SAR Image: {image_path.name}")

plt.axis("off")

plt.show()


# --------------------------------------------------
# 7. Display pixel-value distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(image.ravel(), bins=256)

plt.title("Pixel Intensity Distribution")

plt.xlabel("Pixel Intensity")

plt.ylabel("Number of Pixels")

plt.show()
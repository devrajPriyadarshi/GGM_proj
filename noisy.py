import cv2
import random

def apply_threshold_and_noise(image_path, threshold_value, noise_prop, output_path):
  """
  Loads an image, applies thresholding, adds salt-and-pepper noise, and saves it.

  Args:
      image_path: Path to the input image file.
      threshold_value: Threshold value for binarization.
      noise_prop: Proportion of pixels to add noise to (0.0 to 1.0).
      output_path: Path to save the processed image.
  """

  # Read the image in grayscale mode
  img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#   img = cv2.resize(img, (128,72))

  # Apply thresholding with binary inversion
  ret, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

  # Add salt-and-pepper noise
  rows, cols = thresh.shape
  num_noise_pixels = int(noise_prop * rows * cols)

  for _ in range(num_noise_pixels):
    i = random.randint(0, rows - 1)
    j = random.randint(0, cols - 1)
    if random.random() < 0.5:
      thresh[i, j] = 0  # Add black noise (salt)
    else:
      thresh[i, j] = 255  # Add white noise (pepper)

  # Save the processed image
  cv2.imwrite(output_path, thresh)

  print(f"Image loaded from: {image_path}")
  print(f"Thresholded and noisy image saved to: {output_path}")

# Example usage (replace with your image paths and desired values)
image_path = "GMM.png"
threshold_value = 127
noise_prop = 0.1  # Adjust this value to control noise amount (0.0 to 1.0)
output_path = "noisy.png"

apply_threshold_and_noise(image_path, threshold_value, noise_prop, output_path)

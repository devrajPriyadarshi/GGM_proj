import cv2

def threshold_image(image_path, threshold_value, output_path):
  """
  Loads an image, applies thresholding, and saves the result.

  Args:
      image_path: Path to the input image file.
      threshold_value: Threshold value for binarization.
      output_path: Path to save the thresholded image.
  """

  # Read the image in grayscale mode
  img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

  # Apply thresholding with binary inversion (pixels below threshold become white)
  ret, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

  # Save the thresholded image
  cv2.imwrite(output_path, thresh)

  print(f"Image loaded from: {image_path}")
  print(f"Thresholded image saved to: {output_path}")

# Example usage (replace with your image paths and desired threshold)
image_path = "GMM.png"
threshold_value = 127  # Adjust this value as needed
output_path = "thresholded_image.png"

threshold_image(image_path, threshold_value, output_path)

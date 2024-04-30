import cv2
import numpy as np

def calculate_psnr(original_image, noisy_image):
    assert original_image.shape == noisy_image.shape, "Original and noisy images must have the same dimensions."

    mse = np.mean((original_image - noisy_image) ** 2)
    max_pixel_value = np.max(original_image)
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr_value

noisy = "GIS.png"
print(noisy)
original_image = cv2.imread("thre.png", cv2.IMREAD_GRAYSCALE)
noisy_image = cv2.imread(noisy, cv2.IMREAD_GRAYSCALE)

psnr = calculate_psnr(original_image, noisy_image)

print("PSNR:", psnr, "dB")

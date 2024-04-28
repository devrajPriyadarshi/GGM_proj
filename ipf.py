import cv2
import numpy as np

def binary_image_to_mrf(image):
    """
    Converts a binary image to a binary Markov Random Field (MRF) representation.

    Args:
    - image: Binary input image (0 for black, 255 for white).

    Returns:
    - mrf: Binary MRF representation of the input image.
    """
    # Threshold the input image to obtain a binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    binary_image = binary_image / 255  # Normalize to 0 or 1

    # Define the neighborhood structure (4-neighborhood)
    neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Initialize the binary MRF
    mrf = np.copy(binary_image)

    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Compute the number of white neighbors
            num_white_neighbors = 0
            for dx, dy in neighborhood:
                ni, nj = i + dx, j + dy
                if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
                    num_white_neighbors += binary_image[ni, nj]

            # Apply the binary MRF constraint
            if num_white_neighbors > len(neighborhood) / 2:
                mrf[i, j] = 1
            else:
                mrf[i, j] = 0

    return mrf.astype(np.uint8)

def iterative_proportional_fitting(mrf, num_iterations=5):
    """
    Performs Iterative Proportional Fitting (IPF) on a binary MRF.

    Args:
    - mrf: Binary MRF representation of the input image.
    - num_iterations: Number of iterations for IPF.

    Returns:
    - denoised_image: Denoised image obtained using IPF.
    """
    # Initialize interaction potentials (random initialization)
    interaction_potentials = np.random.rand(*mrf.shape)

    # Define the neighborhood structure (4-neighborhood)
    neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Iterate IPF
    for _ in range(num_iterations):
        print("starting "+str(_))
        # Initialize expected counts
        expected_counts = np.zeros((2, 2))

        # Compute expected counts
        for i in range(mrf.shape[0]):
            for j in range(mrf.shape[1]):
                for dx, dy in neighborhood:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < mrf.shape[0] and 0 <= nj < mrf.shape[1]:
                        expected_counts[mrf[i, j], mrf[ni, nj]] += np.exp(interaction_potentials[i, j] + interaction_potentials[ni, nj])

        # Normalize expected counts
        print("done1")
        expected_counts /= np.sum(expected_counts)

        # Update interaction potentials
        for i in range(mrf.shape[0]):
            for j in range(mrf.shape[1]):
                for dx, dy in neighborhood:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < mrf.shape[0] and 0 <= nj < mrf.shape[1]:
                        interaction_potentials[i, j] += np.log(expected_counts[mrf[i, j], mrf[ni, nj]]) - np.log(np.exp(interaction_potentials[i, j]) + np.exp(interaction_potentials[ni, nj]))
        print("done2")

    # Denoise the image based on the interaction potentials
    denoised_image = np.zeros_like(mrf)
    for i in range(mrf.shape[0]):
        for j in range(mrf.shape[1]):
            # Compute the energy for each state
            energies = [0, 0]
            for dx, dy in neighborhood:
                ni, nj = i + dx, j + dy
                if 0 <= ni < mrf.shape[0] and 0 <= nj < mrf.shape[1]:
                    energies[0] += interaction_potentials[i, j] * mrf[ni, nj]
                    energies[1] += interaction_potentials[i, j] * (1 - mrf[ni, nj])
            # Choose the state with minimum energy
            denoised_image[i, j] = np.argmin(energies)
    print("done3")

    return denoised_image

# Load the input image using OpenCV
input_image = cv2.imread("noisy.png", cv2.IMREAD_GRAYSCALE)

# Convert the input image to a binary MRF representation
binary_mrf = binary_image_to_mrf(input_image)

# Perform Iterative Proportional Fitting (IPF) for image denoising
denoised_mrf = iterative_proportional_fitting(binary_mrf)

# Display the original and denoised images
# cv2.imshow("Original Image", input_image)
# cv2.imshow("Denoised Image", denoised_mrf * 255)  # Convert denoised image back to 0-255 range
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("IPF5.png", denoised_mrf*255)
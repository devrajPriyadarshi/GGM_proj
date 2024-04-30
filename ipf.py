import cv2
import numpy as np

def binary_image_to_mrf(image):

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    binary_image = binary_image / 255  # Normalize to 0 or 1

    # Define the neighborhood structure (4-neighborhood)
    neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    mrf = np.copy(binary_image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
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

    interaction_potentials = np.random.rand(*mrf.shape)
    neighborhood = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for _ in range(num_iterations):
        print("starting "+str(_))
        expected_counts = np.zeros((2, 2))

        for i in range(mrf.shape[0]):
            for j in range(mrf.shape[1]):
                for dx, dy in neighborhood:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < mrf.shape[0] and 0 <= nj < mrf.shape[1]:
                        expected_counts[mrf[i, j], mrf[ni, nj]] += np.exp(interaction_potentials[i, j] + interaction_potentials[ni, nj])

        print("done1")
        expected_counts /= np.sum(expected_counts)

        for i in range(mrf.shape[0]):
            for j in range(mrf.shape[1]):
                for dx, dy in neighborhood:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < mrf.shape[0] and 0 <= nj < mrf.shape[1]:
                        interaction_potentials[i, j] += np.log(expected_counts[mrf[i, j], mrf[ni, nj]]) - np.log(np.exp(interaction_potentials[i, j]) + np.exp(interaction_potentials[ni, nj]))
        print("done2")

    denoised_image = np.zeros_like(mrf)
    for i in range(mrf.shape[0]):
        for j in range(mrf.shape[1]):
            energies = [0, 0]
            for dx, dy in neighborhood:
                ni, nj = i + dx, j + dy
                if 0 <= ni < mrf.shape[0] and 0 <= nj < mrf.shape[1]:
                    energies[0] += interaction_potentials[i, j] * mrf[ni, nj]
                    energies[1] += interaction_potentials[i, j] * (1 - mrf[ni, nj])
            denoised_image[i, j] = np.argmin(energies)
    print("done3")

    return denoised_image

input_image = cv2.imread("noisy_thresholded_image.png", cv2.IMREAD_GRAYSCALE)

binary_mrf = binary_image_to_mrf(input_image)

denoised_mrf = iterative_proportional_fitting(binary_mrf)

cv2.imwrite("IPF.png", denoised_mrf*255)
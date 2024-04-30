import numpy as np
import itertools
import cv2
from tqdm import tqdm

def ising_energy(x, w, neighbors):

    energy = 0.0
    height, width = x.shape
    # print("Calculating energy")
    for i in range(height):
        for j in range(width):
            for k, (di, dj) in enumerate(neighbors):
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    energy += w[i, j, k] * x[i, j] * x[ni, nj]
    return -energy

def compute_marginals(w, neighbors, shape):
    height, width = shape
    marginals = np.zeros(shape)
    print("\nCalculating marginal distributions:")
    for x in itertools.product([0, 1], repeat=height * width):
        x = np.array(x).reshape(shape)
        energy = ising_energy(x, w, neighbors)
        prob = np.exp(-energy)
        marginals += x * prob
    marginals /= np.sum(marginals)
    return marginals

def gis(data, neighbors, num_iters=1, tol=1e-6):
    num_samples, height, width = data.shape
    num_neighbors = len(neighbors)
    w = np.zeros((height, width, num_neighbors))
    
    print("\nRunning GIS")
    for iter in range(num_iters):
        print("iteration "+str(iter)+":")
        marginals = compute_marginals(w, neighbors, (height, width))
        empirical_marginals = np.mean(data, axis=0)
        
        for i in range(height):
            for j in range(width):
                for k, (di, dj) in enumerate(neighbors):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        w[i, j, k] *= np.log(empirical_marginals[i, j] * empirical_marginals[ni, nj] / marginals[i, j] / marginals[ni, nj])
        
        diff = np.abs(marginals - empirical_marginals).sum()
        if diff < tol:
            break
    
    return w

def denoise_image(noisy_image, w, neighbors, num_iters=1):
    height, width = noisy_image.shape
    denoised_image = noisy_image.copy()
    
    print("\nDenoising Image")
    for _ in range(num_iters):
        print("iteration 1:")
        for i in range(height):
            for j in range(width):
                current_energy = ising_energy(denoised_image, w, neighbors)
                denoised_image[i, j] = 1 - denoised_image[i, j]
                new_energy = ising_energy(denoised_image, w, neighbors)
                if new_energy >= current_energy:
                    denoised_image[i, j] = 1 - denoised_image[i, j]
    
    return denoised_image

neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected neighborhood

# Load binary image data
data_ = cv2.imread("noisy.png", cv2.IMREAD_GRAYSCALE)
# data_ = cv2.resize(data_, dsize=(96,54),interpolation=cv2.INTER_NEAREST)
# cv2.imwrite("noisy_dw.png",data_)

data = np.array([data_])
print(data.shape)

# Run GIS for image denoising
w = gis(data, neighbors)

# Denoise the image 
denoised_image = denoise_image(data, w, neighbors)

cv2.imwrite("GIS_denoise.png",denoised_image)
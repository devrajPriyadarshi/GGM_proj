import numpy as np
from scipy.stats import norm
import cv2
np.random.seed(0)

def getEntropy(prob):
    return -np.mean(prob * np.log(prob + 1e-10) + (1 - prob) * np.log(1 - prob + 1e-10))

def VAriationalInference_denoise_quickly(data, prob, mu, width, height, episolon=1e-3, J=1, std=1):

    def matrixUpdate(mask):
        # left neighbors
        left_matrix = mu.copy()
        left_matrix[:,1:] = left_matrix[:,:-1]
        left_matrix[:,0] = mu[:,-1]
        # right neighbors
        right_matrix = mu.copy()
        right_matrix[:,:-1] = right_matrix[:,1:]
        right_matrix[:, -1] = mu[:, 0]
        # upper neighbors
        up_matrix = mu.copy()
        up_matrix[1:,:] = up_matrix[:-1,:]
        up_matrix[0,:] = mu[-1,:]
        # bottom neighbors
        bottom_matrix = mu.copy()
        bottom_matrix[:-1,:] = bottom_matrix[1:,:]
        bottom_matrix[-1,:] = mu[0,:]
        neighbours_mean_value_sum = left_matrix + right_matrix + up_matrix + bottom_matrix
        L_positive = np.log(norm.pdf(data, loc=1, scale=std) + 1e-10)
        L_negative = np.log(norm.pdf(data, loc=-1, scale=std) + 1e-10)
        positive = np.exp(J * neighbours_mean_value_sum + L_positive)
        negative = np.exp(-J * neighbours_mean_value_sum + L_negative)
        prob[mask] = (positive / (positive + negative))[mask]

        a_i = J * neighbours_mean_value_sum + 0.5 * (L_positive - L_negative)
        mu[mask] = np.tanh(a_i)[mask]

    lastEntropy = -np.inf
    while (1):
        # even's start state
        mask = np.full(width * height, False)
        mask[::2] = True
        matrixUpdate(mask.reshape([height, width]))
        # odd's start state
        mask = np.full(width * height, False)
        mask[1::2] = True
        matrixUpdate(mask.reshape([height, width]))

        currentEntropy = getEntropy(prob)
        if (np.abs(currentEntropy - lastEntropy) < episolon):
            break
        lastEntropy = currentEntropy

    data[mu >= 0] = 1
    data[mu < 0] = -1

    return data

def Process(data, episolon, J, std):

    width = data.shape[1]
    height = data.shape[0]
    data[data == 255] = 1
    # prob: probability of get 1. eg. data[0][0] = 0.8 means 80% will get 1, initially set them equal to the original image
    prob = data.copy()
    data[data == 0] = -1
    # initialize the mu as the same as data
    mu = data.copy()
    denoised_data = VAriationalInference_denoise_quickly(data, prob, mu, width, height, episolon=episolon, J=J, std=std)
    denoised_data[denoised_data == 1] = 255
    denoised_data[denoised_data == -1] = 0
    cv2.imwrite('VI.png', denoised_data.astype(np.uint8))

if __name__ == "__main__":
    image = cv2.imread("noisy_thresholded_image.png", cv2.IMREAD_GRAYSCALE)
    Process(image.astype(np.float32), episolon=1e-4, J=1, std=1)

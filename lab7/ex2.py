from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt

def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2)

def circle_mask(rows, cols, radius):
    center = [rows//2,cols//2]
    
    img = np.arange(rows).reshape(-1,1) * np.arange(cols).reshape(1,-1)
    for i in range(rows):
        for j in range(cols):
            if dist(i, j, center[0], center[1]) <= radius ** 2:
                img[i][j] = 1
            else:
                img[i][j] = 0
    
    return img

def compute_snr(original, processed):
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    
    signal_power = np.sum(orig ** 2)
    noise_power = np.sum((orig - proc) ** 2)
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

X = datasets.face(gray=True)
Y = np.fft.fftshift(np.fft.fft2(X))

def compress_image(radius):
    rows, cols = X.shape
    ifft_mask = circle_mask(rows, cols, radius)

    ifft_filtered = Y * ifft_mask

    compressed_img = np.abs(np.fft.ifft2(np.fft.fftshift(ifft_filtered)))

    # plt.imshow(ifft_mask, cmap='gray')
    # plt.show()

    # plt.imshow(compressed_img, cmap='gray')
    # plt.show()

    return (radius, compute_snr(X, compressed_img))

data = [compress_image(r) for r in range(0, 200, 10)]
for d in data:
    print(d, sep='\n')
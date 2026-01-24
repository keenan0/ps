from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt

# a
# Sinusoida oblica. Spectrul are doua puncte simetrice fata de origine
# N = 128
# n1 = np.arange(N).reshape(-1, 1) / N
# n2 = np.arange(N).reshape(1, -1) / N

# x = np.sin(2 * np.pi * n1 + 2 * np.pi * n2)

# y = np.fft.fft2(x)
# y_shift = np.fft.fftshift(y)
# y_mag = np.abs(y_shift)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(x, cmap='gray')
# plt.title('x(n1,n2)')

# plt.subplot(1, 2, 2)
# plt.imshow(y_mag, cmap='viridis')
# plt.title('FFT')
# plt.show()

# b
# Dungi verticale si orizontale. Spectrul are 4 puncte, doua pentru orizontala si doua pe verticala
# N = 128
# n1 = np.arange(N).reshape(-1, 1) / N
# n2 = np.arange(N).reshape(1, -1) / N

# x = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)

# y = np.fft.fft2(x)
# y_shift = np.fft.fftshift(y)
# y_mag = np.abs(y_shift)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(x, cmap='gray')
# plt.title('x(n1,n2)')

# plt.subplot(1, 2, 2)
# plt.imshow(y_mag, cmap='viridis')
# plt.title('FFT')
# plt.show()

# c
# In spectru sunt doar doua puncte simetrice pe o orizontala. Dungi orizontale
# N = 128
# Y = np.zeros((N, N), dtype=complex)
# Y[0, 5] = Y[0, N-5] = 1

# x_resed = np.fft.ifft2(Y)
# x_resed = np.real(x_resed)

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(np.fft.fftshift(Y)), cmap='viridis')
# plt.title('Y')

# plt.subplot(1, 2, 2)
# plt.imshow(x_resed, cmap='gray')
# plt.title('IFFT')

# plt.show()

# d
# Analog cu punctul e, doar ca imaginea este pe verticala
# N = 128
# Y = np.zeros((N, N), dtype=complex)
# Y[5, 0] = Y[N-5, 0] = 1

# x_resed = np.fft.ifft2(Y)
# x_resed = np.real(x_resed)

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(np.fft.fftshift(Y)), cmap='viridis')
# plt.title('Y')

# plt.subplot(1, 2, 2)
# plt.imshow(x_resed, cmap='gray')
# plt.title('IFFT')

# plt.show()

# e
# Componente pe diagonala. Imaginea este o sinusoida oblica
# N = 128
# Y = np.zeros((N, N), dtype=complex)
# Y[5, 5] = Y[N-5, N-5] = 1

# x_resed = np.fft.ifft2(Y)
# x_resed = np.real(x_resed)

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(np.fft.fftshift(Y)), cmap='viridis')
# plt.title('Y')

# plt.subplot(1, 2, 2)
# plt.imshow(x_resed, cmap='gray')
# plt.title('IFFT')

# plt.show()
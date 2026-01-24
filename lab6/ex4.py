import numpy as np
import matplotlib.pyplot as plt

N, B, d = 20, 0.4, 7
n = np.linspace(0, 1, N)
x = 5 * np.sin(2 * np.pi * n) + np.random.rand()

y = np.roll(x, d)

FFTX = np.fft.fft(x)
FFTY = np.fft.fft(y)

d1 = np.fft.ifft(FFTX * FFTY)
d2 = np.fft.ifft(FFTY / FFTX)

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.stem(x, label='Original Signal', linefmt='C0', markerfmt='C0')
plt.stem(y, label=f'Shifted Signal', linefmt='C1', markerfmt='C1')
plt.legend()

plt.subplot(3, 1, 2)
plt.stem(np.abs(d1), linefmt='C2')
plt.title("IFFT(FFT(x) * FFT(y))")

plt.subplot(3, 1, 3)
plt.stem(np.abs(d2), linefmt='C3')
plt.title("IFFT(FFT(y)) / FFT(X))")
plt.tight_layout()
plt.show()

# The second formula finds the correct d value
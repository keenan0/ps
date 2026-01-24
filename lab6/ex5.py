import numpy as np
import matplotlib.pyplot as plt

def rectangle_window(N):
    return np.ones(N)

def hanning_window(N):
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / N))

f = 100
Nw = 200
fs = 800

t = np.arange(Nw) / fs
sig = np.sin(2 * np.pi * f * t)

w_rect = rectangle_window(Nw)
w_hann = hanning_window(Nw)

sig_rect = sig * w_rect
sig_hann = sig * w_hann

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, sig)
plt.title("Original Signal")
plt.grid(True) 

plt.subplot(3, 1, 2)
plt.plot(t, sig * w_rect, color='red')
plt.title("Rectangular Window")
plt.grid(True) 

plt.subplot(3, 1, 3)
plt.plot(t, sig * w_hann, color='green')
plt.title("Hanning Window")
plt.grid(True)

plt.tight_layout()
plt.show()
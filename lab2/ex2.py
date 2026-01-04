import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

def sin_sig(amp, freq, time, phi = 0):
    return amp * np.sin(2 * np.pi * freq * time + phi)

def cos_sig(amp, freq, time, phi = 0):
    return amp * np.cos(2 * np.pi * freq * time + phi)

# Exercise 2.1.
fig, axs = plt.subplots(1)
fig.suptitle("Exercitiul 2")

timestamps = [0, 0.005]
BUCKETS = 1600
FREQ = 300
AMP = 1

phi_data = [0, 10, 100, -200]
x_data = np.linspace(timestamps[0], timestamps[1], BUCKETS)
y_data = [sin_sig(AMP, FREQ, x_data, phi) for phi in phi_data]

for y in y_data:
    axs.plot(x_data, y)
    axs.grid(True, linestyle='--', alpha=0.6)

plt.show()
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

def sin_sig(amp, freq, time, phi = 0):
    return amp * np.sin(2 * np.pi * freq * time + phi)

def cos_sig(amp, freq, time, phi = 0):
    return amp * np.cos(2 * np.pi * freq * time + phi)

# Exercise 2.2.
timestamps = [0, 0.005]
BUCKETS = 1600
FREQ = 300
AMP = 1

fig, axs = plt.subplots(4)
fig.suptitle("Exercitiul 2 + Zgomot")

x_data = np.linspace(timestamps[0], timestamps[1], BUCKETS)
y_data = sin_sig(AMP, FREQ, x_data, 0)

snr = [0.1, 1, 10, 100]

y_data_copy = [y_data.copy() for i in range(4)]
noise_data = np.random.normal(size=BUCKETS)

x_norm = np.linalg.norm(y_data)
for i, y in enumerate(y_data_copy):
    z_norm = np.linalg.norm(noise_data)

    x_sq = x_norm ** 2
    z_sq = z_norm ** 2

    print(x_sq, snr[i], z_sq)

    gamma = np.sqrt(x_sq / snr[i] / z_sq)

    print(gamma)
    print(noise_data)

    y_data_copy[i] = y_data_copy[i] + gamma * noise_data

for i, ax in enumerate(axs):
    ax.plot(x_data, y_data_copy[i])

plt.show()
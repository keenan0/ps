import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

def sin_sig(amp, freq, time, phi = 0):
    return amp * np.sin(2 * np.pi * freq * time + phi)

def cos_sig(amp, freq, time, phi = 0):
    return amp * np.cos(2 * np.pi * freq * time + phi)

fig, axs = plt.subplots(2)
fig.suptitle("Sinus | Cosinus Suprapuse")

timestamps = [0, 10e-3]
FREQ = 300
AMP = 2
BUCKETS = 1600

x_data = np.linspace(timestamps[0], timestamps[1], BUCKETS)
y_data_sin = sin_sig(AMP, FREQ, x_data)
y_data_cos = cos_sig(AMP, FREQ, x_data, (-1) * np.pi / 2)

axs[0].plot(x_data, y_data_sin)
axs[1].plot(x_data, y_data_cos)

plt.show()
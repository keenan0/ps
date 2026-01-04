from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def sin_sig(t, A, freq, phi):
    return A * np.sin(np.pi * 2 * freq * t + phi)

def sawtooth(t, A, freq, phi):
    T = 1 / freq
    return  A * ((t / T) - np.floor(t / T + 0.5))
    
fig, axs = plt.subplots(3)
fig.suptitle("Exercitiul 4")

time_axis = np.linspace(0, 0.005, 44100)
sin_s = sin_sig(time_axis, 1, 400, 0)
saw_s = sawtooth(time_axis, 1, 800, 0)

axs[0].plot(time_axis, sin_s)
axs[1].plot(time_axis, saw_s)
axs[2].plot(time_axis, sin_s + saw_s)

plt.show()
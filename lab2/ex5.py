from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def sin_sig(t, A, freq, phi):
    return A * np.sin(np.pi * 2 * freq * t + phi)

fig, axs = plt.subplots(3)
fig.suptitle("Exercitiul 4")

time_axis_0 = np.linspace(0, 0.005, 44100)
time_axis_1 = np.linspace(0, 2 * 0.005, 2 * 44100)
sin_s = [sin_sig(time_axis_0, 1, 400, 0),sin_sig(time_axis_0, 1, 900, 0)]

comb_s = np.concatenate((sin_s[0],sin_s[1]))

axs[0].plot(time_axis_0, sin_s[0])
axs[1].plot(time_axis_0, sin_s[1])
axs[2].plot(time_axis_1, comb_s)

comb_s = comb_s / np.max(np.abs(comb_s)) * 32767
wavfile.write("ex5.wav", 44100, (comb_s).astype(np.int16))
print(comb_s)

plt.show()
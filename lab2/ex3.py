from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

# Exercise 3.1.
def x(t: float) -> float:
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t: float) -> float:
    return np.cos(280 * np.pi * t - np.pi / 3)

def z(t: float) -> float:
    return np.cos(120 * np.pi * t + np.pi / 3)

# Parameters
FUNCTION_MAX_BINS = int(10e4)

n_waves = 3
wave_functions = [x, y, z]
timestamps = [0, 0.03]
step = 0.0005
sample_freq = 200

AUDIO_FREQ = 44100

fig, ax = plt.subplots(4)
fig.suptitle("Exercitiul 3")

def sample_wave(buckets: int, timestamps: List[int], freq: int, wave_function: Callable[[float,int],float], index: int) -> None:
    x_axis = np.linspace(timestamps[0], timestamps[1], buckets)
    ax[index].plot(x_axis, wave_function(x_axis, freq))

    return wave_function(x_axis, freq)
    
a_sig = sample_wave(AUDIO_FREQ, [0,1], 400, lambda t, freq: np.sin(2 * np.pi * freq *  t), 0)
b_sig = sample_wave(AUDIO_FREQ, [0,3], 800, lambda t, freq: np.sin(2 * np.pi * freq *  t), 1)
c_sig = sample_wave(AUDIO_FREQ, [0,0.1], 240, lambda t, freq: np.mod(t - np.floor(t), 1 / freq), 2)
d_sig = sample_wave(AUDIO_FREQ, [0,0.1], 300, lambda t, freq: np.sign(np.sin(2 * np.pi * t * freq)), 3)

a_sig = a_sig / np.max(np.abs(a_sig)) * 32767
wavfile.write("a.wav", AUDIO_FREQ, (a_sig).astype(np.int16))
print(a_sig)

rate, x = wavfile.read("a.wav")
print(rate, x)

# sd.play(x, rate)
# sd.wait()  # Nu merge pe wsl : raise PortAudioError(f'Error querying device {device}')

plt.show()
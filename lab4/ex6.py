from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# a
samplerate, data = wavfile.read('./aeiou.wav')

if len(data.shape) > 1:
    data = data[:, 0]

# b
N = len(data)
window_size = int(0.01 * N)
hop_size = window_size // 2

groups = []
for i in range(0, N - window_size, hop_size):
    current_group = data[i : i + window_size]
    groups.append(current_group)

# c & d
fft_data = []
for group in groups:
    f_res = np.fft.fft(group)
    
    magnitude = np.abs(f_res)
    
    half_magnitude = magnitude[:window_size // 2]
    
    fft_data.append(half_magnitude)

# e
S = np.array(fft_data).T
S = S[10:, 10:]

plt.figure(figsize=(12, 6))

plt.imshow(np.log10(S), aspect='auto', origin='lower', cmap='magma')

plt.title("Spectogram FFT")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(label="Intensity")
plt.show()
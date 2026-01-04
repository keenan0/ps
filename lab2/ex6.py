import numpy as np
import matplotlib.pyplot as plt

def sin_sig(t, amp, freq, phi):
    return amp * np.sin(2 * np.pi * t * freq + phi)

fs = 200 
timestamps = [0, 0.5]
time_data = np.linspace(timestamps[0], timestamps[1], int(fs * (timestamps[1] - timestamps[0])))

fig, axs = plt.subplots(3, 1, figsize=(8, 6))
fig.suptitle("Exercitiul 6 - Diverse frecvente")

# a
f = fs / 2
y = sin_sig(time_data, 1, f, 0)
axs[0].plot(time_data, y, 'o-')
axs[0].grid(True)

# b
f = fs / 4
y = sin_sig(time_data, 1, f, 0)
axs[1].plot(time_data, y, 'o-')
axs[1].grid(True)

# c
f = 0
y = sin_sig(time_data, 1, f, 0)
axs[2].plot(time_data, y, 'o-')
axs[2].grid(True)

plt.show()
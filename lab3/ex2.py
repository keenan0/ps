import numpy as np
import matplotlib.pyplot as plt
import math

def my_sin(t, Amp, freq, phi):
    return Amp * np.sin(2 * np.pi * t * freq + phi)

FREQ = 14
fs = 1000
time = np.linspace(0, 1, fs)
n = np.arange(fs)

omega = FREQ / fs

xn = my_sin(time, 6, FREQ, 0)
yn = xn * (np.exp(-2j * np.pi * n * omega)) 

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(time, xn)
axs[0].set_title("Semnal Simplu")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Amp")

axs[1].plot(yn.real, yn.imag)
axs[1].set_title("Infasuratoarea Semnalului")
axs[1].set_xlabel("Re")
axs[1].set_ylabel("Im")
axs[1].spines['left'].set_position('zero')
axs[1].spines['bottom'].set_position('zero')
axs[1].spines['right'].set_color('none')
axs[1].spines['top'].set_color('none')
axs[1].set_xlim([-6, 6])
axs[1].set_ylim([-6, 6])
axs[1].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

omegas_hz = [5, 10, 14, 20]
omegas = [f / fs for f in omegas_hz]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for ax, w in zip(axs.flatten(), omegas):
    zw = xn * np.exp(-2j * np.pi * w * n)
    ax.plot(zw.real, zw.imag)
    ax.set_title(f"omega = {w * fs}")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
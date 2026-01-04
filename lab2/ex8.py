import numpy as np
import matplotlib.pyplot as plt

def pade(alpha):
    return (alpha - ((7 * alpha ** 3) / 60)) / (1 + ((alpha ** 2)/(20)))

alpha = [- np.pi / 2, np.pi / 2]
freq = 100

fig, axs = plt.subplots(2)

time = np.linspace(alpha[0], alpha[1], int(freq * (alpha[1] - alpha[0])))
sin_sig = np.sin(time)
pade_sig = pade(time)
error_sig = np.abs(sin_sig - time)
error_sig_pade = np.abs(pade_sig - sin_sig)

axs[0].set_title("α vs sin(α) vs pade(α)")
axs[0].plot(time, time, label="α")
axs[0].plot(time, sin_sig, label="sin(α)")
axs[0].plot(time, pade_sig, label="pade(α)")

axs[1].set_title("Eroarea dintre semnale")
axs[1].plot(time, error_sig, label="err sin(α)")
axs[1].plot(time, error_sig_pade, label="err pade(α)")
axs[1].set_yscale("log")

for ax in axs:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
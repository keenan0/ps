import numpy as np
import matplotlib.pyplot as plt

f0 = 20
f1 = 100
f2 = 260

fs = max(f0, f1, f2) * 2 + 10
t_final = 0.1

t_cont = np.linspace(0, t_final, 1000)
t_esant = np.arange(0, t_final, 1/fs)

sig0 = np.sin(2 * np.pi * f0 * t_cont)
sig1 = np.sin(2 * np.pi * f1 * t_cont)
sig2 = np.sin(2 * np.pi * f2 * t_cont)

esc_0 = np.sin(2 * np.pi * f0 * t_esant)
esc_1 = np.sin(2 * np.pi * f1 * t_esant)
esc_2 = np.sin(2 * np.pi * f2 * t_esant)

plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(t_cont, sig0, color='blue', label=f'f={f0}Hz')
plt.plot(t_esant, esc_0, color='blue', alpha=0.5)
plt.stem(t_esant, esc_0, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title(f'fs={fs}Hz')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(t_cont, sig1, color='red', label=f'f={f1}Hz')
plt.plot(t_esant, esc_1, color='red', alpha=0.5)
plt.stem(t_esant, esc_1, linefmt='r-', markerfmt='ro', basefmt=' ')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(t_cont, sig2, color='green', label=f'f={f2}Hz')
plt.plot(t_esant, esc_2, color='green', alpha=0.5)
plt.stem(t_esant, esc_2, linefmt='g-', markerfmt='go', basefmt=' ')

plt.legend()

plt.xlabel('Time')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

f0 = 20
fs = 80

t_continuu = np.linspace(0, 0.2, 1000)
t_esantionat = np.arange(0, 0.2, 1/fs)

f1 = f0 + 1 * fs
f2 = f0 + 3 * fs

sig0 = np.sin(2 * np.pi * f0 * t_continuu)
sig1 = np.sin(2 * np.pi * f1 * t_continuu)
sig2 = np.sin(2 * np.pi * f2 * t_continuu) 

esantioane = np.sin(2 * np.pi * f0 * t_esantionat)

plt.figure(figsize=(12, 6))

plt.plot(t_continuu, sig0, label=f'f={f0}Hz', color='blue', alpha=0.5)
plt.plot(t_continuu, sig1, label=f'f={f1}Hz', color='red', alpha=0.5)
plt.plot(t_continuu, sig2, label=f'f={f2}Hz', color='green', alpha=0.5)

plt.scatter(t_esantionat, esantioane, color='black', s=50, zorder=5, label='Eșantioane (fs=80Hz)')
plt.stem(t_esantionat, esantioane, linefmt='k-', markerfmt='ko', basefmt=' ')

plt.title(f'Aliasing')
plt.xlabel('Time')
plt.ylabel('Amp')
plt.grid(True)
plt.legend()
plt.show()
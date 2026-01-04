import numpy as np
import matplotlib.pyplot as plt

tp = 2 * np.pi
def my_sig(t, f1, f2, f3, f4):
    return np.cos(tp * f1 * t) + 3 * np.cos(tp * f2 * t) + np.sin(tp * f3 * t) + np.sin (tp * f4 * t)

def my_dft(xn):
    N = len(xn)
    X = np.zeros(N, dtype=complex)
    
    for m in range(N):
        for n in range(N):
            X[m] += xn[n] * np.exp(-2j * np.pi * m * n / N)
    
    return X

fs = 800
N = 256
Ts = 1 / fs

n = np.arange(N)
t = n * Ts

freqs = [5, 100, 260, 440]

xn = my_sig(t, *freqs)
Xn = my_dft(xn)

analysed_freqs = n * fs / N
mag = np.abs(Xn)

print("Analysed frequencies: ")
for i in range(N):
    print(analysed_freqs[i], "  -  ", mag[i], end='\n')

plt.stem(analysed_freqs, mag)
plt.show()

# fs = 100
# T = 1
# time = np.linspace(0, T, int(fs*T))
# xn = my_sig(time, 5, 20, 25, 44)

# N = len(xn)
# n = np.arange(N)

# ws = np.linspace(0, fs, fs * 2)
# Xn = np.array([np.sum(xn * np.exp(-1j * 2 * np.pi * w * n / fs)) for w in ws])
# mag = np.abs(Xn)

# plt.figure(figsize=(8,4))
# plt.plot(ws, mag)
# plt.xlim(0, fs/2)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("|X(f)|")
# plt.title("DFT Abs")


# plt.show()
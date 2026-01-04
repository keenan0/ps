import numpy as np
import matplotlib.pyplot as plt

import time

def my_sig(t, f1, f2, f3, f4):
    return np.cos(2 * np.pi * f1 * t) + 3 * np.cos(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f3 * t) + np.sin (2 * np.pi * f4 * t)

def my_dft(xn):
    N = len(xn)
    X = np.zeros(N, dtype=complex)
    
    for m in range(N):
        for n in range(N):
            X[m] += xn[n] * np.exp(-2j * np.pi * m * n / N)
    
    return X

def my_fft(x):
    n = len(x)
    
    if n <= 1:
        return x
    
    even = my_fft(x[0::2])
    odd = my_fft(x[1::2])
    
    res = [0] * n
    for k in range(n // 2):
        chosen_x = np.exp(-2j * np.pi * k / n)
        
        res[k] = even[k] + chosen_x * odd[k]
        res[k + n // 2] = even[k] - chosen_x * odd[k]
        
    return res

fs = 800
Ts = 1 / fs
freqs = [5, 100, 260, 340]

N_values = [128, 256, 512, 1024, 2048, 4096, 8192]

times_dft = []
times_my_fft = []
times_numpy_fft = []

for N in N_values:
    n = np.arange(N)
    t = n * Ts
    xn = my_sig(t, *freqs)
    
    start_time = time.time()
    _ = my_dft(xn)
    end_time = time.time()
    times_dft.append(end_time - start_time)
    print(f"  N={N}: DFT took {end_time - start_time:.6f}s")
    
    start_time = time.time()
    _ = my_fft(xn)
    end_time = time.time()
    times_my_fft.append(end_time - start_time)
    print(f"  N={N}: My FFT took {end_time - start_time:.6f}s")

    start_time = time.time()
    _ = np.fft.fft(xn)
    end_time = time.time()
    times_numpy_fft.append(end_time - start_time)
    print(f"  N={N}: NumPy FFT took {end_time - start_time:.6f}s")
    print("-" * 30)

plt.figure(figsize=(12, 7))
plt.plot(N_values, times_dft, marker='o', label='My DFT', color='red')
plt.plot(N_values, times_my_fft, marker='x', label='My FFT', color='green')
plt.plot(N_values, times_numpy_fft, marker='s', label='NumPy FFT', color='blue')

plt.xscale('log', base=2)
plt.yscale('log')

plt.xticks(N_values, [str(N) for N in N_values])
plt.legend()
plt.grid(True)
plt.show()
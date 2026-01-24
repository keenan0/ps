import numpy as np

def multiply(P, Q):
    n = len(P)
    m = len(Q)
    
    res = [0] * (n + m - 1)

    for i in range(n):
        for j in range(m):
            res[i + j] += p[i] * q[j]
            
    return np.array(res)

N = 4
p = np.random.randint(-10, 10, N + 1)
q = np.random.randint(-10, 10, N + 1)

print(f"P: {p}")
print(f"Q: {q}")

r_direct = multiply(p, q)

size_fft = len(p) + len(q) - 1

P_freq = np.fft.fft(p, n=size_fft)
Q_freq = np.fft.fft(q, n=size_fft)

R_freq = P_freq * Q_freq
r_fft = np.fft.ifft(R_freq)

r_fft_final = np.real(r_fft).round().astype(int)

print(f"Polynomial Multiplication: {r_direct}")
print(f"FFT -> Multiply -> IFFT:   {r_fft_final}")
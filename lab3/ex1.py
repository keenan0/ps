import numpy as np
import matplotlib.pyplot as plt

def euler(m, k, n):
    return (np.cos(2 * np.pi * m * k / n) - 1j * np.sin(2 * np.pi * m * k / n))

def dft_component(m, xn):
    sum = 0
    n = len(xn)
    
    for k,x in enumerate(xn):
        sum += x * euler(m, k, n)

    return sum

N = 8
freq = 2

time = np.linspace(0, 1, N)
sig = np.sin(2 * np.pi * time * freq)

F = np.zeros((N,N), dtype=complex)

for m in range(N):
    for k in range(N):
        F[m][k] = euler(m, k, N)

fig, axs = plt.subplots(N, 2, figsize=(8, 10))

for m in range(N):
    axs[m, 0].plot(np.real(F[m, :]))
    axs[m, 1].plot(np.imag(F[m, :]))

axs[0, 0].set_title("Re(F)")
axs[0, 1].set_title("Im(F)")

plt.tight_layout()
plt.show()

FH_F = np.conjugate(F.T) @ F
I = np.eye(N)
is_unitary = np.allclose(FH_F, N * I)
print("F unitara: ", is_unitary)
print("FH * F - N·I =", np.linalg.norm(FH_F - N * I))
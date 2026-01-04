import numpy as np
import matplotlib.pyplot as plt

# a
N = 1000
t = np.arange(N)

a, b, c = 0.00004, -0.025, 0.005
trend = a * t * t + b * t + c

a1, a2 = 4, 9
phi1, phi2 = 0.1, -0.5
f1, f2 = 80, 130
season = a1 * np.sin(2 * np.pi * f1 * t + phi1) + a2 * np.sin(2 * np.pi * f2 * t + phi2)

residuals = np.random.normal(0, 0.5, N)

observed = trend + season + residuals

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axs[0].plot(t, trend, color='crimson', lw=2)
axs[0].set_title('Trend')
axs[0].grid(True)

axs[1].plot(t, season, color='forestgreen', lw=2)
axs[1].set_title('Season')
axs[1].grid(True)

axs[2].plot(t, residuals, color='royalblue', lw=1)
axs[2].set_title('Noise')
axs[2].grid(True)

axs[3].plot(t, observed, color='purple', lw=1.5)
axs[3].set_title('Observed Time Series')
axs[3].set_xlabel('Time')
axs[3].grid(True)

#plt.tight_layout()
#plt.show()

# b 

P, M = 2, 6

y = np.arange(10)

Y = np.column_stack((y[1:-1], y[:-2])) 
y_target = y[2:]

Gamma = Y.T @ Y
xsol = np.linalg.inv(Gamma) @ Y.T @ y_target

print("Y:\n", Y)
print("Gamma:\n", Gamma)
print("xsol:", xsol)

def autocorrelation(y, p, normalize=True):
    N = len(y)
    gamma = np.zeros(p+1)
    for t in range(p+1):
        s = 0
        for j in range(t, N):
            s += y[j] * y[j-t]
        gamma[t] = s / (N - t)
    if normalize:
        gamma /= gamma[0]     
    return gamma

withFlip = np.convolve(y, y[::-1])
withFlip = withFlip[len(y)-1:]

print(autocorrelation(y, 2))
print(withFlip / withFlip[0])
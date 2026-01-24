import numpy as np
import matplotlib.pyplot as plt

# a
N = 1000
t = np.arange(N)

a, b, c = 0.00004, -0.025, 0.005
trend = a * t * t + b * t + c

a1, a2 = 4, 9
phi1, phi2 = 0.1, -0.5
f1, f2 = 1/15, 1/26
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

plt.tight_layout()
plt.show()

# b 
P, M = 2, 6

y = np.arange(10)
Y = np.column_stack((y[1:-1], y[:-2])) 
# y_target = y[2:]

# Gamma = Y.T @ Y
# xsol = np.linalg.inv(Gamma) @ Y.T @ y_target

# print("Y:\n", Y)
# print("Gamma:\n", Gamma)
# print("xsol:", xsol)

def autocorrelation(y, p, normalize=True):
    N = len(y)
    gamma = np.zeros(p+1)
    for t in range(p+1):
        s = 0
        for j in range(t, N):
            s += y[j] * y[j-t]
        gamma[t] = s / (N)
    if normalize:
        gamma /= gamma[0]     
    return gamma

withFlip = np.convolve(y, y[::-1])
withFlip = withFlip[len(y)-1:]

print(autocorrelation(y, 2))
print(withFlip / withFlip[0])

y_centered = observed - np.mean(observed)
N = len(y_centered)

autocorr_full = np.correlate(y_centered, y_centered, mode='full')
autocorr = autocorr_full[N-1:]
autocorr_norm = autocorr / autocorr[0]

plt.figure(figsize=(10, 4))
plt.stem(autocorr_norm[:200])
plt.title("Autocorrelation Vector")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid(True)
plt.show()

# c
p = 50 

Gamma = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        Gamma[i, j] = autocorr[abs(i - j)]

gamma = autocorr[1:p+1]
x_star = np.linalg.solve(Gamma, gamma)

predictions = np.zeros(N)
for t in range(p, N):
    predictions[t] = np.dot(y_centered[t-p:t][::-1], x_star)

predictions_final = predictions + np.mean(observed)

plt.figure(figsize=(12, 5))
plt.plot(observed, label='Original', alpha=0.6)
plt.plot(range(p, N), predictions_final[p:], label='AR Prediction', color='red', alpha=0.5)
plt.legend()
plt.show()

#d out: For m = 1, best p = 935
p_vals = [p for p in range(0, 1000, 5)]
mse_errors = []

for p_test in p_vals:
    G_test = np.zeros((p_test, p_test))
    for i in range(p_test):
        for j in range(p_test):
            G_test[i, j] = autocorr[abs(i - j)]
            
    g_test = autocorr[1:p_test+1]
    w_test = np.linalg.solve(G_test, g_test)
    
    y_pred_test = np.zeros(N)
    for t in range(p_test, N):
        y_pred_test[t] = np.dot(y_centered[t-p_test:t][::-1], w_test)
    
    Error = np.mean((y_centered[p_test:] - y_pred_test[p_test:])**2)
    mse_errors.append(Error)

plt.figure(figsize=(8, 5))
plt.plot(p_vals, mse_errors, marker='o', linestyle='--', color='green')
plt.title("Hyperparameter Tuning")
plt.xlabel("p")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

best_p = p_vals[np.argmin(mse_errors)]
print(f"\nFor m = 1, best p = {best_p}")
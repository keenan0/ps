import numpy as np
import matplotlib.pyplot as plt

N = 500
t = np.arange(N)

a, b, c = 0.00004, -0.025, 0.005

a1, a2 = 4, 9
phi1, phi2 = 0.1, -0.5
f1, f2 = 1/20, 1/36

trend = a * t * t + b * t + c
season = a1 * np.sin(2 * np.pi * f1 * t + phi1) + a2 * np.sin(2 * np.pi * f2 * t + phi2)
residuals = np.random.normal(0, 0.5, N)

for i in range(N):
    season[i] *= i / N
    residuals[i] *= i / N

observed = trend + season + residuals

# Moving Average
def moving_agerage(q = 5):
    theta = np.linspace(0.5, 0.1, q) 
    mu = np.mean(observed) 

    N = len(observed)
    errors = np.zeros(N)
    ma_predictions = np.zeros(N)

    ma_predictions[0] = mu
    errors[0] = observed[0] - ma_predictions[0]

    for i in range(1, N):
        weighted_errors = 0
        for j in range(1, min(i, q) + 1):
            weighted_errors += theta[j-1] * errors[i-j]
        
        ma_predictions[i] = mu + weighted_errors
        errors[i] = observed[i] - ma_predictions[i]

    return ma_predictions

plt.figure(figsize=(12, 6))
plt.plot(observed, color="red", alpha=0.4, label="Original")

for q in range(1, 20, 4):
    plt.plot(moving_agerage(q), label=f"MA(q={q})")
    
plt.title("Moving Average")
plt.legend()
plt.show()
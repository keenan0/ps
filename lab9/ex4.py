import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

N = 2000
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

limit = 10
p_range = range(0, limit) 
q_range = range(0, limit)

best_aic = float("inf")
best_order = None
best_model = None

print("Start searching for parameters")

for p in p_range:
    for q in q_range:
        try:
            model = ARIMA(observed, order=(p, 0, q))
            results = model.fit()
            print(f"Tried values p={p} q={q}")

            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, q)
                best_model = results
        except:
            continue

print(f"\nResults")
print(f"p={best_order[0]}, q={best_order[1]}")

predictions = best_model.predict(start=0, end=len(observed)-1)

plt.figure(figsize=(12, 6))
plt.plot(observed, label='Original', alpha=0.5)
plt.plot(predictions, label=f'ARMA({best_order[0]},{best_order[1]})', color='red', linestyle='--')
plt.title("Best params for ARIMA model")
plt.legend()
plt.show()
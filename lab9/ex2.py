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

# Mediere Exponentiala
alpha = 0.5

x = observed
s = np.zeros(N)
for t in range(1,N):
    s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]

a = x[1:-1] - s[:-2]
b = x[2:] - s[:-2]

plt.figure(figsize=(12, 6))
plt.plot(s, color="green", label=f"s | alpha = {alpha}")
computed_alpha = np.dot(a, b) / np.linalg.norm(a)**2
computed_alpha = np.clip(computed_alpha, 0, 1)
print(f"Best alpha found = {computed_alpha}")

s = np.zeros(N)
for t in range(1,N):
    s[t] = computed_alpha * x[t] + (1 - computed_alpha) * s[t - 1]

plt.title("Mediere Exponentiala")
plt.plot(s, color="orange", label=f"s | alpha = {computed_alpha}")
plt.plot(x, color="red", alpha=0.4)
plt.tight_layout()
plt.show()

# Mediere Exponentiala Dubla
x = observed

def double_exponential_smoothing(series, alpha, beta):
    n = len(series)
    s = np.zeros(n)
    b = np.zeros(n)
    s[0] = series[0]
    b[0] = series[1] - series[0]
    
    for t in range(1, n):
        s[t] = alpha * series[t] + (1 - alpha) * (s[t-1] + b[t-1])
        b[t] = beta * (s[t] - s[t-1]) + (1 - beta) * b[t-1]
    return s, b

best_mse = float("inf")
best_alpha, best_beta = 0, 0
best_x_hat = []

for a in np.linspace(0.01, 1, 20):
    for b in np.linspace(0.01, 1, 20):
        s, trend = double_exponential_smoothing(observed, a, b)
        
        x_hat = s[:-1] + trend[:-1]
        x_real = observed[1:]
        
        mse = np.mean((x_hat - x_real)**2)
        
        if mse < best_mse:
            best_x_hat = x_hat
            best_mse = mse
            best_alpha, best_beta = a, b

print(f"Best Alpha: {best_alpha:.2f}")
print(f"Best Beta: {best_beta:.2f}")

plt.figure(figsize=(12, 6))
plt.title("Mediere Exponentiala Dubla")
plt.plot(x, color="red", alpha=0.3, label="Original")
plt.plot(best_x_hat, color="blue", label="x_hat")
plt.legend()
plt.show()

# Mediere Exponentiala Tripla

def triple_exponential_smoothing(series, L, alpha, beta, gamma):
    N = len(series)
    s = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    
    x_hat = np.zeros(N)
    
    s[0] = series[0]
    b[0] = series[1] - series[0]
    
    for i in range(L):
        c[i] = series[i] - s[0]
        
    for t in range(1, N):
        if t >= L:
            s[t] = alpha * (series[t] - c[t-L]) + (1 - alpha) * (s[t-1] + b[t-1])
            b[t] = beta * (s[t] - s[t-1]) + (1 - beta) * b[t-1]
            c[t] = gamma * (series[t] - s[t] - b[t-1]) + (1 - gamma) * c[t-L]
            
            if t < N - 1:
                x_hat[t+1] = s[t] + b[t] + c[t-L+1]
        else:
            s[t] = alpha * series[t] + (1 - alpha) * (s[t-1] + b[t-1])
            b[t] = beta * (s[t] - s[t-1]) + (1 - beta) * b[t-1]
            c[t] = series[t] - s[t]
            if t < N - 1:
                x_hat[t+1] = s[t] + b[t]
                
    return x_hat

L = 20 
best_mse = float("inf")
best_params = (0, 0, 0)
best_x_hat = []

for a in np.linspace(0.1, 0.9, 5):
    for b in np.linspace(0.1, 0.9, 5):
        for g in np.linspace(0.1, 0.9, 5):
            predictions = triple_exponential_smoothing(observed, L, a, b, g)
            
            mse = np.mean((observed[L:] - predictions[L:])**2)
            
            if mse < best_mse:
                best_x_hat = predictions
                best_mse = mse
                best_params = (a, b, g)

print(f"Alpha={best_params[0]}, Beta={best_params[1]}, Gamma={best_params[2]}")
plt.figure(figsize=(12, 6))
plt.title("Mediere Exponentiala Tripla")
plt.plot(x, color="red", alpha=0.3, label="Original")
plt.plot(best_x_hat, color="blue", label="x_hat")
plt.legend()
plt.show()
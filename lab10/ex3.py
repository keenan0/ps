import numpy as np
from sklearn.linear_model import Lasso

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

p = 20
m = N - p

Y = np.zeros((m, p))
for i in range(p):
    Y[:, i] = observed[p-1-i : -1-i]
y_target = observed[p:]

x_ls = np.linalg.lstsq(Y, y_target, rcond=None)[0]
print("Classic Solution", x_ls)

# Greedy Implementation
def greedy_ar(Y, y, k_taps):
    m, p = Y.shape
    selected_indices = []
    residual = y.copy()
    x_greedy = np.zeros(p)

    for _ in range(k_taps):
        correlations = np.abs(Y.T @ residual)
        
        for idx in selected_indices:
            correlations[idx] = -1
        
        best_idx = np.argmax(correlations)
        selected_indices.append(best_idx)

        Y_subset = Y[:, selected_indices]
        x_subset = np.linalg.lstsq(Y_subset, y, rcond=None)[0]

        residual = y - Y_subset @ x_subset
        
        x_greedy[selected_indices] = x_subset

    return x_greedy

x_sparse_greedy = greedy_ar(Y, y_target, k_taps=5)
print("Greedy\n", x_sparse_greedy)

# L1 Normalization
from sklearn.linear_model import Lasso

for lambda_reg in np.linspace(0.01, 1.0, 10): 
    lasso_model = Lasso(alpha=lambda_reg)
    lasso_model.fit(Y, y_target)

    x_sparse_l1 = lasso_model.coef_
    print(f"lambda={lambda_reg}\n", x_sparse_l1)

# EX 5
from ex4 import get_roots

def check_stationarity(ar_coefficients):
    p = len(ar_coefficients)
    
    coeffs_companion = -ar_coefficients[::-1] 
    
    roots = get_roots(coeffs_companion)
    magnitudes = np.abs(roots)
    
    is_stationary = np.all(magnitudes < 1)
    max_magnitude = np.max(magnitudes)
    
    return is_stationary, max_magnitude

models = {
    "Least Squares": x_ls,
    "Greedy Sparse": x_sparse_greedy,
    "Lasso (L1)": x_sparse_l1
}

for name, coeffs in models.items():
    stationary, m_val = check_stationarity(coeffs)
    status = "STATIONARY" if stationary else "NOT STATIONARY"
    print(f"- Model {name:15}: {status} (Max Magnitude: {m_val:.4f})")
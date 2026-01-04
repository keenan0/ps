import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')
data = df.iloc[:, 1].astype(float).to_numpy()

print(data)

def AR(p, m, data_in): 
    N = len(data_in)
    if m + p > N:
        return

    print(f"Starting AR({p})")
    t = np.arange(N)
    y = data_in

    Y = np.column_stack([y[i:N-p+i] for i in range(p)])
    y_target = y[p:]
    Gamma = Y.T @ Y
    xsol = np.linalg.inv(Gamma) @ Y.T @ y_target

    print("Y:\n", Y)
    print("Gamma:\n", Gamma)
    print("xsol:", xsol)

    y_hat = list(y[:p])

    for t in range(p, len(y)):
        next_val = sum(xsol[j] * y_hat[t-j-1] for j in range(p))
        y_hat.append(next_val)

    y_hat = np.array(y_hat)

    plt.plot(y)
    plt.plot(y_hat, 'r')

    plt.show()

# print(data)
# AR(2, 10, data)
# AR(3, 10, data)
# AR(4, 10, data)
# AR(5, 10, data)

from statsmodels.tsa.ar_model import AutoReg

y = data 
model = AutoReg(y, lags=2)
model_fit = model.fit()

n_forecast = 5
left_offset = 0
a = len(y) - left_offset
b = a+n_forecast-1
y_pred = model_fit.predict(start=a, end=b)

plt.plot(y)
plt.plot(range(a,b+1), y_pred, 'r--')
plt.show()
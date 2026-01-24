import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

observed = df['Close'].values
N = len(observed)

y_centered = observed - np.mean(observed)
acf_full = np.correlate(y_centered, y_centered, mode='full')
acf = acf_full[N-1:]
acf_norm = acf / acf[0]

plt.figure(figsize=(10, 4))
plt.stem(acf_norm) 
plt.title("Autocorrelation NVIDIA")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.show()
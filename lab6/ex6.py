import numpy as np
import matplotlib.pyplot as plt

# a
start, three_days = 1952, 24 * 3
x = np.genfromtxt('./Train.csv', delimiter=',')
x = x[start:start + three_days,2]

Ts = 1 / 8
t = np.arange(72) * Ts
s = np.mean(x) * np.sin(2 * np.pi * t) + np.mean(x) // 2

# plt.plot(x)
# plt.plot(s)
# plt.show()

# b
# w_vals = [5, 9, 13, 17]
# for i,w in enumerate(w_vals): 
#     y = np.convolve(x, np.ones(w), 'valid') / w

#     plt.subplot(4, 1, i + 1)
#     plt.title(f'w = {w}')
#     plt.plot(y)
#     plt.grid(True)

# plt.tight_layout()
#plt.show()

# c

'''
I was thinking of dividing a day into 3 parts relevant for the traffic: morning, afternoon and evening

This means that we could keep frequencies up to 8 samples / hour and cut off faster frequencies that induce noise into the general trend.  

We can choose a cutoff frequency f_s = 1 / 8. Converted into Hz 
'''

# d
from scipy import signal

# Wn = 0.25
# N = 10
# rp = 5 # When rp = 0, butterworth = chebyshev

# b_but, a_but = signal.butter(N, Wn, btype='low')
# b_cheb, a_cheb = signal.cheby1(N, rp, Wn, btype='low')

# x_but = signal.filtfilt(b_but, a_but, x)
# x_cheb = signal.filtfilt(b_cheb, a_cheb, x)

# plt.plot(x, label='Original Signal', alpha=0.3)
# plt.plot(x_but, label='Butterworth (Flat)')
# plt.plot(x_cheb, label='Chebyshev (Sharp)')
# plt.legend()
# plt.show()

# e
# The Butterworth filter seems to visually describe the trend better than the Chebyshev.

# f 
Wn = 0.25
N = 10
butterworth = [signal.butter(n, Wn, btype='low') for n in range(1, N, 2)]

for i,data in enumerate(butterworth):
    b, a = data
    x_but = signal.filtfilt(b,a,x)
    plt.plot(x_but, label=f'n = {i * 2 + 1}')

plt.plot(x, label='Original Signal', alpha=0.3)
plt.title("Butterworth")
plt.legend()
plt.show()

rp = 2
cheby = [signal.cheby1(n, rp, Wn, btype='low') for n in range(1, N, 2)]

for i,data in enumerate(cheby):
    b, a = data
    x_cheb = signal.filtfilt(b,a,x)
    plt.plot(x_cheb, label=f'n = {i * 2 + 1}')

plt.plot(x, label='Original Signal', alpha=0.3)
plt.title("Chebyshev")
plt.legend()
plt.show()
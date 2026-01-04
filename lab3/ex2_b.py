import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

def my_sin(t, Amp, freq, phi):
    return Amp * np.sin(2 * np.pi * t * freq + phi)

fs = 1000
N = 1000        
time = np.linspace(0, 1, N)
n = np.arange(N)

freq = 14      
xn = my_sin(time, 3, freq, 2)

omega = 24 / fs    
yn = xn * np.exp(-2j * np.pi * omega * n)

x = yn.real
y = yn.imag

d = np.sqrt(x*x + y*y)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, ax = plt.subplots(figsize=(6, 6))

lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(d.min(), d.max()))
lc.set_array(d)
lc.set_linewidth(2)
ax.add_collection(lc)

ax.set_xlim([x.min() - 1, x.max() + 1])
ax.set_ylim([y.min() - 1, y.max() + 1])
ax.set_aspect('equal')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_aspect('equal', adjustable='box')
plt.colorbar(lc, label="Distance to origin")

plt.title("Static Coloring")
plt.show()

fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.set_xlim([x.min() - 1, x.max() + 1])
ax2.set_ylim([y.min() - 1, y.max() + 1])
ax2.set_aspect('equal')
ax2.spines["left"].set_position('zero')
ax2.spines["bottom"].set_position('zero')
ax2.spines["right"].set_color('none')
ax2.spines["top"].set_color('none')
ax2.set_aspect('equal', adjustable='box')

lc2 = LineCollection([], cmap='viridis', norm=plt.Normalize(d.min(), d.max()))
ax2.add_collection(lc2)

point, = ax2.plot([], [], 'ro')

def update(i):
    lc2.set_segments(segments[:i])
    lc2.set_array(d[:i])
    point.set_data([x[i]], [y[i]])
    return lc2, point

anim = FuncAnimation(fig2, update, frames=N, interval=100, blit=True)

plt.title("Animation")
plt.show()

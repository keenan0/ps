from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt

# Exercise 3.1.
def x(t: float) -> float:
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t: float) -> float:
    return np.cos(280 * np.pi * t - np.pi / 3)

def z(t: float) -> float:
    return np.cos(120 * np.pi * t + np.pi / 3)

# Parameters
FUNCTION_MAX_BINS = int(10e4)

n_waves = 3
wave_functions = [x, y, z]
timestamps = [0, 0.03]
step = 0.0005
sample_freq = 200

fig, axs = plt.subplots(n_waves)
fig.suptitle("Exercise 3.1.")
x_axis = np.linspace(timestamps[0], timestamps[1], int(timestamps[1] / step))
time_axis = np.linspace(timestamps[0], timestamps[1], int(sample_freq * timestamps[1]))

for i in range(n_waves):
    axs[i].stem(time_axis, wave_functions[i](time_axis), linefmt="C1--")
    axs[i].plot(x_axis, wave_functions[i](x_axis))

plt.show()

def sample_wave(buckets: int, timestamps: List[int], freq: int, wave_function: Callable[[float,int],float], title: str) -> None:
    fig, ax = plt.subplots(1)
    fig.suptitle(title)
    x_axis = np.linspace(timestamps[0], timestamps[1], buckets)
    ax.plot(x_axis, wave_function(x_axis, freq))
    ax.stem(x_axis, wave_function(x_axis, freq), linefmt="C1--")
    
    plt.show()

# sample_wave(1600, [0,1], 400, lambda t, freq: np.sin(2 * np.pi * freq *  t), "Exercise 3.2.a.")
# sample_wave(int(10e2), [0,3], 800, lambda t, freq: np.sin(2 * np.pi * freq *  t), "Exercise 3.2.b.")
# sample_wave(int(10e2), [0,0.1], 240, lambda t, freq: np.mod(t - np.floor(t), 1 / freq), "Exercise 3.2.c.")
# sample_wave(int(10e2), [0,0.1], 300, lambda t, freq: np.sign(np.sin(2 * np.pi * t * freq)), "Exercise 3.2.d.")

# Exercise 3.2.e.
random_2d_data = np.random.rand(128,128)
# plt.imshow(random_2d_data)
# plt.show()

# Exercise 3.2.f.
def neighbour_mean(arr: np.ndarray) -> np.ndarray:
    res = np.zeros((128,128))

    for i in range(1, arr.shape[0] - 1):
        for j in range(1, arr.shape[1] - 1):
            neighbours = [arr[i+1,j],arr[i+1,j+1],arr[i+1,j-1],arr[i,j+1],arr[i,j-1],arr[i-1,j],arr[i-1,j+1],arr[i-1,j-1]]
            res[i][j] = np.mean(neighbours)

    return res

random_2d_data = np.random.rand(128,128)

fig, axs = plt.subplots(2)
for ax in axs:
    ax.imshow(neighbour_mean(random_2d_data))
    ax.axis("off")

plt.show()

# Exercise 3.3.a.
# 1 / 2000 s

# Exercise 3.3.b.
# 3600 * 2000 * 0.5 = 3.600.000 B = 3.6 MB
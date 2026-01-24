import numpy as np
import matplotlib.pyplot as plt

def conv(random = True):
    N = 100
    if random:
        x = np.random.rand(N)
    else:
        x = np.zeros(N)
        x[40:60] = 1

    semnale = [x]
    for i in range(3):
        x_urmator = np.convolve(semnale[-1], semnale[0], mode='full')
        semnale.append(x_urmator)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    for i, s in enumerate(semnale):
        axes[i].plot(s, color='orange', lw=2)
        axes[i].set_title(f"x <- x * x [{i} {'time' if i == 1 else 'times'}]")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

conv()
conv(False)

# After applying multiple convolutions, the signal looks like a normal distribution
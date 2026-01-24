import numpy as np
import matplotlib.pyplot as plt

left, right = -3, 3
t_cont = np.linspace(left, right, 1024)

def plot_sinc(B = 1):
    x_sinc = np.sinc(t_cont * B) ** 2

    plt.title("sinc() ^ 2")
    plt.plot(t_cont, x_sinc)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fs = [1, 1.5, 2, 4]

    for i, f in enumerate(fs):
        Ts = 1.0 / f

        t_disc = np.arange(np.ceil(left / Ts), np.floor(right / Ts)) * Ts
        xn = np.sinc(t_disc * B) ** 2

        x_hat = np.zeros_like(t_cont)
        for j in range(len(t_disc)):
            x_hat += xn[j] * np.sinc((t_cont - t_disc[j]) / Ts)

        plt.subplot(2, 2, i + 1)
        plt.grid(True)
        plt.plot(t_cont, x_sinc, color='red')
        plt.plot(t_cont, x_hat, '--', color="green", alpha=0.6)
        plt.stem(t_disc, xn)

    plt.tight_layout()
    plt.show()

plot_sinc()
plot_sinc(20) # Big peak in the center
plot_sinc(0.05) # Acts like zooming inside the sinc with B = 1
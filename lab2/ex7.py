import numpy as np
import matplotlib.pyplot as plt

def sin_sig(t, amp, freq, phi):
    return amp * np.sin(2 * np.pi * freq * t + phi)

FREQ_ESANTIONARE = 1000 
FRECVENTA_SEMNAL = 50    
DURATA = 0.1

t = np.linspace(0, DURATA, int(FREQ_ESANTIONARE * DURATA), endpoint=False)
x = sin_sig(t, 1, FRECVENTA_SEMNAL, 0)

factor_decimare = 4
x_decimat = x[::factor_decimare]
t_decimat = t[::factor_decimare]

x_decimat2 = x[1::factor_decimare]
t_decimat2 = t[1::factor_decimare]

plt.figure(figsize=(10, 5))
plt.plot(t, x)
plt.plot(t_decimat, x_decimat, 'o-')
plt.plot(t_decimat2, x_decimat2, 'x--')
plt.title('Exercițiul 7 – Decimarea semnalului')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()

# Cand pornim de la al doilea element din vector, se da sample la valorile peak de pe semnalul original
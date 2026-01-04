# 18288

'''
(a) Care este frecventa de esantionare a semnalului din Train.csv?
Datele masoara: "Numarul de masini care
trec printr-o intersectie a fost masurat din ora in ora"

R: Ceea ce inseamna ca frecventa este f = 1 sample / 1 ora
fs = 1 / 3600 s = 0.00027Hz

(b) Ce interval de timp acopera esantioanele din fisier?
1 sample ... 1 ora 
18288 samples ... 18288 ore

18288 / 24 = 762 zile

(c) Considerand ca semnalul a fost esantionat corect (fara aliere) si optim care este frecventa maxima prezenta in semnal?

B = fs / 2 = 0.000138

(d) Utilizati functia np.fft.fft(x) pentru a calcula transformata Fourier a semnalului s, i afisati grafic modulul transformatei.
Deoarece valorile pe care le veti calcula sunt in Hz, este important
sa definiti corect frecventa de esantionare (astfel incat valorile de
frecvente pe care le obtineti utilizand ultima secventa de cod din
Sectiunea 3 sa aiba interpretare corecta din punct de vedere fizic).

'''
import numpy as np
import matplotlib.pyplot as plt

# 1 cycle / hour
fs = 1

x = np.genfromtxt('./Train.csv', delimiter=',')
print(x)
x = x[1:,2]
print(x)

x = x - np.mean(x)
N = len(x)

X = np.fft.fft(x)
X = np.abs(X/N)
X = X[:N//2]

f = fs * np.linspace(0, N//2, N//2) / N
plt.plot(f[5:], X[5:])
plt.title("Traffic")
plt.xlabel("Frequency (cycles/hour)")
plt.ylabel("Amp")
plt.show()

'''

(e) Prezinta acest semnal o componenta continua? Daca da, eliminati-o.
Daca nu, specificati cum ati determinat.

R: La frecventa 0Hz, semnalul are un peak foarte mare comparativ cu celelalte peak-uri. (140 vs 30 20 sau 15), ceea ce sugereaza un semnal cu o componenta continua. Pentru a elimina aceasta componenta, am centrat semnalul in 0 scazand media semnalului.

(f) Care sunt frecventele principale continute in semnal, asa cum apar ele
in transformata Fourier? Mai exact, determinati primele 4 cele mai
mari valori ale modulului transformatei si specificati caror frecvente
(in Hz) le corespund. Caror fenomene periodice din semnal se asociaza fiecare?

'''

offset = 10 
X_search = X[offset:]
f_search = f[offset:]

top_indices = np.argsort(X_search)[-8:]

top_frequencies = f_search[top_indices]
top_magnitudes = X_search[top_indices]

for i in range(8):
    freq_hz = top_frequencies[i]

    period_hours = 1 / freq_hz
    print(f"f = {freq_hz:.5f} ore^-1, T = {period_hours:.2f} ore, Mag = {top_magnitudes[i]:.2f}")

'''

Cele mai mari 4 frecvente sunt:
f = 0.04167 ore^-1, T = 24.00 ore, Mag = 27.10
f = 0.00596 ore^-1, T = 167.76 ore, Mag = 19.00
f = 0.04173 ore^-1, T = 23.97 ore, Mag = 13.95
f = 0.08334 ore^-1, T = 12.00 ore, Mag = 12.23
f = 0.04162 ore^-1, T = 24.03 ore, Mag = 11.59
f = 0.00591 ore^-1, T = 169.31 ore, Mag = 9.99
f = 0.01192 ore^-1, T = 83.88 ore, Mag = 8.49
f = 0.00055 ore^-1, T = 1828.60 ore, Mag = 7.91

Cele mai importante frecvente sunt:
1/24 - spike zilnic 
1/12 - spike dimieata si seara
1/168 - spike saptamanal
1/84 - odata la 3.5 zile se schimba ceva

(g) Incepand de la esantion ales de voi mai mare decat 1000, vizualizati,
pe un grafic separat, o luna de trafic. Alegeti esantionul de start
astfel incat reprezentarea sa inceapa intr-o zi de luni.

'''

S = 3410
month = 24 * 30

monthly_traffic = x[S : S + month]

plt.figure(figsize=(15, 5))
plt.plot(monthly_traffic, label='Trafic pe o lună')

for week in range(0, month, 168):
    plt.axvline(x=week, color='r', linestyle='--', alpha=0.5)

plt.title(f"One Month of Traffic (S = {S})")
plt.xlabel("Hours")
plt.ylabel("Cars")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

'''

(h) Nu se cunoaste data la care a inceput masurarea acestui semnal.
Concepeti o metoda (descrieti in cuvinte) prin care sa determinati,
doar analizand semnalul in timp, aceasta data. Comentati ce neajunsuri ar putea avea solutia propusa si care sunt factorii de care depinde acuratetea ei.

Daca presupunem ca in weekend traficul este mai redus, am putea cauta perioade de 48 de ore in care traficul este semnificativ mai slab si sa presupunem ca ce este dupa va reprezenta un inceput sa saptamana. Am aplicat aceeasi logica si la exercitiul anterior pentru a determina o zi de luni.

Ca si neajunsuri, datele ar putea fi eronate in cazul in care exista un eveniment special intr o zi de weekend. Datele vor fi mai precise in functie de acuratetea diferentei dintre traficul din timpul saptamanii si weekend. 

(i) Filtrati semnalul, eliminati componentele de frecventa inalta 

'''

X = np.fft.fft(x)
N = len(X)

cutoff = 128
# How many values we want to keep from the original signal. The closer the value to fs / 2, the closer the filtered signal will be to the original signal.

X_filtered = np.zeros(N, dtype=complex)
X_filtered[:cutoff] = X[:cutoff]
X_filtered[-cutoff:] = X[-cutoff:]

x_smooth = np.fft.ifft(X_filtered).real

plt.figure(figsize=(15, 6))
plt.plot(x[S:S+month], alpha=0.3, label='Original Signal')
plt.plot(x_smooth[S:S+month], color='red', linewidth=2, label='Filtered Signal')
plt.title("Filtering Signal")
plt.legend()
plt.show()
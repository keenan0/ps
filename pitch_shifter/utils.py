import numpy as np

def sine(
        amplitude = 1.0,
        frequency = 440,
        phase = 0,
        sampling_freq = 44100,
        duration = 1.0
        ):
    
    t = np.linspace(0, duration, int(sampling_freq * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    return t, signal


from scipy.io import wavfile
def salveaza_audio(nume_fisier, vector_semnal, sample_rate=44100):
    """
    Salvează un vector de sample-uri într-un fișier .wav.
    Asigură conversia corectă de la float la int16 pentru compatibilitate.
    """
    # 1. Normalizare (Opțional, dar recomandat)
    # Ne asigurăm că cel mai mare vârf este la 1.0, să nu avem distorsiuni
    if np.max(np.abs(vector_semnal)) > 0:
        vector_semnal = vector_semnal / np.max(np.abs(vector_semnal))
    
    # 2. Conversie la 16-bit PCM (formatul standard pentru .wav)
    # Fișierele .wav standard așteaptă valori între -32768 și 32767
    semnal_int16 = (vector_semnal * 32767).astype(np.int16)
    
    # 3. Scrierea efectivă
    wavfile.write(nume_fisier, sample_rate, semnal_int16)
    print(f"Fișierul '{nume_fisier}' a fost salvat cu succes!")

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def sine(amplitude = 1.0, frequency = 440, phase = 0, sampling_freq = 44100, duration = 1.0):
    t = np.linspace(0, duration, int(sampling_freq * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    return t, signal

def save_audio(file_name, signal, sample_rate=44100):
    """
    Saves an array of samples as a .wav file. \n
    .wav should be included in the file_name by the user. \n 
    Handles conversion from float to int16.
    """

    # Normalize the input: prevents overflow of 16 bits when multiplying by 2^15. 
    # i.e. consider a signal with amp = 10 => 10 * 2 ^ 15 overflows to 0 -> 0101[0000000000000000]
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    
    # Convert to int16 for wav file
    semnal_int16 = (signal * 32767).astype(np.int16)
    
    wavfile.write(file_name, sample_rate, semnal_int16)
    print(f"File '{file_name}' saved.")

def plot_window_functions():
    N = 512
    n = np.arange(N)

    hann = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    blackman = (
        0.42
        - 0.5 * np.cos(2 * np.pi * n / (N - 1))
        + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    )
    triangular = 1 - np.abs((n - (N - 1) / 2) / ((N - 1) / 2))

        
    plt.figure()
    plt.plot(hann, label="Hann (Hanning)")
    plt.plot(hamming, label="Hamming")
    plt.plot(blackman, label="Blackman")
    plt.plot(triangular, label="Bartlett")

    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.title("Window Functions")
    plt.legend()
    plt.grid(True)

    plt.show()

def fft_analysis(signal, fs=44100, label="Signal"):
    n = len(signal)
    n_fft = 2**np.ceil(np.log2(n)).astype(int)
    
    fft_vals = np.abs(np.fft.rfft(signal, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/fs)

    idx_max = np.argmax(fft_vals)
    peak_freq = freqs[idx_max]
    
    print(f"[{label}] Dominant Frequency: {peak_freq:.2f} Hz")
    return freqs, fft_vals, peak_freq

def plot_fft_comparison(original, shifted, fs=44100):
    n = 2**16
    
    fft_orig = np.abs(np.fft.rfft(original, n))
    fft_shift = np.abs(np.fft.rfft(shifted, n))
    
    freqs = np.fft.rfftfreq(n, 1/fs)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, fft_orig, label="Original (C4 ~261Hz)", color='gray', alpha=0.6)
    plt.plot(freqs, fft_shift, label="Shifted (+6 semitones ~369Hz)", color='red')
    
    plt.xlim(0, 1000)
    plt.title("FFT Spectrum Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
import numpy as np
from scipy.io import wavfile

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

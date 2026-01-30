import numpy as np
import matplotlib.pyplot as plt
from utils import sine, save_audio
from Resampling.resample import resample

def overlap_add_after_filter(
        signal, 
        h_filter, 
        window_size,
        hop_size,
        factor):
    S = len(signal)
    L = window_size
    M = len(h_filter)
    N = L + M - 1

    hop_synthesis = hop_size * factor
    # trebuie sa explic de ce e formula asta asa
    n_frames = 1 + int((S - L) / hop_size)

    print(n_frames)
    output = np.zeros((n_frames - 1) * hop_synthesis + N)
    h_filter_fft = np.fft.fft(h_filter, n=N)
    for i in range(n_frames):
        start = i * hop_size
        end = start + L
        frame = signal[start : end]

        X = np.fft.fft(frame, n=N)
        Y = X * h_filter_fft
        y = np.abs(np.fft.ifft(Y))
        plt.plot(np.arange(start, start + N), y)
        
    print("test")
    plt.show()

def overlap_add_output(
        signal, 
        h_filter, 
        window_size,
        hop_size,
        factor):
    S = len(signal)
    L = window_size
    M = len(h_filter)
    N = L + M - 1

    hop_synthesis = hop_size * factor
    # trebuie sa explic de ce e formula asta asa
    n_frames = 1 + int((S - L) / hop_size)

    print(n_frames)
    output = np.zeros((n_frames - 1) * hop_synthesis + N)
    h_filter_fft = np.fft.fft(h_filter, n=N)
    for i in range(n_frames):
        start = i * hop_size
        end = start + L
        frame = signal[start : end]

        X = np.fft.fft(frame, n=N)
        Y = X * h_filter_fft
        y = np.abs(np.fft.ifft(Y))
        
        # define indices for the frame
        start_out = i * hop_synthesis
        end_out = start_out + N

        output[start_out : end_out] += y.real

    plt.plot(output[:2000])
    plt.show()

    return output

def overlap_add(
        signal, 
        h_filter, 
        window_size,
        hop_size,
        factor):
    S = len(signal)
    L = window_size
    M = len(h_filter)
    N = L + M - 1

    hop_synthesis = int(hop_size * factor)
    # trebuie sa explic de ce e formula asta asa
    n_frames = 1 + int((S - L) / hop_size)

    print(n_frames)
    output = np.zeros((n_frames - 1) * hop_synthesis + N)
    h_filter_fft = np.fft.fft(h_filter, n=N)
    for i in range(n_frames):
        start = i * hop_size
        end = start + L
        frame = signal[start : end]

        # apply hanning window for signal edge smoothing
        window = np.hanning(L)
        frame = frame * window

        X = np.fft.fft(frame, n=N)
        Y = X * h_filter_fft
        y = np.fft.ifft(Y).real
        
        # define indices for the frame
        start_out = i * hop_synthesis
        end_out = start_out + N

        output[start_out : end_out] += y.real

    plt.plot(output[:2000])
    plt.show()

    return output

t,sig = sine(frequency=261, duration=1.0)
#overlap_add_after_filter(sig, np.array([1]), 1024, 512, 2)
sigx2 = overlap_add(sig, [1], 1024, 512, 2)

save_audio("OverlapAdd/sig_261hz.wav", sig)
save_audio("OverlapAdd/sig_261hz_x2.wav", sigx2)

resampled = resample(sigx2, 2)
save_audio("OverlapAdd/sig261hz_resampled.wav", resampled)
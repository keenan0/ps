import numpy as np
import matplotlib.pyplot as plt

from Resampling.resample import resample
from utils import save_audio, sine, fft_analysis, plot_fft_comparison

def synchronized_overlap_add(signal, window_size, hop_size, factor=1.0, window_function=None, normalize=True):
    n_frames = int(np.floor((len(signal) - window_size) / hop_size)) + 1
    hop_synchronize = int(hop_size * factor)
    
    safety_margin = window_size * 2
    total_len_out = (n_frames - 1) * hop_synchronize + window_size + safety_margin
    combined_signal = np.zeros(total_len_out)
    norm_buffer = np.zeros(total_len_out)

    win = window_function(window_size) if window_function else np.ones(window_size)
    frames = []
    for i in range(n_frames):
        start = i * hop_size
        frames.append(signal[start : start + window_size] * win)

    first_frame = frames[0]
    combined_signal[0:window_size] += first_frame
    norm_buffer[0:window_size] += win

    last_actual_start = 0

    for i in range(1, n_frames):
        theoretical_start = last_actual_start + hop_synchronize
        
        overlap_len = window_size - hop_synchronize
        target_area = combined_signal[theoretical_start : theoretical_start + overlap_len]
        candidate_area = frames[i][:overlap_len]

        if len(target_area) == overlap_len:
            if factor < 0.99: 
                best_shift = 0
            else:
                correlation = np.correlate(target_area, candidate_area, mode='full')
                mid_point = overlap_len - 1
                best_shift = np.argmax(correlation) - mid_point
        else:
            best_shift = 0

        final_start = theoretical_start + best_shift
        final_end = final_start + window_size

        frame = frames[i]
        combined_signal[final_start:final_end] += frame
        norm_buffer[final_start:final_end] += win
        
        last_actual_start = final_start

    actual_end = last_actual_start + window_size
    combined_signal = combined_signal[:actual_end]
    norm_buffer = norm_buffer[:actual_end]

    if normalize:
        norm_buffer[norm_buffer < 1e-10] = 1.0
        combined_signal /= norm_buffer

    return combined_signal

import numpy as np
import matplotlib.pyplot as plt

fs = 44100
t, sig = sine(frequency=440, duration=1.0)

n_semitones = 6
stretch_factor = 2 ** (n_semitones / 12)

ss = synchronized_overlap_add(sig, 2048, 1024, stretch_factor, window_function=np.hanning, normalize=False)
res = resample(ss, stretch_factor)

save_audio("OverlapAdd/SOLA/trumped_signal_original.wav", sig)
save_audio(f"OverlapAdd/SOLA/trumpet_signal_resampled_{n_semitones}_semitones.wav", res)

f_orig, mag_orig, peak_orig = fft_analysis(sig, fs, "Original")
f_sola, mag_sola, peak_sola = fft_analysis(ss , fs, "SOLA")
f_res, mag_res, peak_res    = fft_analysis(res, fs, "RES")
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(f_orig, mag_orig, color='blue', label=f'Original (Peak: {peak_orig:.1f}Hz)')
plt.xlim(0, 800)
plt.title("Original Signal FFT")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(f_sola, mag_sola, color='red', label=f'SOLA (Peak: {peak_sola:.1f}Hz)')
plt.xlim(0, 800)
plt.xlabel("Frequency (Hz)")
plt.title(f"SOLA FFT (Stretch x{stretch_factor:.2f})")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(f_res, mag_res, color='orange', label=f'RES (Peak: {peak_res:.1f}Hz)')
plt.xlim(0, 800)
plt.xlabel("Frequency (Hz)")
plt.title(f"Resampled FFT (Stretch x{stretch_factor:.2f})")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
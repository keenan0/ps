import numpy as np
import librosa
import matplotlib.pyplot as plt
from utils import save_audio

def wsola(signal, window_size, hop_size, factor=1.0, window_function=None, tolerance=None):
    if tolerance is None:
        tolerance = window_size // 4
        
    n_frames = int(np.floor((len(signal) - window_size) / hop_size)) + 1
    hop_s = int(hop_size * factor)
    
    total_len_out = (n_frames - 1) * hop_s + window_size + tolerance
    combined_signal = np.zeros(total_len_out)
    norm_buffer = np.zeros(total_len_out)
    
    win = window_function(window_size) if window_function else np.ones(window_size)
    
    combined_signal[0:window_size] += signal[0:window_size] * win
    norm_buffer[0:window_size] += win
    
    last_analysis_idx = 0 
    
    for i in range(1, n_frames):
        natural_next_pos = last_analysis_idx + hop_size
        overlap_len = window_size - hop_s
        
        if natural_next_pos + overlap_len > len(signal): 
            break
        
        target_ref = signal[natural_next_pos : natural_next_pos + overlap_len]
        
        theoretical_pos = i * hop_size
        start_search = max(0, theoretical_pos - tolerance)
        end_search = min(len(signal) - window_size, theoretical_pos + tolerance)
        
        best_offset = 0
        max_corr = -float('inf')
        
        search_region = signal[start_search : end_search + overlap_len]
        
        for k in range(len(search_region) - overlap_len):
            candidate = search_region[k : k + overlap_len]
            corr = np.dot(target_ref, candidate)
            if corr > max_corr:
                max_corr = corr
                best_offset = (start_search + k) - theoretical_pos

        actual_analysis_idx = theoretical_pos + best_offset
        frame = signal[actual_analysis_idx : actual_analysis_idx + window_size] * win
        
        start_s = i * hop_s
        combined_signal[start_s : start_s + window_size] += frame
        norm_buffer[start_s : start_s + window_size] += win
        
        last_analysis_idx = actual_analysis_idx

    actual_end = (n_frames - 1) * hop_s + window_size
    combined_signal = combined_signal[:actual_end]
    norm_buffer = norm_buffer[:actual_end]
    
    norm_buffer[norm_buffer < 1e-10] = 1.0
    return combined_signal / norm_buffer

filename = librosa.ex("trumpet")
sig, sr = librosa.load(filename, sr=44100, mono=True)
sig = sig[:44100 * 3]

import numpy as np
import librosa
from Resampling.resample import resample
from utils import save_audio, fft_analysis

n_semitones = -4
stretch_factor = 2 ** (n_semitones / 12)

print(f" WSOLA (Stretch factor: {stretch_factor:.2f})...")
ss_wsola = wsola(sig, 2048, 1024, stretch_factor, window_function=np.hanning, tolerance=256)

final_res = resample(ss_wsola, stretch_factor)

save_audio(f"OverlapAdd/WSOLA/trumpet_pitch_shifted_{n_semitones}_semitones.wav", final_res, sr)

f_orig, mag_orig, peak_orig = fft_analysis(sig, sr, "Original")
f_wsola, mag_wsola, peak_wsola = fft_analysis(ss_wsola, sr, "WSOLA (Stretched)")
f_final, mag_final, peak_final = fft_analysis(final_res, sr, "Final (Pitch Shifted)")

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(f_orig, mag_orig, label=f'Original ({peak_orig:.1f} Hz)')
plt.xlim(0, 1000)
plt.legend()
plt.title("Original FFT")

plt.subplot(3, 1, 2)
plt.plot(f_wsola, mag_wsola, color='green', label=f'WSOLA Stretched ({peak_wsola:.1f} Hz)')
plt.xlim(0, 1000)
plt.legend()
plt.title("WSOLA FFT")

plt.subplot(3, 1, 3)
plt.plot(f_final, mag_final, color='orange', label=f'Final Shifted ({peak_final:.1f} Hz)')
plt.xlim(0, 1000)
plt.legend()
plt.title(f"Final FFT ({n_semitones} semitones)")

plt.tight_layout()
plt.show()
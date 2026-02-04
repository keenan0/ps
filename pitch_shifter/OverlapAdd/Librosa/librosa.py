import librosa
import numpy as np
import matplotlib.pyplot as plt
from utils import save_audio, fft_analysis
from OverlapAdd.overlap_add import overlap_add_colored
from Resampling.resample import resample

filename = librosa.ex('trumpet')
sig, fs = librosa.load(filename, sr=44100, mono=True)

sig = sig[:44100*3]
n_semitones = -6

stretch_factor = 2 ** (n_semitones / 12)
ss = overlap_add_colored(sig, 2048, 1024, stretch_factor, window_function=np.hanning, normalize=False)
res = resample(ss, stretch_factor)

save_audio("OverlapAdd/Librosa/trumpet_original.wav", sig)
save_audio(f"OverlapAdd/Librosa/trumped_{n_semitones}_semitones.wav", res)

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
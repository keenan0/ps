import numpy as np
import matplotlib.pyplot as plt
from utils import sine, save_audio, plot_fft_comparison
from Resampling.resample import resample

def overlap_add(signal, window_size, hop_size, factor=1.0):
    # A frame is defined as a window here
    n_frames = int(np.floor((len(signal) - window_size) / hop_size)) + 1
    hop_synchronize = int(hop_size * factor)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    
    ax1.set_title(f"Signal cut into Frames (Hop = {hop_size})")
    ax1.plot(signal, color='lightgray', alpha=0.5, label="Original Signal")

    frames = []
    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size

        frame = signal[start:end]
        frames.append(frame)

        color = np.random.rand(3)

        ax1.axvspan(start, end, facecolor=color, alpha=0.2, edgecolor="red")
        ax1.plot(np.arange(start, end), frame, color=color, linewidth=1)

    ax2.set_title(f"Synthesised Signal (Hop = {hop_synchronize}, Factor = {factor})")
    
    # Output signal
    total_len_out = (n_frames - 1) * hop_synchronize + window_size
    combined_signal = np.zeros(total_len_out)

    for i in range(n_frames):
        start_out = i * hop_synchronize
        end_out = start_out + window_size
        
        frame = frames[i]
        color = ax1.patches[i].get_facecolor()
        
        combined_signal[start_out:end_out] += frame
        
        ax2.axvspan(start_out, end_out, facecolor=color, alpha=0.2, edgecolor="blue")
        ax2.plot(np.arange(start_out, end_out), frame, color=color, alpha=0.7)

    ax2.plot(combined_signal, color='black', linewidth=1.5, label="Amplitudes Sum")
    
    max_len = max(len(signal), len(combined_signal))
    ax1.set_xlim(0, max_len)
    ax2.set_xlim(0, max_len)

    ax1.legend(loc='upper right', fontsize='small')
    ax2.legend(loc='upper right', fontsize='small')
    
    print(len(signal), len(combined_signal))

    plt.tight_layout()
    plt.show()

    return combined_signal

def visualize_framing(signal, window_size, hop_size, n_to_show=5):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    display_limit = (n_to_show - 1) * hop_size + window_size
    t_axis = np.arange(display_limit)
    ax.plot(t_axis, signal[:display_limit], color='black', alpha=0.2, label="Original Signal")

    colors = plt.cm.viridis(np.linspace(0, 1, n_to_show))

    for i in range(n_to_show):
        start = i * hop_size
        end = start + window_size
        frame = signal[start:end]
        
        offset = - (i * 0.5) 
        
        ax.axvspan(start, end, ymin=0.6, ymax=0.9, color=colors[i], alpha=0.15)
        
        ax.plot(np.arange(start, end), frame + offset, color=colors[i], 
                linewidth=2, label=f"Frame {i}")
        
        ax.annotate('', xy=(start, offset), xytext=(start, offset - 0.2),
                    arrowprops=dict(arrowstyle='->', color=colors[i]))
        
        if i < n_to_show - 1:
            ax.hlines(y=offset - 0.4, xmin=start, xmax=start + hop_size, 
                      colors=colors[i], linestyles='dashed')
            ax.text(start + hop_size/4, offset - 0.6, f'Hop {i}', color=colors[i], fontsize=8)

    ax.set_title(f"Step 1: Framing (Window Size = {window_size}, Hop Size = {hop_size})")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (staggered for visibility)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def overlap_add_colored(signal, window_size, hop_size, factor=1.0, window_function=None, normalize=True):
    n_frames = int(np.floor((len(signal) - window_size) / hop_size)) + 1
    hop_synchronize = int(hop_size * factor)
    
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, n_frames)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    
    # --- ANALYSIS ---
    ax1.set_title(f"Analysis: Original Signal with Frame Overlays (Hop In = {hop_size})")
    ax1.plot(signal, color='lightgray', alpha=0.8, label="Original Signal")

    frames = []
    win = window_function(window_size) if window_function else np.ones(window_size)

    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size
        
        frame = signal[start:end].copy() 
        windowed_frame = frame * win
        frames.append(windowed_frame)

        current_color = colors[i]
        ax1.axvspan(start, end, facecolor=current_color, alpha=0.1, edgecolor="none")
        ax1.plot(np.arange(start, end), frame, color=current_color, linewidth=0.5, alpha=0.4)

    # --- SYNTHESIS ---
    ax2.set_title(f"Synthesis: Reassembled Signal (Normalize={normalize}, Factor={factor})")
    
    total_len_out = (n_frames - 1) * hop_synchronize + window_size
    combined_signal = np.zeros(total_len_out)
    norm_buffer = np.zeros(total_len_out)

    for i in range(n_frames):
        start_out = i * hop_synchronize
        end_out = start_out + window_size
        
        frame = frames[i]
        current_color = colors[i]
        
        combined_signal[start_out:end_out] += frame
        norm_buffer[start_out:end_out] += win
        
        ax2.axvspan(start_out, end_out, facecolor=current_color, alpha=0.15, edgecolor="none")
        ax2.plot(np.arange(start_out, end_out), frame, color=current_color, alpha=0.6)

    if normalize:
        norm_buffer[norm_buffer < 1e-10] = 1.0
        combined_signal /= norm_buffer

    ax2.plot(combined_signal, color='black', linewidth=1.5, label="Final Output (Summed)", zorder=10)
    
    max_len = max(len(signal), len(combined_signal))
    ax1.set_xlim(0, max_len)
    ax2.set_xlim(0, max_len)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    # plt.show()

    return combined_signal

t, sig = sine(frequency=261, duration=1.0)

# sig = sig[:1000]

n_semitones = 6
upsample_factor = 2 ** (n_semitones / 12)
downsample_factor = 2 ** (- n_semitones / 12)

a = overlap_add_colored(sig, 1024, 512, factor=upsample_factor, window_function=np.hanning, normalize=False)
a_res = resample(a, upsample_factor)
# b = overlap_add_colored(sig, 512, 256, factor=downsample_factor, window_function=np.hanning, normalize=True)
# b_res = resample(b, downsample_factor)

import scipy.signal as sp

# plot_fft_comparison(sig, a)
# plot_fft_comparison(sig, a_res)
print(len(sig), len(a), len(a_res))

from utils import fft_analysis

fs = 44100
stretch_factor = upsample_factor
ss = overlap_add_colored(sig, 2048, 1024, stretch_factor, window_function=np.hanning, normalize=False)
res = resample(ss, stretch_factor)

save_audio("OverlapAdd/OLA_hanning/hanning_x1_0.wav", sig)
save_audio(f"OverlapAdd/OLA_hanning/hanning_x{stretch_factor}.wav", res)

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
plt.title(f"OLA FFT (Stretch x{stretch_factor:.2f})")
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
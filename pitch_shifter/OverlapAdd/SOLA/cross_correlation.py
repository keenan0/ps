import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def cross_correlate_frames(target_window, candidate_window, shift_range=100, title="Signal Synchronization"):
    L = min(len(target_window), len(candidate_window))
    x = target_window[:L]
    y = candidate_window[:L]

    full_correlation = np.correlate(x, y, mode='full')
    lags = np.arange(-L + 1, L)
    
    search_mask = (lags >= -shift_range) & (lags <= shift_range)
    
    restricted_lags = lags[search_mask]
    restricted_corr = full_correlation[search_mask]
    
    k_max = restricted_lags[np.argmax(restricted_corr)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10))
    plt.subplots_adjust(hspace=0.6)

    ax1.plot(x, label="Target (In Buffer)", color='#1f77b4', alpha=0.8)
    ax1.plot(y, label="Candidate (Unsynced)", color='#d62728', alpha=0.6, linestyle='--')
    ax1.set_title(f"1. Original Overlap - {title}")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    ax2.plot(lags, full_correlation, color='gray', alpha=0.3, label="Correlation (Outside Range)")
    ax2.plot(restricted_lags, restricted_corr, color='#9467bd', linewidth=2, label="Search Range (Allowed)")
    
    ax2.axvspan(-shift_range, shift_range, color='yellow', alpha=0.1, label="Search Range Area")
    
    ax2.axvline(x=k_max, color='#2ca02c', linestyle='--', label=f'Best Lag found: k = {k_max}')
    ax2.set_title(r"2. Restricted Cross-Correlation $R_{xy}(k)$")
    ax2.set_xlabel("Lag (Samples)")
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, alpha=0.2)

    ax3.plot(x, label="Target", color='#1f77b4', linewidth=2)
    ax3.plot(np.arange(L) + k_max, y, label=f"Synced (Shift: {k_max})", color='#2ca02c', linewidth=2)
    ax3.set_title(f"3. Aligned Phase (Restricted to ±{shift_range} samples)")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim(min(0, k_max) - 10, max(L, L + k_max) + 10)

    plt.show()
    return k_max

import librosa

filename = librosa.ex('trumpet')
sig, fs = librosa.load(filename, sr=44100, mono=True)

from utils import sine
t,sig = sine(frequency=200)

start = 28200
window_size = 1024
hop = window_size // 2

target_raw = sig[start : start + window_size]
candidate_raw = sig[start + hop : start + hop + window_size]

overlap_target = sig[start + hop : start + window_size] 
artificial_drift = 40 
overlap_candidate = sig[start + hop + artificial_drift : start + window_size + artificial_drift]
#cross_correlate_frames(overlap_target, overlap_candidate, shift_range=50)

win = np.hanning(window_size)
target_windowed = target_raw * win
candidate_windowed = candidate_raw * win

#cross_correlate_frames(target_windowed, candidate_windowed, title="Trumpet Sample Analysis")

def visualize_single_overlap_sola():
    filename = librosa.ex('trumpet')
    sig, fs = librosa.load(filename, sr=44100, mono=True)
    
    window_size = 512
    hop_size = window_size // 2
    overlap_len = window_size - hop_size
    
    start_idx = 12000
    
    frame1 = sig[start_idx : start_idx + window_size]
    
    artificial_drift = 35
    start_idx_2 = start_idx + hop_size + artificial_drift
    frame2 = sig[start_idx_2 : start_idx_2 + window_size]

    target_overlap = frame1[hop_size:] 
    
    candidate_overlap = frame2[:overlap_len]
    
    win_overlap = np.hanning(len(target_overlap))
    
    correlation = np.correlate(target_overlap * win_overlap, 
                               candidate_overlap * win_overlap, 
                               mode='full')
    
    lags = np.arange(-len(candidate_overlap) + 1, len(target_overlap))
    
    search_range = 100
    valid_indices = np.where(np.abs(lags) <= search_range)[0]
    
    best_idx_restricted = np.argmax(correlation[valid_indices])
    best_shift = lags[valid_indices][best_idx_restricted]
    
    print(f"Drift artificial introdus: {artificial_drift}")
    print(f"Shift detectat de SOLA: {best_shift} (Ar trebui să fie aprox {-artificial_drift})")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)
    
    x_axis = np.arange(window_size + hop_size)

    ax1.plot(np.arange(window_size), frame1, color='blue', label="Frame 1 (Buffer Tail)", alpha=0.6)
    ax1.plot(np.arange(window_size) + hop_size, frame2, color='red', linestyle='--', label="Frame 2 (Candidate Head)", alpha=0.6)
    
    ax1.axvspan(hop_size, window_size, color='yellow', alpha=0.2, label="Overlap Zone")
    ax1.set_title("Before Synchronisation")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(lags, correlation, color='purple')
    ax2.axvline(best_shift, color='green', linestyle='--', label=f'Best Shift: {best_shift}')
    ax2.axvspan(-search_range, search_range, color='green', alpha=0.1)
    ax2.set_title(f"Cross-Correlation on Overlap Area (Best Shift = {best_shift})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    corrected_pos = hop_size + best_shift
    
    ax3.plot(np.arange(window_size), frame1, color='red', label="Frame 1", linewidth=2)
    ax3.plot(np.arange(window_size) + corrected_pos, frame2, color='yellow', label="Frame 2 (Shifted)", linewidth=2, alpha=0.8)
    
    ax3.set_title("After SOLA: New Window Shifted to Match Phase")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    ax3.set_xlim(hop_size - 100, hop_size + 300) 

    plt.show()

visualize_single_overlap_sola()
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
import time

# --- 1. DEFINIREA ALGORITMILOR ---

# OLA Standard (optimizat pentru procesare, nu pentru desenat)
def overlap_add_fast(signal, window_size, hop_size, factor=1.0, window_function=np.hanning):
    n_frames = int(np.floor((len(signal) - window_size) / hop_size)) + 1
    hop_s = int(hop_size * factor)
    
    total_len_out = (n_frames - 1) * hop_s + window_size
    combined_signal = np.zeros(total_len_out)
    norm_buffer = np.zeros(total_len_out)
    
    win = window_function(window_size)
    
    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start : start + window_size] * win
        
        start_out = i * hop_s
        combined_signal[start_out : start_out + window_size] += frame
        norm_buffer[start_out : start_out + window_size] += win
        
    norm_buffer[norm_buffer < 1e-10] = 1.0
    return combined_signal / norm_buffer

# SOLA (Codul tau)
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
        
        k_max = 150

        if factor < 0.99:
            best_shift = 0
        else:
            # Calculăm corelația
            correlation = np.correlate(target_area, candidate_area, mode='full')
            mid_point = overlap_len - 1
            
            # Restricționăm zona în care căutăm maximul
            # Ne uităm doar în intervalul [mid_point - k_max, mid_point + k_max]
            search_start = max(0, mid_point - k_max)
            search_end = min(len(correlation), mid_point + k_max)
            
            # Găsim maximul DOAR în acea zonă
            sub_correlation = correlation[search_start : search_end]
            best_shift_local = np.argmax(sub_correlation)
            
            # Calculăm shift-ul real
            best_shift = (best_shift_local + search_start) - mid_point
        
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

# WSOLA (Codul tau)
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

# --- 2. LOGICA EXPERIMENTULUI ---

def run_comparison_test(signal, sr, name, n_semitones=6):
    print(f"\n--- Rulare Test: {name} ---")
    
    # Parametri
    window_size = 2048
    hop_size = 512 # Hop de analiză
    factor = 2 ** (n_semitones / 12.0) # Stretch factor > 1
    
    # Folder salvare
    output_dir = "Audios"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 1. Procesare OLA
    start_time = time.time()
    y_ola_stretch = overlap_add_fast(signal, window_size, hop_size, factor, np.hanning)
    # Resample invers pentru Pitch Shift
    y_ola_pitch = librosa.resample(y_ola_stretch, orig_sr=sr*factor, target_sr=sr)
    ola_time = time.time() - start_time
    results['OLA'] = y_ola_pitch
    
    # 2. Procesare SOLA
    start_time = time.time()
    y_sola_stretch = synchronized_overlap_add(signal, window_size, hop_size, factor, np.hanning)
    y_sola_pitch = librosa.resample(y_sola_stretch, orig_sr=sr*factor, target_sr=sr)
    sola_time = time.time() - start_time
    results['SOLA'] = y_sola_pitch

    # 3. Procesare WSOLA
    start_time = time.time()
    y_wsola_stretch = wsola(signal, window_size, hop_size, factor, np.hanning, tolerance=512)
    y_wsola_pitch = librosa.resample(y_wsola_stretch, orig_sr=sr*factor, target_sr=sr)
    wsola_time = time.time() - start_time
    results['WSOLA'] = y_wsola_pitch
    
    print(f"Execution Times: OLA={ola_time:.3f}s, SOLA={sola_time:.3f}s, WSOLA={wsola_time:.3f}s")
    
    # 4. Salvare Audio
    sf.write(f"{output_dir}/{name}_original.wav", signal, sr)
    sf.write(f"{output_dir}/{name}_OLA_+{n_semitones}st.wav", y_ola_pitch, sr)
    sf.write(f"{output_dir}/{name}_SOLA_+{n_semitones}st.wav", y_sola_pitch, sr)
    sf.write(f"{output_dir}/{name}_WSOLA_+{n_semitones}st.wav", y_wsola_pitch, sr)
    
    # 5. Generare Ploturi (Spectrograme)
    plt.figure(figsize=(15, 10))
    
    # Original
    plt.subplot(4, 1, 1)
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=2048)), ref=np.max)
    librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{name}: Original Signal")
    
    # OLA
    plt.subplot(4, 1, 2)
    min_len = min(len(y_ola_pitch), len(signal))
    D_ola = librosa.amplitude_to_db(np.abs(librosa.stft(y_ola_pitch[:min_len], n_fft=2048)), ref=np.max)
    librosa.display.specshow(D_ola, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"OLA (+{n_semitones} semitones)")
    
    # SOLA
    plt.subplot(4, 1, 3)
    min_len = min(len(y_sola_pitch), len(signal))
    D_sola = librosa.amplitude_to_db(np.abs(librosa.stft(y_sola_pitch[:min_len], n_fft=2048)), ref=np.max)
    librosa.display.specshow(D_sola, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"SOLA (+{n_semitones} semitones)")
    
    # WSOLA
    plt.subplot(4, 1, 4)
    min_len = min(len(y_wsola_pitch), len(signal))
    D_wsola = librosa.amplitude_to_db(np.abs(librosa.stft(y_wsola_pitch[:min_len], n_fft=2048)), ref=np.max)
    librosa.display.specshow(D_wsola, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"WSOLA (+{n_semitones} semitones)")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_spectrogram_comparison.pdf")
    plt.show()

# --- 3. EXECUȚIA TESTELOR ---

if __name__ == "__main__":
    sr = 44100
    t = np.linspace(0, 2, sr * 2)
    sig_sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    run_comparison_test(sig_sine, sr, "SineWave_440Hz")

    filename = librosa.ex('choice')
    sig_complex, sr = librosa.load(filename, sr=None, duration=4.0)
    run_comparison_test(sig_complex, sr, "Drum + Bass")

    filename = librosa.ex('trumpet')
    sig_trumpet, sr = librosa.load(filename, sr=None, duration=4.0)
    run_comparison_test(sig_trumpet, sr, "Trumpet")

    filename = librosa.ex('libri1')
    sig_voice, sr = librosa.load(filename, sr=None, duration=4.0)
    run_comparison_test(sig_voice, sr, "Voice Speech")

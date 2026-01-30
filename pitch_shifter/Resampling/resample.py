import numpy as np 
import matplotlib.pyplot as plt

def resample(signal, factor):
    """
    factor > 1.0 -> Higher pitch
    factor < 1.0 -> Lower pitch
    """
    
    old_indices = np.arange(len(signal))
    
    # Get new indices based on the factor: i.e. [0,1,2,3,4,5] -(factor=2)-> [0,2,4] or [0,1,2,3,4,5] -(factor=0.5)-> [0, 0.5, 1, 1.5, 2, ..., 4.5, 5] 
    new_indices = np.arange(0, len(signal) - 1, factor)
    
    # Interpolate the values
    resampled = np.interp(new_indices, old_indices, signal)
    
    return resampled

def plot_resampling_comparison(original, fs, factor_down=2.0, factor_up=0.5):
    res_down = resample(original, factor_down)
    res_up = resample(original, factor_up)

    t_orig = np.arange(len(original)) / fs
    t_down = np.arange(len(res_down)) / fs
    t_up = np.arange(len(res_up)) / fs
    
    max_time = max(t_orig[-1], t_up[-1])
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [2, 1]})

    axes[0, 0].plot(t_orig, original, color='#34495e', label='Original')
    axes[1, 0].plot(t_down, res_down, color='#e67e22', label=f'Downsampled (x{factor_down:.3f})')
    axes[2, 0].plot(t_up, res_up, color='#27ae60', label=f'Upsampled (x{factor_up:.3f})')
    
    for i in range(3):
        axes[i, 0].set_xlim(0, max_time)
        axes[i, 0].legend(loc='upper right')
        axes[i, 0].set_ylabel('Amplitude')

    n_view = 10 
    
    axes[0, 1].stem(np.arange(n_view), original[:n_view], basefmt=" ")
    axes[0, 1].set_title('Original Samples')

    n_view_down = int(n_view / factor_down)
    axes[1, 1].stem(np.arange(n_view_down), res_down[:n_view_down], linefmt='C1-', markerfmt='C1o', basefmt=" ")
    axes[1, 1].set_title('Samples Skipped (Downsampled)')

    n_view_up = int(n_view / factor_up)
    axes[2, 1].stem(np.arange(n_view_up), res_up[:n_view_up], linefmt='C2-', markerfmt='C2o', basefmt=" ")
    axes[2, 1].set_title('Samples Interpolated (Upsampled)')

    for i in range(3):
        axes[i, 1].grid(True, alpha=0.2)

    axes[2, 0].set_xlabel('Time (seconds)')
    axes[2, 1].set_xlabel('Sample Index (n)')
    
    plt.tight_layout()
    plt.show()
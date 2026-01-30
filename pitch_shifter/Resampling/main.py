from Resampling.resample import resample, plot_resampling_comparison
from utils import sine
import numpy as np
from utils import save_audio

fs = 44100
t, sig = sine()

n_semitones = 6
upsample_factor = 2 ** (n_semitones / 12)
downsample_factor = 2 ** (- n_semitones / 12)

up = resample(sig, upsample_factor)
down = resample(sig, downsample_factor)

plot_resampling_comparison(sig[:2000], fs, 1 / downsample_factor, 1 / upsample_factor)\
    
save_audio("Resampling/signal_original.wav", sig, fs)
save_audio("Resampling/signal_downsampled.wav", down, fs)
save_audio("Resampling/signal_upsampled.wav", up, fs)
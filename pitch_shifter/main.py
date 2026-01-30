from utils import sine, save_audio
from Resampling.resample import resample

t, sig = sine()
save_audio("Resampling/sine_440hz_1x.wav", sig)
sig_sped = resample(sig, 2)
save_audio("Resampling/sine_440hz_2x.wav", sig_sped)
sig_slowed = resample(sig, 0.5)
save_audio("Resampling/sine_440hz_0.5x.wav", sig_slowed)
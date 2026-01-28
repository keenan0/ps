from utils import sine, salveaza_audio
from Resampling.resample import resample_signal

t, sig = sine()
salveaza_audio("Resampling/sine_440hz_1x.wav", sig)
sig_sped = resample_signal(sig, 2)
salveaza_audio("Resampling/sine_440hz_2x.wav", sig_sped)
sig_slowed = resample_signal(sig, 0.5)
salveaza_audio("Resampling/sine_440hz_0.5x.wav", sig_slowed)
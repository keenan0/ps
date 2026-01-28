import numpy as np 

def resample_signal(signal, factor):
    """
    factor > 1.0 -> Sunet mai scurt, pitch mai înalt
    factor < 1.0 -> Sunet mai lung, pitch mai jos
    """
    old_indices = np.arange(len(signal))
    # Calculăm noile indexuri pe baza factorului
    new_indices = np.arange(0, len(signal) - 1, factor)
    
    # Interpolarea liniară: calculăm valorile pentru noile indexuri
    # np.interp(unde vrem valori, indexuri vechi, valori vechi)
    resampled = np.interp(new_indices, old_indices, signal)
    
    return resampled
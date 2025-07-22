import numpy as np
from scipy.signal import welch

def bandpower(signal, fs, band):
    """
    Calcula la potencia de una banda usando el método de Welch.
    - signal: lista o array de la señal EEG de un canal
    - fs: frecuencia de muestreo (Hz)
    - band: tupla con rango de la banda (ej. (4,8))
    """
    # Convertir la señal a un array de numpy, si no lo es ya
    signal = np.array(signal)
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    power = np.trapz(psd[idx], freqs[idx])
    return power

def calculate_tbr(eeg_data, fs=256, channel_index=0):
    """
    Calcula la relación Theta/Beta de un canal EEG específico.
    - eeg_data: lista de muestras [[ch1, ch2, ...], ...]
    - channel_index: canal a analizar (por ejemplo, 0 = Fp1)
    """
    # Convertir eeg_data a un array de numpy
    eeg_data = np.array(eeg_data)

    # Seleccionar los datos del canal específico
    signal = eeg_data[:, channel_index]

    # Calcular las potencias de las bandas
    theta_power = bandpower(signal, fs, (4, 8))
    beta_power  = bandpower(signal, fs, (12, 30))

    # Calcular la relación TBR
    tbr = theta_power / beta_power if beta_power > 0 else 0
    return round(tbr, 3)

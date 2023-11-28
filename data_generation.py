import numpy as np
import ruptures as rpt


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Computes the Signal-to-Noise Ratio (SNR) between a signal and noise.
s
    Parameters:
    signal (np.ndarray): The signal.
    noise (np.ndarray): The noise.

    Returns:
    float: The Signal-to-Noise Ratio (SNR) in decibels (dB).
    """

    # Calculate the power of the signal and noise
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    # Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB)
    snr = 10 * np.log10(signal_power / noise_power)

    return snr


def generate_signals(
    n_informative_dimensions: int,
    length: int,
    n_dimensions: int,
    n_bkps: int,
    sigma: float,
):
    """
    Generates synthetic signals with informative and noisy dimensions.

    Parameters:
    n_informative_dimensions (int): Number of informative dimensions in the signal.
    length (int): Length of the generated signal.
    n_dimensions (int): Total number of dimensions in the generated signal.
    n_bkps (int): Number of breakpoints where in the signal along informatives dimensions.
    sigma (float): Standard deviation of the noise to be added.

    Returns:
    tuple: A tuple containing:
        - noisy_signal (np.ndarray): The generated noisy signal with informative and noisy dimensions.
        - segmentation (np.ndarray): The segmentation points.
        - snr (float): Signal-to-Noise Ratio (SNR) of the generated signal.
    """

    # Generate informative and noisy segments separately
    signal, segmentation = rpt.pw_constant(
        length, n_informative_dimensions, n_bkps, noise_std=0
    )
    noisy_signal, _ = rpt.pw_constant(
        length, n_dimensions - n_informative_dimensions, 0, noise_std=0
    )

    # Combine the informative and noisy segments into the final signal
    final_signal = np.zeros((length, n_dimensions))
    final_signal[:, 0:n_informative_dimensions] = signal
    final_signal[:, n_informative_dimensions:n_dimensions] = noisy_signal

    # Normalize the signal to have zero mean and unit variance
    normalized_signal = (final_signal - np.mean(final_signal)) / np.std(final_signal)

    # Generate noise with the specified standard deviation
    noise = np.random.normal(0, sigma, normalized_signal.shape)

    # Add the noise to the normalized signal to create the noisy signal
    noisy_signal = normalized_signal + noise

    # Calculate the Signal-to-Noise Ratio (SNR) of the generated signal
    snr = compute_snr(
        normalized_signal, noise
    )  # Assuming compute_snr function is defined

    return noisy_signal, segmentation, snr

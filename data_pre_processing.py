import numpy as np
from scipy.signal import stft, butter, lfilter
from breakpoints_computation import get_segment_ids
from typing import Optional, List, Tuple


def apply_low_pass_filter(
    data: np.ndarray, sample_rate: float, cutoff_freq: Optional[float] = None
) -> np.ndarray:
    """
    Apply a low-pass filter to input data.

    Args:
        data (np.ndarray): The input data to be filtered.
        sample_rate (float): The sample rate of the input data.
        cutoff_freq (float, optional): The cutoff frequency for the low-pass filter.
            If None, no filtering is applied, and the original data is returned.

    Returns:
        np.ndarray: The filtered data if cutoff_freq is not None, otherwise, the original data.

    Note:
        This function uses a Butterworth low-pass filter to filter the input data.
    """
    if cutoff_freq is None:
        return (
            data  # If cutoff_freq is None, return the original data without filtering.
        )

    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq

    # Design the Butterworth low-pass filter
    b, a = butter(8, normal_cutoff, btype="low", analog=False)

    # Apply the filter to the input data
    filtered_data = lfilter(b, a, data, axis=0)

    return filtered_data  # Return the filtered data.


def center_and_reduce_signal(signal: np.ndarray) -> np.ndarray:
    """
    Center and reduce the input signal.

    Args:
        signal (np.ndarray): The input signal to be processed.

    Returns:
        np.ndarray: The centered and reduced signal.
    """
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    std_without_zeros = np.where(std < 1e-10, 1.0, std)
    normalized_data = (signal - mean) / std_without_zeros
    return normalized_data


def signals_pre_process(
    signals: List[np.ndarray], sample_rate: float, cutoff_freq: Optional[float] = None
) -> np.ndarray:
    """
    Pre-process a list of signals by applying a low-pass filter, centering, reducing,
    and converting them into a np.ndarray.

    Args:
        signals (List[np.ndarray]): A list of input signals to be pre-processed.
        sample_rate (float): The sample rate of the input signals.
        cutoff_freq (float, optional): The cutoff frequency for the low-pass filter.
            If None, no filtering is applied.

    Returns:
        np.ndarray: The pre-processed signals as a np.ndarray of shape
            (nb_signals, signal_length, nb_features).
    """
    # Apply low-pass filter
    filtered_signals = []
    for signal in signals:
        filtered_signal = apply_low_pass_filter(signal, sample_rate, cutoff_freq)
        filtered_signals.append(filtered_signal)

    # Center and reduce
    centered_reduced_filtered_signals = []
    for filtered_signal in filtered_signals:
        centered_reduced_filtered_signal = center_and_reduce_signal(filtered_signal)
        centered_reduced_filtered_signals.append(centered_reduced_filtered_signal)

    # Convert the centered, reduced, and filtered signals into a jnp.ndarray
    np_signals = np.array(centered_reduced_filtered_signals)

    return np_signals


def segmentations_pre_process(
    segmentations: List[List[int]], transformed_signal_size: int, signal_max_length : int, 
) -> Tuple[np.ndarray, List[List[int]], np.ndarray]:
    """
    Pre-process a list of segmentations by resizing and converting them into numpy arrays.

    Args:
        segmentations (List[List[int]]): A list of segmentations represented as lists of integers.
        transformed_signal_size (int): The target size for the transformed segmentations.

    Returns:
        Tuple[np.ndarray, List[List[int]], np.ndarray]: A tuple containing:
            - np_segmentations (np.ndarray): Segmentations as numpy arrays of shape (nb_segmentations, SIGNAL_LENGTH).
            - size_adapted_segmentations (List[List[int]]): Resized segmentations as lists of lists.
            - size_adapted_np_segmentations (np.ndarray): Resized segmentations as numpy arrays of shape (nb_segmentations, transformed_signal_size).
    """

    np_segmentations = np.zeros((len(segmentations), signal_max_length))
    stride = int(signal_max_length / transformed_signal_size)
    size_adapted_segmentations = []
    size_adapted_np_segmentations = np.zeros(
        (len(segmentations), transformed_signal_size)
    )

    for i in range(len(segmentations)):
        np_segmentations[i] = get_segment_ids(np.array(segmentations[i]), signal_max_length)
        size_adapted_segmentations.append(
            [int(element / stride) for element in segmentations[i]]
        )
        size_adapted_np_segmentations[i] = get_segment_ids(
            np.array(size_adapted_segmentations[i]), transformed_signal_size
        )

    np_segmentations = np.array(np_segmentations, dtype=int)
    size_adapted_np_segmentations = np.array(size_adapted_np_segmentations, dtype=int)

    return (
        np_segmentations,
        size_adapted_segmentations,
        size_adapted_np_segmentations,
    )


def signal_to_spectrogram(
    signal: np.ndarray, nperseg: int = 300, noverlap: int = 298, fs: int = 100
) -> np.ndarray:
    """
    Compute the spectrogram of a signal and return the result.

    Args:
        signal (np.ndarray): The input signal to compute the spectrogram.
        nperseg (int, optional): Length of each segment for the spectrogram computation.
        noverlap (int, optional): Number of overlap points between segments.
        fs (int, optional): The sampling frequency of the input signal.

    Returns:
        np.ndarray: The computed spectrogram as a numpy array.
    """
    f_signal, t_signal, Zxx_signal = [], [], []
    for i in range(0, signal.shape[0]):
        f, t, Zxx = stft(signal[i], nperseg=nperseg, noverlap=noverlap, fs=fs)
        arr = np.delete(Zxx, -1, axis=1)  # --> We delete the last column
        f_signal.append(f)
        t_signal.append(t)
        Zxx_signal.append(np.abs(arr).T)

    fourier_transformed_signal_all_dim = np.dstack(
        (Zxx_signal[0], Zxx_signal[1], Zxx_signal[2])
    )
    fourier_transformed_signal = np.linalg.norm(
        fourier_transformed_signal_all_dim, axis=2
    )

    return fourier_transformed_signal


def signals_to_spectrograms(
    signals: np.ndarray, nperseg: int = 300, noverlap: int = 298, fs: int = 100
) -> np.ndarray:
    """
    Compute spectrograms for a list of signals and return the results.

    Args:
        signals (np.ndarray): The input signals as a numpy array of shape (nb_signals, signal_length).
        nperseg (int, optional): Length of each segment for the spectrogram computation.
        noverlap (int, optional): Number of overlap points between segments.
        fs (int, optional): The sampling frequency of the input signals.

    Returns:
        np.ndarray: The computed spectrograms as a numpy array of shape (nb_signals, time_bins, frequency_bins).
    """
    spectrograms = np.zeros(
        (
            len(signals),
            int(signals.shape[1] / (nperseg - noverlap)),
            int(nperseg // 2 + 1),
        )
    )
    for i in range(len(signals)):
        spectrograms[i] = signal_to_spectrogram(signals[i].T, nperseg, noverlap, fs)

    return spectrograms



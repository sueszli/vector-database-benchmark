import fractions
import itertools
import math
import sys
from collections import Counter
import numpy as np
from urh.ainterpretation import Wavelet
from urh.cythonext import auto_interpretation as c_auto_interpretation
from urh.cythonext import signal_functions
from urh.cythonext import util
from urh.signalprocessing.IQArray import IQArray

def max_without_outliers(data: np.ndarray, z=3):
    if False:
        print('Hello World!')
    if len(data) == 0:
        return None
    return np.max(data[abs(data - np.mean(data)) <= z * np.std(data)])

def min_without_outliers(data: np.ndarray, z=2):
    if False:
        print('Hello World!')
    if len(data) == 0:
        return None
    return np.min(data[abs(data - np.mean(data)) <= z * np.std(data)])

def get_most_frequent_value(values: list):
    if False:
        i = 10
        return i + 15
    '\n    Return the most frequent value in list.\n    If there is no unique one, return the maximum of the most frequent values\n\n    :param values:\n    :return:\n    '
    if len(values) == 0:
        return None
    most_common = Counter(values).most_common()
    (result, max_count) = most_common[0]
    for (value, count) in most_common:
        if count < max_count:
            return result
        else:
            result = value
    return result

def most_common(values: list):
    if False:
        i = 10
        return i + 15
    '\n    Return the most common value in a list. In case of ties, return the value that appears first in list\n    :param values:\n    :return:\n    '
    counter = Counter(values)
    return max(values, key=counter.get)

def detect_noise_level(magnitudes):
    if False:
        return 10
    if len(magnitudes) <= 3:
        return 0
    chunksize_percent = 1
    chunksize = max(1, int(len(magnitudes) * chunksize_percent / 100))
    chunks = [magnitudes[i - chunksize:i] for i in range(len(magnitudes), 0, -chunksize) if i - chunksize >= 0]
    mean_values = np.fromiter((np.mean(chunk) for chunk in chunks), dtype=np.float32, count=len(chunks))
    (minimum, maximum) = util.minmax(mean_values)
    if maximum == 0 or minimum / maximum > 0.9:
        return 0
    indices = np.nonzero(mean_values <= 1.1 * np.min(mean_values))[0]
    try:
        result = np.max([np.max(chunks[i]) for i in indices if len(chunks[i]) > 0])
    except ValueError:
        return 0
    return math.ceil(result * 10000) / 10000

def segment_messages_from_magnitudes(magnitudes: np.ndarray, noise_threshold: float):
    if False:
        i = 10
        return i + 15
    '\n    Get the list of start, end indices of messages\n\n    :param magnitudes: Magnitudes of samples\n    :param noise_threshold: Threshold for noise\n    :return:\n    '
    return c_auto_interpretation.segment_messages_from_magnitudes(magnitudes, noise_threshold)

def merge_message_segments_for_ook(segments: list):
    if False:
        print('Hello World!')
    if len(segments) <= 1:
        return segments
    result = []
    pauses = np.fromiter((segments[i + 1][0] - segments[i][1] for i in range(len(segments) - 1)), count=len(segments) - 1, dtype=np.uint64)
    pulses = np.fromiter((segments[i][1] - segments[i][0] for i in range(len(segments))), count=len(segments), dtype=np.uint64)
    min_pulse_length = min_without_outliers(pulses, z=1)
    large_pause_indices = np.nonzero(pauses >= 8 * min_pulse_length)[0]
    for i in range(0, len(large_pause_indices) + 1):
        if i == 0:
            (start, end) = (0, large_pause_indices[i] + 1 if len(large_pause_indices) >= 1 else len(segments))
        elif i == len(large_pause_indices):
            (start, end) = (large_pause_indices[i - 1] + 1, len(segments))
        else:
            (start, end) = (large_pause_indices[i - 1] + 1, large_pause_indices[i] + 1)
        msg_begin = segments[start][0]
        msg_length = sum((segments[j][1] - segments[j][0] for j in range(start, end)))
        msg_length += sum((segments[j][0] - segments[j - 1][1] for j in range(start + 1, end)))
        result.append((msg_begin, msg_begin + msg_length))
    return result

def detect_modulation(data: np.ndarray, wavelet_scale=4, median_filter_order=11) -> str:
    if False:
        i = 10
        return i + 15
    n_data = len(data)
    data = data[np.abs(data) > 0]
    if len(data) == 0:
        return None
    if n_data - len(data) > 3:
        return 'OOK'
    data = data / np.abs(np.max(data))
    mag_wavlt = np.abs(Wavelet.cwt_haar(data, scale=wavelet_scale))
    if len(mag_wavlt) == 0:
        return None
    norm_mag_wavlt = np.abs(Wavelet.cwt_haar(data / np.abs(data), scale=wavelet_scale))
    var_mag = np.var(mag_wavlt)
    var_norm_mag = np.var(norm_mag_wavlt)
    var_filtered_mag = np.var(c_auto_interpretation.median_filter(mag_wavlt, k=median_filter_order))
    var_filtered_norm_mag = np.var(c_auto_interpretation.median_filter(norm_mag_wavlt, k=median_filter_order))
    if all((v < 0.15 for v in (var_mag, var_norm_mag, var_filtered_mag, var_filtered_norm_mag))):
        return 'OOK'
    if var_mag > 1.5 * var_norm_mag:
        return 'ASK'
    elif var_mag > 10 * var_filtered_mag:
        return 'PSK'
    else:
        fft = np.fft.fft(data[0:2 ** int(np.log2(len(data)))])
        fft = np.abs(np.fft.fftshift(fft))
        ten_greatest_indices = np.argsort(fft)[::-1][0:10]
        greatest_index = ten_greatest_indices[0]
        min_distance = 10
        min_freq = 100
        if any((abs(i - greatest_index) >= min_distance and fft[i] >= min_freq for i in ten_greatest_indices)):
            return 'FSK'
        else:
            return 'OOK'

def detect_modulation_for_messages(signal: IQArray, message_indices: list) -> str:
    if False:
        return 10
    max_messages = 100
    modulations_for_messages = []
    complex = signal.as_complex64()
    for (start, end) in message_indices[0:max_messages]:
        mod = detect_modulation(complex[start:end])
        if mod is not None:
            modulations_for_messages.append(mod)
    if len(modulations_for_messages) == 0:
        return None
    return most_common(modulations_for_messages)

def detect_center(rectangular_signal: np.ndarray, max_size=None):
    if False:
        for i in range(10):
            print('nop')
    rect = rectangular_signal[rectangular_signal > -4]
    rect = rect[int(0.05 * len(rect)):int(0.95 * len(rect))]
    if max_size is not None and len(rect) > max_size:
        rect = rect[0:max_size]
    (hist_min, hist_max) = util.minmax(rect)
    hist_step = float(np.var(rect))
    try:
        (y, x) = np.histogram(rect, bins=np.arange(hist_min, hist_max + hist_step, hist_step))
    except (ZeroDivisionError, ValueError):
        return None
    num_values = 2
    most_common_levels = []
    window_size = max(2, int(0.05 * len(y)) + 1)

    def get_elem(arr, index: int, default):
        if False:
            i = 10
            return i + 15
        if 0 <= index < len(arr):
            return arr[index]
        else:
            return default
    for index in np.argsort(y)[::-1]:
        if all((y[index] > get_elem(y, index + i, 0) and y[index] > get_elem(y, index - i, 0) for i in range(1, window_size))):
            most_common_levels.append(x[index])
        if len(most_common_levels) == num_values:
            break
    if len(most_common_levels) == 0:
        return None
    return np.mean(most_common_levels)

def estimate_tolerance_from_plateau_lengths(plateau_lengths, relative_max=0.05) -> int:
    if False:
        print('Hello World!')
    if len(plateau_lengths) <= 1:
        return None
    unique = np.unique(plateau_lengths)
    maximum = max_without_outliers(unique, z=2)
    limit = relative_max * maximum
    if unique[0] > 1 and unique[0] >= limit:
        return 0
    result = 0
    for value in unique:
        if value > 1 and value >= limit:
            break
        result = value
    return result

def merge_plateau_lengths(plateau_lengths, tolerance=None) -> list:
    if False:
        while True:
            i = 10
    if tolerance is None:
        tolerance = estimate_tolerance_from_plateau_lengths(plateau_lengths)
    if tolerance == 0 or tolerance is None:
        return plateau_lengths
    return c_auto_interpretation.merge_plateaus(plateau_lengths, tolerance, max_count=10000)

def round_plateau_lengths(plateau_lengths: list):
    if False:
        while True:
            i = 10
    '\n    Round plateau lengths to next divisible number of digit count e.g. 99 -> 100, 293 -> 300\n\n    :param plateau_lengths:\n    :return:\n    '
    digit_counts = [len(str(p)) for p in plateau_lengths]
    n_digits = min(3, int(np.percentile(digit_counts, 50)))
    f = 10 ** (n_digits - 1)
    for (i, plateau_len) in enumerate(plateau_lengths):
        plateau_lengths[i] = int(round(plateau_len / f)) * f

def get_tolerant_greatest_common_divisor(numbers):
    if False:
        i = 10
        return i + 15
    '\n    Get the greatest common divisor of the numbers in a tolerant manner:\n    Calculate each gcd of each pair of numbers and return the most common one\n\n    '
    gcd = math.gcd if sys.version_info >= (3, 5) else fractions.gcd
    gcds = [gcd(x, y) for (x, y) in itertools.combinations(numbers, 2) if gcd(x, y) != 1]
    if len(gcds) == 0:
        return 1
    return get_most_frequent_value(gcds)

def get_bit_length_from_plateau_lengths(merged_plateau_lengths) -> int:
    if False:
        for i in range(10):
            print('nop')
    if len(merged_plateau_lengths) == 0:
        return 0
    if len(merged_plateau_lengths) == 1:
        return int(merged_plateau_lengths[0])
    round_plateau_lengths(merged_plateau_lengths)
    histogram = c_auto_interpretation.get_threshold_divisor_histogram(merged_plateau_lengths)
    if len(histogram) == 0:
        return 0
    else:
        sorted_indices = np.argsort(histogram)[::-1]
        max_count = histogram[sorted_indices[0]]
        result = sorted_indices[0]
        for i in range(1, len(sorted_indices)):
            if histogram[sorted_indices[i]] < 0.25 * max_count:
                break
            if sorted_indices[i] <= 0.5 * result:
                result = sorted_indices[i]
        return int(result)

def estimate(iq_array: IQArray, noise: float=None, modulation: str=None) -> dict:
    if False:
        return 10
    if isinstance(iq_array, np.ndarray):
        iq_array = IQArray(iq_array)
    magnitudes = iq_array.magnitudes
    noise = detect_noise_level(magnitudes) if noise is None else noise
    message_indices = segment_messages_from_magnitudes(magnitudes, noise_threshold=noise)
    modulation = detect_modulation_for_messages(iq_array, message_indices) if modulation is None else modulation
    if modulation is None:
        return None
    if modulation == 'OOK':
        message_indices = merge_message_segments_for_ook(message_indices)
    if modulation == 'OOK' or modulation == 'ASK':
        data = signal_functions.afp_demod(iq_array.data, noise, 'ASK', 2)
    elif modulation == 'FSK':
        data = signal_functions.afp_demod(iq_array.data, noise, 'FSK', 2)
    elif modulation == 'PSK':
        data = signal_functions.afp_demod(iq_array.data, noise, 'PSK', 2)
    else:
        raise ValueError('Unsupported Modulation')
    centers = []
    bit_lengths = []
    tolerances = []
    for (start, end) in message_indices:
        msg_rect_data = data[start:end]
        center = detect_center(msg_rect_data)
        if center is None:
            continue
        plateau_lengths = c_auto_interpretation.get_plateau_lengths(msg_rect_data, center, percentage=25)
        tolerance = estimate_tolerance_from_plateau_lengths(plateau_lengths)
        if tolerance is None:
            tolerance = 0
        else:
            tolerances.append(tolerance)
        merged_lengths = merge_plateau_lengths(plateau_lengths, tolerance=tolerance)
        if len(merged_lengths) < 2:
            continue
        bit_length = get_bit_length_from_plateau_lengths(merged_lengths)
        min_bit_length = tolerance + 1
        if bit_length > min_bit_length:
            centers.append(center)
            bit_lengths.append(bit_length)
    if modulation == 'OOK' or modulation == 'ASK':
        center = min_without_outliers(np.array(centers), z=2)
        if center is None:
            return None
    elif len(centers) > 0:
        center = np.mean(centers)
    else:
        return None
    bit_length = get_most_frequent_value(bit_lengths)
    if bit_length is None:
        return None
    try:
        tolerance = np.percentile(tolerances, 50)
    except IndexError:
        tolerance = max(1, int(0.05 * bit_length))
    result = {'modulation_type': 'ASK' if modulation == 'OOK' else modulation, 'bit_length': bit_length, 'center': center, 'tolerance': int(tolerance), 'noise': noise}
    return result
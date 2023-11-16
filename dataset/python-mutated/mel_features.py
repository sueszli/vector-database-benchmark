"""Defines routines to compute mel spectrogram features from audio waveform."""
import numpy as np

def frame(data, window_length, hop_length):
    if False:
        i = 10
        return i + 15
    'Convert array into a sequence of successive possibly overlapping frames.\n\n  An n-dimensional array of shape (num_samples, ...) is converted into an\n  (n+1)-D array of shape (num_frames, window_length, ...), where each frame\n  starts hop_length points after the preceding one.\n\n  This is accomplished using stride_tricks, so the original data is not\n  copied.  However, there is no zero-padding, so any incomplete frames at the\n  end are not included.\n\n  Args:\n    data: np.array of dimension N >= 1.\n    window_length: Number of samples in each frame.\n    hop_length: Advance (in samples) between each window.\n\n  Returns:\n    (N+1)-D np.array with as many rows as there are complete frames that can be\n    extracted.\n  '
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def periodic_hann(window_length):
    if False:
        print('Hello World!')
    'Calculate a "periodic" Hann window.\n\n  The classic Hann window is defined as a raised cosine that starts and\n  ends on zero, and where every value appears twice, except the middle\n  point for an odd-length window.  Matlab calls this a "symmetric" window\n  and np.hanning() returns it.  However, for Fourier analysis, this\n  actually represents just over one cycle of a period N-1 cosine, and\n  thus is not compactly expressed on a length-N Fourier basis.  Instead,\n  it\'s better to use a raised cosine that ends just before the final\n  zero value - i.e. a complete cycle of a period-N cosine.  Matlab\n  calls this a "periodic" window. This routine calculates it.\n\n  Args:\n    window_length: The number of points in the returned window.\n\n  Returns:\n    A 1D np.array containing the periodic hann window.\n  '
    return 0.5 - 0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length))

def stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    if False:
        print('Hello World!')
    'Calculate the short-time Fourier transform magnitude.\n\n  Args:\n    signal: 1D np.array of the input time-domain signal.\n    fft_length: Size of the FFT to apply.\n    hop_length: Advance (in samples) between each frame passed to FFT.\n    window_length: Length of each block of samples to pass to FFT.\n\n  Returns:\n    2D np.array where each row contains the magnitudes of the fft_length/2+1\n    unique values of the FFT for the corresponding frame of input samples.\n  '
    frames = frame(signal, window_length, hop_length)
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def hertz_to_mel(frequencies_hertz):
    if False:
        i = 10
        return i + 15
    'Convert frequencies to mel scale using HTK formula.\n\n  Args:\n    frequencies_hertz: Scalar or np.array of frequencies in hertz.\n\n  Returns:\n    Object of same size as frequencies_hertz containing corresponding values\n    on the mel scale.\n  '
    return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)

def spectrogram_to_mel_matrix(num_mel_bins=20, num_spectrogram_bins=129, audio_sample_rate=8000, lower_edge_hertz=125.0, upper_edge_hertz=3800.0):
    if False:
        while True:
            i = 10
    'Return a matrix that can post-multiply spectrogram rows to make mel.\n\n  Returns a np.array matrix A that can be used to post-multiply a matrix S of\n  spectrogram values (STFT magnitudes) arranged as frames x bins to generate a\n  "mel spectrogram" M of frames x num_mel_bins.  M = S A.\n\n  The classic HTK algorithm exploits the complementarity of adjacent mel bands\n  to multiply each FFT bin by only one mel weight, then add it, with positive\n  and negative signs, to the two adjacent mel bands to which that bin\n  contributes.  Here, by expressing this operation as a matrix multiply, we go\n  from num_fft multiplies per frame (plus around 2*num_fft adds) to around\n  num_fft^2 multiplies and adds.  However, because these are all presumably\n  accomplished in a single call to np.dot(), it\'s not clear which approach is\n  faster in Python.  The matrix multiplication has the attraction of being more\n  general and flexible, and much easier to read.\n\n  Args:\n    num_mel_bins: How many bands in the resulting mel spectrum.  This is\n      the number of columns in the output matrix.\n    num_spectrogram_bins: How many bins there are in the source spectrogram\n      data, which is understood to be fft_size/2 + 1, i.e. the spectrogram\n      only contains the nonredundant FFT bins.\n    audio_sample_rate: Samples per second of the audio at the input to the\n      spectrogram. We need this to figure out the actual frequencies for\n      each spectrogram bin, which dictates how they are mapped into mel.\n    lower_edge_hertz: Lower bound on the frequencies to be included in the mel\n      spectrum.  This corresponds to the lower edge of the lowest triangular\n      band.\n    upper_edge_hertz: The desired top edge of the highest frequency band.\n\n  Returns:\n    An np.array with shape (num_spectrogram_bins, num_mel_bins).\n\n  Raises:\n    ValueError: if frequency edges are incorrectly ordered or out of range.\n  '
    nyquist_hertz = audio_sample_rate / 2.0
    if lower_edge_hertz < 0.0:
        raise ValueError('lower_edge_hertz %.1f must be >= 0' % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' % (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError('upper_edge_hertz %.1f is greater than Nyquist %.1f' % (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        (lower_edge_mel, center_mel, upper_edge_mel) = band_edges_mel[i:i + 3]
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix

def log_mel_spectrogram(data, audio_sample_rate=8000, log_offset=0.0, window_length_secs=0.025, hop_length_secs=0.01, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Convert waveform to a log magnitude mel-frequency spectrogram.\n\n  Args:\n    data: 1D np.array of waveform data.\n    audio_sample_rate: The sampling rate of data.\n    log_offset: Add this to values when taking log to avoid -Infs.\n    window_length_secs: Duration of each window to analyze.\n    hop_length_secs: Advance between successive analysis windows.\n    **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.\n\n  Returns:\n    2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank\n    magnitudes for successive frames.\n  '
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    spectrogram = stft_magnitude(data, fft_length=fft_length, hop_length=hop_length_samples, window_length=window_length_samples)
    mel_spectrogram = np.dot(spectrogram, spectrogram_to_mel_matrix(num_spectrogram_bins=spectrogram.shape[1], audio_sample_rate=audio_sample_rate, **kwargs))
    return np.log(mel_spectrogram + log_offset)
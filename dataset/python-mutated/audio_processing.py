import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util

def window_sumsquare(window, n_frames, hop_length=200, win_length=800, n_fft=800, dtype=np.float32, norm=None):
    if False:
        return 10
    '\n    # from librosa 0.6\n    Compute the sum-square envelope of a window function at a given hop length.\n\n    This is used to estimate modulation effects induced by windowing\n    observations in short-time fourier transforms.\n\n    Parameters\n    ----------\n    window : string, tuple, number, callable, or list-like\n        Window specification, as in `get_window`\n\n    n_frames : int > 0\n        The number of analysis frames\n\n    hop_length : int > 0\n        The number of samples to advance between frames\n\n    win_length : [optional]\n        The length of the window function.  By default, this matches `n_fft`.\n\n    n_fft : int > 0\n        The length of each analysis frame.\n\n    dtype : np.dtype\n        The data type of the output\n\n    Returns\n    -------\n    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`\n        The sum-squared envelope of the window function\n    '
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

def griffin_lim(magnitudes, stft_fn, n_iters=30):
    if False:
        i = 10
        return i + 15
    '\n    PARAMS\n    ------\n    magnitudes: spectrogram magnitudes\n    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods\n    '
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    for i in range(n_iters):
        (_, angles) = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal

def dynamic_range_compression(x, C=1, clip_val=1e-05):
    if False:
        i = 10
        return i + 15
    '\n    PARAMS\n    ------\n    C: compression factor\n    '
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    if False:
        print('Hello World!')
    '\n    PARAMS\n    ------\n    C: compression factor used to compress\n    '
    return torch.exp(x) / C
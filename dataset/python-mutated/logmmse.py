import numpy as np
import math
from scipy.special import expn
from collections import namedtuple
NoiseProfile = namedtuple('NoiseProfile', 'sampling_rate window_size len1 len2 win n_fft noise_mu2')

def profile_noise(noise, sampling_rate, window_size=0):
    if False:
        i = 10
        return i + 15
    '\n    Creates a profile of the noise in a given waveform.\n    \n    :param noise: a waveform containing noise ONLY, as a numpy array of floats or ints. \n    :param sampling_rate: the sampling rate of the audio\n    :param window_size: the size of the window the logmmse algorithm operates on. A default value \n    will be picked if left as 0.\n    :return: a NoiseProfile object\n    '
    (noise, dtype) = to_float(noise)
    noise += np.finfo(np.float64).eps
    if window_size == 0:
        window_size = int(math.floor(0.02 * sampling_rate))
    if window_size % 2 == 1:
        window_size = window_size + 1
    perc = 50
    len1 = int(math.floor(window_size * perc / 100))
    len2 = int(window_size - len1)
    win = np.hanning(window_size)
    win = win * len2 / np.sum(win)
    n_fft = 2 * window_size
    noise_mean = np.zeros(n_fft)
    n_frames = len(noise) // window_size
    for j in range(0, window_size * n_frames, window_size):
        noise_mean += np.absolute(np.fft.fft(win * noise[j:j + window_size], n_fft, axis=0))
    noise_mu2 = (noise_mean / n_frames) ** 2
    return NoiseProfile(sampling_rate, window_size, len1, len2, win, n_fft, noise_mu2)

def denoise(wav, noise_profile: NoiseProfile, eta=0.15):
    if False:
        return 10
    '\n    Cleans the noise from a speech waveform given a noise profile. The waveform must have the \n    same sampling rate as the one used to create the noise profile. \n    \n    :param wav: a speech waveform as a numpy array of floats or ints.\n    :param noise_profile: a NoiseProfile object that was created from a similar (or a segment of \n    the same) waveform.\n    :param eta: voice threshold for noise update. While the voice activation detection value is \n    below this threshold, the noise profile will be continuously updated throughout the audio. \n    Set to 0 to disable updating the noise profile.\n    :return: the clean wav as a numpy array of floats or ints of the same length.\n    '
    (wav, dtype) = to_float(wav)
    wav += np.finfo(np.float64).eps
    p = noise_profile
    nframes = int(math.floor(len(wav) / p.len2) - math.floor(p.window_size / p.len2))
    x_final = np.zeros(nframes * p.len2)
    aa = 0.98
    mu = 0.98
    ksi_min = 10 ** (-25 / 10)
    x_old = np.zeros(p.len1)
    xk_prev = np.zeros(p.len1)
    noise_mu2 = p.noise_mu2
    for k in range(0, nframes * p.len2, p.len2):
        insign = p.win * wav[k:k + p.window_size]
        spec = np.fft.fft(insign, p.n_fft, axis=0)
        sig = np.absolute(spec)
        sig2 = sig ** 2
        gammak = np.minimum(sig2 / noise_mu2, 40)
        if xk_prev.all() == 0:
            ksi = aa + (1 - aa) * np.maximum(gammak - 1, 0)
        else:
            ksi = aa * xk_prev / noise_mu2 + (1 - aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(ksi_min, ksi)
        log_sigma_k = gammak * ksi / (1 + ksi) - np.log(1 + ksi)
        vad_decision = np.sum(log_sigma_k) / p.window_size
        if vad_decision < eta:
            noise_mu2 = mu * noise_mu2 + (1 - mu) * sig2
        a = ksi / (1 + ksi)
        vk = a * gammak
        ei_vk = 0.5 * expn(1, np.maximum(vk, 1e-08))
        hw = a * np.exp(ei_vk)
        sig = sig * hw
        xk_prev = sig ** 2
        xi_w = np.fft.ifft(hw * spec, p.n_fft, axis=0)
        xi_w = np.real(xi_w)
        x_final[k:k + p.len2] = x_old + xi_w[0:p.len1]
        x_old = xi_w[p.len1:p.window_size]
    output = from_float(x_final, dtype)
    output = np.pad(output, (0, len(wav) - len(output)), mode='constant')
    return output

def to_float(_input):
    if False:
        i = 10
        return i + 15
    if _input.dtype == np.float64:
        return (_input, _input.dtype)
    elif _input.dtype == np.float32:
        return (_input.astype(np.float64), _input.dtype)
    elif _input.dtype == np.uint8:
        return ((_input - 128) / 128.0, _input.dtype)
    elif _input.dtype == np.int16:
        return (_input / 32768.0, _input.dtype)
    elif _input.dtype == np.int32:
        return (_input / 2147483648.0, _input.dtype)
    raise ValueError('Unsupported wave file format')

def from_float(_input, dtype):
    if False:
        return 10
    if dtype == np.float64:
        return (_input, np.float64)
    elif dtype == np.float32:
        return _input.astype(np.float32)
    elif dtype == np.uint8:
        return (_input * 128 + 128).astype(np.uint8)
    elif dtype == np.int16:
        return (_input * 32768).astype(np.int16)
    elif dtype == np.int32:
        print(_input)
        return (_input * 2147483648).astype(np.int32)
    raise ValueError('Unsupported wave file format')
"""Utility class for extracting features from the text and audio input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
import numpy as np

def compute_spectrogram_feature(samples, sample_rate, stride_ms=10.0, window_ms=20.0, max_freq=None, eps=1e-14):
    if False:
        i = 10
        return i + 15
    'Compute the spectrograms for the input samples(waveforms).\n\n  More about spectrogram computation, please refer to:\n  https://en.wikipedia.org/wiki/Short-time_Fourier_transform.\n  '
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError('max_freq must not be greater than half of sample rate.')
    if stride_ms > window_ms:
        raise ValueError('Stride size must not be greater than window size.')
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)
    assert np.all(windows[:, 1] == samples[stride_size:stride_size + window_size])
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= 2.0 / scale
    fft[(0, -1), :] /= scale
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return np.transpose(specgram, (1, 0))

class AudioFeaturizer(object):
    """Class to extract spectrogram features from the audio input."""

    def __init__(self, sample_rate=16000, window_ms=20.0, stride_ms=10.0):
        if False:
            while True:
                i = 10
        'Initialize the audio featurizer class according to the configs.\n\n    Args:\n      sample_rate: an integer specifying the sample rate of the input waveform.\n      window_ms: an integer for the length of a spectrogram frame, in ms.\n      stride_ms: an integer for the frame stride, in ms.\n    '
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.stride_ms = stride_ms

def compute_label_feature(text, token_to_idx):
    if False:
        print('Hello World!')
    'Convert string to a list of integers.'
    tokens = list(text.strip().lower())
    feats = [token_to_idx[token] for token in tokens]
    return feats

class TextFeaturizer(object):
    """Extract text feature based on char-level granularity.

  By looking up the vocabulary table, each input string (one line of transcript)
  will be converted to a sequence of integer indexes.
  """

    def __init__(self, vocab_file):
        if False:
            i = 10
            return i + 15
        lines = []
        with codecs.open(vocab_file, 'r', 'utf-8') as fin:
            lines.extend(fin.readlines())
        self.token_to_index = {}
        self.index_to_token = {}
        self.speech_labels = ''
        index = 0
        for line in lines:
            line = line[:-1]
            if line.startswith('#'):
                continue
            self.token_to_index[line] = index
            self.index_to_token[index] = line
            self.speech_labels += line
            index += 1
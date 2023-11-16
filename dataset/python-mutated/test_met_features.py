"""Regression tests on metlab features"""
import os
import pytest
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass
import numpy as np
import scipy.io
import scipy.signal
from test_core import load, files
import librosa
__EXAMPLE_FILE = os.path.join('tests', 'data', 'test1_22050.wav')

def met_stft(y, n_fft, hop_length, win_length, normalize):
    if False:
        print('Hello World!')
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=scipy.signal.hamming, center=False))
    if normalize:
        S = S / (S[0] + np.sum(2 * S[1:], axis=0))
    return S

@pytest.mark.parametrize('infile', files(os.path.join('tests', 'data', 'met-centroid-*.mat')))
def test_spectral_centroid(infile):
    if False:
        for i in range(10):
            print('nop')
    DATA = load(infile)
    (y, sr) = librosa.load(os.path.join('tests', DATA['wavfile'][0]), sr=None, mono=True)
    n_fft = DATA['nfft'][0, 0].astype(int)
    hop_length = DATA['hop_length'][0, 0].astype(int)
    S = met_stft(y, n_fft, hop_length, n_fft, True)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
    assert np.allclose(centroid, DATA['centroid'])

@pytest.mark.parametrize('infile', files(os.path.join('tests', 'data', 'met-contrast-*.mat')))
def test_spectral_contrast(infile):
    if False:
        return 10
    DATA = load(infile)
    (y, sr) = librosa.load(os.path.join('tests', DATA['wavfile'][0]), sr=None, mono=True)
    n_fft = DATA['nfft'][0, 0].astype(int)
    hop_length = DATA['hop_length'][0, 0].astype(int)
    S = met_stft(y, n_fft, hop_length, n_fft, True)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, linear=True)
    assert np.allclose(contrast, DATA['contrast'], rtol=0.001, atol=0.01)

@pytest.mark.parametrize('infile', files(os.path.join('tests', 'data', 'met-rolloff-*.mat')))
def test_spectral_rolloff(infile):
    if False:
        return 10
    DATA = load(infile)
    (y, sr) = librosa.load(os.path.join('tests', DATA['wavfile'][0]), sr=None, mono=True)
    n_fft = DATA['nfft'][0, 0].astype(int)
    hop_length = DATA['hop_length'][0, 0].astype(int)
    pct = DATA['pct'][0, 0]
    S = met_stft(y, n_fft, hop_length, n_fft, True)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=pct)
    assert np.allclose(rolloff, DATA['rolloff'])

@pytest.mark.parametrize('infile', files(os.path.join('tests', 'data', 'met-bandwidth-*.mat')))
def test_spectral_bandwidth(infile):
    if False:
        print('Hello World!')
    DATA = load(infile)
    (y, sr) = librosa.load(os.path.join('tests', DATA['wavfile'][0]), sr=None, mono=True)
    n_fft = DATA['nfft'][0, 0].astype(int)
    hop_length = DATA['hop_length'][0, 0].astype(int)
    S = DATA['S']
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, centroid=DATA['centroid'], norm=False, p=1)
    assert np.allclose(bw, S.shape[0] * DATA['bw'])
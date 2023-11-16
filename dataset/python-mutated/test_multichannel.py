import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass
import librosa
import glob
import numpy as np
import scipy.io
import scipy.stats
import pytest
import warnings
from unittest import mock
from typing import List, Union
from contextlib import nullcontext as dnr
from test_core import srand

@pytest.fixture(scope='module', params=['test1_44100.wav'])
def y_multi(request):
    if False:
        print('Hello World!')
    infile = request.param
    return librosa.load(os.path.join('tests', 'data', infile), sr=None, mono=False)

@pytest.fixture(scope='module')
def s_multi(y_multi):
    if False:
        for i in range(10):
            print('nop')
    (y, sr) = y_multi
    return (np.abs(librosa.stft(y)), sr)

@pytest.fixture(scope='module')
def tfr_multi(y_multi):
    if False:
        print('Hello World!')
    (y, sr) = y_multi
    return librosa.reassigned_spectrogram(y, fill_nan=True)

@pytest.mark.parametrize('aggregate', [None, np.mean, np.sum])
@pytest.mark.parametrize('ndim,axis', [(1, 0), (1, -1), (2, 0), (2, 1), (2, -1), (3, 0), (3, 2), (3, -1), (4, 0), (4, 3), (4, -1)])
def test_sync_multi(aggregate, ndim: int, axis: int):
    if False:
        return 10
    data = np.ones([6] * ndim, dtype=float)
    slices = [slice(1, 3), slice(3, 4)]
    dsync = librosa.util.sync(data, slices, aggregate=aggregate, axis=axis)
    assert dsync.shape[axis] == len(slices)
    s_test = list(dsync.shape)
    del s_test[axis]
    s_orig = list(data.shape)
    del s_orig[axis]
    assert s_test == s_orig
    idx: List[Union[slice, int]] = [slice(None)] * ndim
    idx[axis] = 0
    if aggregate is np.sum:
        assert np.allclose(dsync[tuple(idx)], 2)
    else:
        assert np.allclose(dsync[tuple(idx)], 1)
    idx[axis] = 1
    assert np.allclose(dsync[tuple(idx)], 1)

def test_stft_multi(y_multi):
    if False:
        return 10
    (y, sr) = y_multi
    D = librosa.stft(y)
    D0 = librosa.stft(y[0])
    D1 = librosa.stft(y[1])
    assert np.allclose(D[0], D0)
    assert np.allclose(D[1], D1)
    assert not np.allclose(D0, D1)

def test_onset_strength(y_multi):
    if False:
        print('Hello World!')
    (y, sr) = y_multi
    S = librosa.stft(y)
    D = librosa.onset.onset_strength(S=S)
    D0 = librosa.onset.onset_strength(S=S[0])
    D1 = librosa.onset.onset_strength(S=S[1])
    assert np.allclose(D[0], D0)
    assert np.allclose(D[1], D1)
    assert not np.allclose(D0, D1)

def test_tempogram(s_multi):
    if False:
        return 10
    (S, sr) = s_multi
    D = librosa.onset.onset_strength(S=S)
    t = librosa.feature.tempogram(y=None, sr=sr, onset_envelope=D, hop_length=512)
    D0 = librosa.onset.onset_strength(S=S[0])
    D1 = librosa.onset.onset_strength(S=S[1])
    t0 = librosa.feature.tempogram(y=None, sr=sr, onset_envelope=D0, hop_length=512)
    t1 = librosa.feature.tempogram(y=None, sr=sr, onset_envelope=D1, hop_length=512)
    assert np.allclose(t[0], t0)
    assert np.allclose(t[1], t1)
    assert not np.allclose(t0, t1)

def test_fourier_tempogram(s_multi):
    if False:
        return 10
    (S, sr) = s_multi
    D = librosa.onset.onset_strength(S=S)
    t = librosa.feature.fourier_tempogram(sr=sr, onset_envelope=D)
    D0 = librosa.onset.onset_strength(S=S[0])
    D1 = librosa.onset.onset_strength(S=S[1])
    t0 = librosa.feature.fourier_tempogram(sr=sr, onset_envelope=D0)
    t1 = librosa.feature.fourier_tempogram(sr=sr, onset_envelope=D1)
    assert np.allclose(t[0], t0, atol=1e-06, rtol=1e-06)
    assert np.allclose(t[1], t1, atol=1e-06, rtol=1e-06)
    assert not np.allclose(t0, t1, atol=1e-06, rtol=1e-06)

def test_tempo_multi(y_multi):
    if False:
        i = 10
        return i + 15
    sr = 22050
    tempi = [78, 128]
    y = np.zeros((2, 20 * sr))
    delay = [librosa.time_to_samples(60 / tempo, sr=sr).item() for tempo in tempi]
    y[0, ::delay[0]] = 1
    y[1, ::delay[1]] = 1
    t = librosa.feature.tempo(y=y, sr=sr, hop_length=512, ac_size=4, aggregate=np.mean, prior=None)
    t0 = librosa.feature.tempo(y=y[0], sr=sr, hop_length=512, ac_size=4, aggregate=np.mean, prior=None)
    t1 = librosa.feature.tempo(y=y[1], sr=sr, hop_length=512, ac_size=4, aggregate=np.mean, prior=None)
    assert np.allclose(t[0], t0)
    assert np.allclose(t[1], t1)
    assert not np.allclose(t0, t1)

@pytest.mark.parametrize('hop_length', [512])
@pytest.mark.parametrize('win_length', [384])
@pytest.mark.parametrize('tempo_min,tempo_max', [(30, 300), (60, None)])
@pytest.mark.parametrize('prior', [None, scipy.stats.lognorm(s=1, loc=np.log(120), scale=120)])
def test_plp_multi(s_multi, hop_length, win_length, tempo_min, tempo_max, prior):
    if False:
        for i in range(10):
            print('nop')
    (S, sr) = s_multi
    D = librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop_length)
    D0 = librosa.onset.onset_strength(S=S[0], sr=sr, hop_length=hop_length)
    D1 = librosa.onset.onset_strength(S=S[1], sr=sr, hop_length=hop_length)
    pulse = librosa.beat.plp(sr=sr, onset_envelope=D, hop_length=hop_length, win_length=win_length, tempo_min=tempo_min, tempo_max=tempo_max, prior=prior)
    pulse0 = librosa.beat.plp(sr=sr, onset_envelope=D0, hop_length=hop_length, win_length=win_length, tempo_min=tempo_min, tempo_max=tempo_max, prior=prior)
    pulse1 = librosa.beat.plp(sr=sr, onset_envelope=D1, hop_length=hop_length, win_length=win_length, tempo_min=tempo_min, tempo_max=tempo_max, prior=prior)
    assert np.allclose(pulse[0], pulse0, atol=1e-06, rtol=1e-06)
    assert np.allclose(pulse[1], pulse1, atol=1e-06, rtol=1e-06)
    assert not np.allclose(pulse0, pulse1, atol=1e-06, rtol=1e-06)

def test_istft_multi(y_multi):
    if False:
        print('Hello World!')
    (y, sr) = y_multi
    D = librosa.stft(y)
    y0m = librosa.istft(D[0])
    y1m = librosa.istft(D[1])
    ys = librosa.istft(D)
    assert np.allclose(y0m, ys[0])
    assert np.allclose(y1m, ys[1])
    assert not np.allclose(ys[0], ys[1])

def test_griffinlim_multi(y_multi):
    if False:
        return 10
    (y, sr) = y_multi
    D = librosa.stft(y)
    yout = librosa.griffinlim(np.abs(D), n_iter=2, length=y.shape[-1])
    assert np.allclose(y.shape, yout.shape)

@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('res_type', [None, 'polyphase'])
def test_cqt_multi(y_multi, scale, res_type):
    if False:
        print('Hello World!')
    (y, sr) = y_multi
    C0 = librosa.cqt(y=y[0], sr=sr, scale=scale, res_type=res_type)
    C1 = librosa.cqt(y=y[1], sr=sr, scale=scale, res_type=res_type)
    Call = librosa.cqt(y=y, sr=sr, scale=scale, res_type=res_type)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('res_type', [None, 'polyphase'])
def test_hybrid_cqt_multi(y_multi, scale, res_type):
    if False:
        return 10
    (y, sr) = y_multi
    C0 = librosa.hybrid_cqt(y=y[0], sr=sr, scale=scale, res_type=res_type)
    C1 = librosa.hybrid_cqt(y=y[1], sr=sr, scale=scale, res_type=res_type)
    Call = librosa.hybrid_cqt(y=y, sr=sr, scale=scale, res_type=res_type)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('length', [None, 22050])
def test_icqt_multi(y_multi, scale, length):
    if False:
        return 10
    (y, sr) = y_multi
    C = librosa.cqt(y=y, sr=sr, scale=scale)
    yboth = librosa.icqt(C, sr=sr, scale=scale, length=length)
    y0 = librosa.icqt(C[0], sr=sr, scale=scale, length=length)
    y1 = librosa.icqt(C[1], sr=sr, scale=scale, length=length)
    if length is not None:
        assert yboth.shape[-1] == length
    assert np.allclose(yboth[0], y0)
    assert np.allclose(yboth[1], y1)
    assert not np.allclose(yboth[0], yboth[1])

def test_griffinlim_cqt_multi(y_multi):
    if False:
        return 10
    (y, sr) = y_multi
    C = librosa.cqt(y, sr=sr)
    yout = librosa.griffinlim_cqt(np.abs(C), n_iter=2, length=y.shape[-1])
    assert np.allclose(y.shape, yout.shape)

def test_spectral_centroid_multi(s_multi):
    if False:
        print('Hello World!')
    (S, sr) = s_multi
    freq = None
    C0 = librosa.feature.spectral_centroid(sr=sr, freq=freq, S=S[0])
    C1 = librosa.feature.spectral_centroid(sr=sr, freq=freq, S=S[1])
    Call = librosa.feature.spectral_centroid(sr=sr, freq=freq, S=S)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_spectral_centroid_multi_variable(s_multi):
    if False:
        return 10
    (S, sr) = s_multi
    freq = np.asarray(np.random.randn(*S.shape))
    C0 = librosa.feature.spectral_centroid(sr=sr, freq=freq[0], S=S[0])
    C1 = librosa.feature.spectral_centroid(sr=sr, freq=freq[1], S=S[1])
    Call = librosa.feature.spectral_centroid(sr=sr, freq=freq, S=S)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_spectral_bandwidth_multi(s_multi):
    if False:
        while True:
            i = 10
    (S, sr) = s_multi
    freq = None
    C0 = librosa.feature.spectral_bandwidth(sr=sr, freq=freq, S=S[0])
    C1 = librosa.feature.spectral_bandwidth(sr=sr, freq=freq, S=S[1])
    Call = librosa.feature.spectral_bandwidth(sr=sr, freq=freq, S=S)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_spectral_bandwidth_multi_variable(s_multi):
    if False:
        return 10
    (S, sr) = s_multi
    freq = np.asarray(np.random.randn(*S.shape))
    C0 = librosa.feature.spectral_bandwidth(sr=sr, freq=freq[0], S=S[0])
    C1 = librosa.feature.spectral_bandwidth(sr=sr, freq=freq[1], S=S[1])
    Call = librosa.feature.spectral_bandwidth(sr=sr, freq=freq, S=S)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_spectral_contrast_multi(s_multi):
    if False:
        while True:
            i = 10
    (S, sr) = s_multi
    freq = None
    C0 = librosa.feature.spectral_contrast(sr=sr, freq=freq, S=S[0])
    C1 = librosa.feature.spectral_contrast(sr=sr, freq=freq, S=S[1])
    Call = librosa.feature.spectral_contrast(sr=sr, freq=freq, S=S)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_spectral_rolloff_multi(s_multi):
    if False:
        for i in range(10):
            print('nop')
    (S, sr) = s_multi
    freq = None
    C0 = librosa.feature.spectral_rolloff(sr=sr, freq=freq, S=S[0])
    C1 = librosa.feature.spectral_rolloff(sr=sr, freq=freq, S=S[1])
    Call = librosa.feature.spectral_rolloff(sr=sr, freq=freq, S=S)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_spectral_rolloff_multi_variable(s_multi):
    if False:
        for i in range(10):
            print('nop')
    (S, sr) = s_multi
    freq = np.asarray(np.random.randn(*S.shape))
    C0 = librosa.feature.spectral_rolloff(sr=sr, freq=freq[0], S=S[0])
    C1 = librosa.feature.spectral_rolloff(sr=sr, freq=freq[1], S=S[1])
    Call = librosa.feature.spectral_rolloff(sr=sr, freq=freq, S=S)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_spectral_flatness_multi(s_multi):
    if False:
        while True:
            i = 10
    (S, sr) = s_multi
    C0 = librosa.feature.spectral_flatness(S=S[0])
    C1 = librosa.feature.spectral_flatness(S=S[1])
    Call = librosa.feature.spectral_flatness(S=S)
    assert np.allclose(C0, Call[0], atol=1e-05)
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_poly_multi_static(s_multi):
    if False:
        return 10
    (mags, sr) = s_multi
    Pall = librosa.feature.poly_features(S=mags, order=5)
    P0 = librosa.feature.poly_features(S=mags[0], order=5)
    P1 = librosa.feature.poly_features(S=mags[1], order=5)
    assert np.allclose(Pall[0], P0)
    assert np.allclose(Pall[1], P1)
    assert not np.allclose(P0, P1)

def test_poly_multi_varying(tfr_multi):
    if False:
        for i in range(10):
            print('nop')
    (times, freqs, mags) = tfr_multi
    Pall = librosa.feature.poly_features(S=mags, freq=freqs, order=5)
    P0 = librosa.feature.poly_features(S=mags[0], freq=freqs[0], order=5)
    P1 = librosa.feature.poly_features(S=mags[1], freq=freqs[1], order=5)
    assert np.allclose(Pall[0], P0)
    assert np.allclose(Pall[1], P1)
    assert not np.allclose(P0, P1)

def test_rms_multi(s_multi):
    if False:
        return 10
    (S, sr) = s_multi
    C0 = librosa.feature.rms(S=S[0])
    C1 = librosa.feature.rms(S=S[1])
    Call = librosa.feature.rms(S=S)
    assert Call.ndim == 3
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_zcr_multi(y_multi):
    if False:
        return 10
    (y, sr) = y_multi
    C0 = librosa.feature.zero_crossing_rate(y=y[0])
    C1 = librosa.feature.zero_crossing_rate(y=y[1])
    Call = librosa.feature.zero_crossing_rate(y=y)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_chroma_stft_multi(s_multi):
    if False:
        print('Hello World!')
    (S, sr) = s_multi
    C0 = librosa.feature.chroma_stft(S=S[0], tuning=0)
    C1 = librosa.feature.chroma_stft(S=S[1], tuning=0)
    Call = librosa.feature.chroma_stft(S=S, tuning=0)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_chroma_cqt_multi(y_multi):
    if False:
        for i in range(10):
            print('nop')
    (y, sr) = y_multi
    C0 = librosa.feature.chroma_cqt(y=y[0], tuning=0)
    C1 = librosa.feature.chroma_cqt(y=y[1], tuning=0)
    Call = librosa.feature.chroma_cqt(y=y, tuning=0)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_chroma_cens_multi(y_multi):
    if False:
        i = 10
        return i + 15
    (y, sr) = y_multi
    C0 = librosa.feature.chroma_cens(y=y[0], tuning=0)
    C1 = librosa.feature.chroma_cens(y=y[1], tuning=0)
    Call = librosa.feature.chroma_cens(y=y, tuning=0)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_tonnetz_multi(y_multi):
    if False:
        while True:
            i = 10
    (y, sr) = y_multi
    C0 = librosa.feature.tonnetz(y=y[0], tuning=0)
    C1 = librosa.feature.tonnetz(y=y[1], tuning=0)
    Call = librosa.feature.tonnetz(y=y, tuning=0)
    assert np.allclose(C0, Call[0], atol=1e-07)
    assert np.allclose(C1, Call[1], atol=1e-07)
    assert not np.allclose(Call[0], Call[1])

def test_mfcc_multi(s_multi):
    if False:
        print('Hello World!')
    (S, sr) = s_multi
    C0 = librosa.feature.mfcc(S=librosa.core.amplitude_to_db(S=S[0], top_db=None))
    C1 = librosa.feature.mfcc(S=librosa.core.amplitude_to_db(S=S[1], top_db=None))
    Call = librosa.feature.mfcc(S=librosa.core.amplitude_to_db(S=S, top_db=None))
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

@pytest.mark.skip(reason='power_to_db leaks information across channels')
def test_mfcc_multi_time(y_multi):
    if False:
        for i in range(10):
            print('nop')
    (y, sr) = y_multi
    C0 = librosa.feature.mfcc(y=y[0])
    C1 = librosa.feature.mfcc(y=y[1])
    Call = librosa.feature.mfcc(y=y)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_melspectrogram_multi(s_multi):
    if False:
        i = 10
        return i + 15
    (S, sr) = s_multi
    C0 = librosa.feature.melspectrogram(S=S[0])
    C1 = librosa.feature.melspectrogram(S=S[1])
    Call = librosa.feature.melspectrogram(S=S)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_melspectrogram_multi_time(y_multi):
    if False:
        return 10
    (y, sr) = y_multi
    C0 = librosa.feature.melspectrogram(y=y[0])
    C1 = librosa.feature.melspectrogram(y=y[1])
    Call = librosa.feature.melspectrogram(y=y)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

@pytest.mark.parametrize('rate', [0.5, 2])
def test_phase_vocoder(y_multi, rate):
    if False:
        while True:
            i = 10
    (y, sr) = y_multi
    D = librosa.stft(y)
    D0 = librosa.phase_vocoder(D[0], rate=rate)
    D1 = librosa.phase_vocoder(D[1], rate=rate)
    D2 = librosa.phase_vocoder(D, rate=rate)
    assert np.allclose(D2[0], D0)
    assert np.allclose(D2[1], D1)
    assert not np.allclose(D2[0], D2[1])

@pytest.mark.parametrize('delay', [1, -1])
def test_stack_memory_multi(delay):
    if False:
        print('Hello World!')
    data = np.random.randn(2, 5, 200)
    C0 = librosa.feature.stack_memory(data[0], delay=delay)
    C1 = librosa.feature.stack_memory(data[1], delay=delay)
    Call = librosa.feature.stack_memory(data, delay=delay)
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert not np.allclose(Call[0], Call[1])

def test_interp_harmonics_multi_static(s_multi):
    if False:
        for i in range(10):
            print('nop')
    (S, sr) = s_multi
    freqs = librosa.fft_frequencies(sr=sr)
    Hall = librosa.interp_harmonics(S, freqs=freqs, harmonics=[0.5, 1, 2])
    H0 = librosa.interp_harmonics(S[0], freqs=freqs, harmonics=[0.5, 1, 2])
    H1 = librosa.interp_harmonics(S[1], freqs=freqs, harmonics=[0.5, 1, 2])
    assert np.allclose(Hall[0], H0)
    assert np.allclose(Hall[1], H1)
    assert not np.allclose(H0, H1)

def test_interp_harmonics_multi_vary(tfr_multi):
    if False:
        while True:
            i = 10
    (times, freqs, mags) = tfr_multi
    Hall = librosa.interp_harmonics(mags, freqs=freqs, harmonics=[0.5, 1, 2], kind='nearest')
    H0 = librosa.interp_harmonics(mags[0], freqs=freqs[0], harmonics=[0.5, 1, 2], kind='nearest')
    H1 = librosa.interp_harmonics(mags[1], freqs=freqs[1], harmonics=[0.5, 1, 2], kind='nearest')
    assert np.allclose(Hall[0], H0)
    assert np.allclose(Hall[1], H1)
    assert not np.allclose(H0, H1)

@pytest.mark.parametrize('filter_peaks', [False, True])
def test_salience_multi_static(s_multi, filter_peaks):
    if False:
        for i in range(10):
            print('nop')
    (S, sr) = s_multi
    freqs = librosa.fft_frequencies(sr=sr)
    sal_all = librosa.salience(S, freqs=freqs, harmonics=[0.5, 1, 2, 3], kind='slinear', filter_peaks=filter_peaks, fill_value=0)
    sal_0 = librosa.salience(S[0], freqs=freqs, harmonics=[0.5, 1, 2, 3], kind='slinear', filter_peaks=filter_peaks, fill_value=0)
    sal_1 = librosa.salience(S[1], freqs=freqs, harmonics=[0.5, 1, 2, 3], kind='slinear', filter_peaks=filter_peaks, fill_value=0)
    assert np.allclose(sal_all[0], sal_0)
    assert np.allclose(sal_all[1], sal_1)
    assert not np.allclose(sal_0, sal_1)

@pytest.mark.parametrize('filter_peaks', [False, True])
def test_salience_multi_dynamic(tfr_multi, filter_peaks):
    if False:
        print('Hello World!')
    (times, freqs, S) = tfr_multi
    sal_all = librosa.salience(S, freqs=freqs, harmonics=[0.5, 1, 2, 3], kind='nearest', filter_peaks=filter_peaks, fill_value=0)
    sal_0 = librosa.salience(S[0], freqs=freqs[0], harmonics=[0.5, 1, 2, 3], kind='nearest', filter_peaks=filter_peaks, fill_value=0)
    sal_1 = librosa.salience(S[1], freqs=freqs[1], harmonics=[0.5, 1, 2, 3], kind='nearest', filter_peaks=filter_peaks, fill_value=0)
    assert np.allclose(sal_all[0], sal_0)
    assert np.allclose(sal_all[1], sal_1)
    assert not np.allclose(sal_0, sal_1)

@pytest.mark.parametrize('center', [False, True])
def test_iirt_multi(y_multi, center):
    if False:
        for i in range(10):
            print('nop')
    (y, sr) = y_multi
    Call = librosa.iirt(y=y, sr=sr, center=center)
    C0 = librosa.iirt(y=y[0], sr=sr, center=center)
    C1 = librosa.iirt(y=y[1], sr=sr, center=center)
    assert np.allclose(Call[0], C0)
    assert np.allclose(Call[1], C1)
    assert not np.allclose(C0, C1)

def test_lpc_multi(y_multi):
    if False:
        i = 10
        return i + 15
    (y, sr) = y_multi
    Lall = librosa.lpc(y, order=6)
    L0 = librosa.lpc(y[0], order=6)
    L1 = librosa.lpc(y[1], order=6)
    assert np.allclose(Lall[0], L0)
    assert np.allclose(Lall[1], L1)
    assert not np.allclose(L0, L1)

def test_yin_multi(y_multi):
    if False:
        i = 10
        return i + 15
    (y, sr) = y_multi
    Pall = librosa.yin(y, fmin=30, fmax=300)
    P0 = librosa.yin(y[0], fmin=30, fmax=300)
    P1 = librosa.yin(y[1], fmin=30, fmax=300)
    assert np.allclose(Pall[0], P0)
    assert np.allclose(Pall[1], P1)
    assert not np.allclose(P0, P1)

@pytest.mark.parametrize('ref', [None, 1.0])
def test_piptrack_multi(s_multi, ref):
    if False:
        return 10
    (S, sr) = s_multi
    (pall, mall) = librosa.piptrack(S=S, sr=sr, ref=ref)
    (p0, m0) = librosa.piptrack(S=S[0], sr=sr, ref=ref)
    (p1, m1) = librosa.piptrack(S=S[1], sr=sr, ref=ref)
    assert np.allclose(pall[0], p0)
    assert np.allclose(pall[1], p1)
    assert np.allclose(mall[0], m0)
    assert np.allclose(mall[1], m1)
    assert not np.allclose(p0, p1)
    assert not np.allclose(m0, m1)

def test_click_multi():
    if False:
        while True:
            i = 10
    click = np.ones((3, 100))
    yout = librosa.clicks(times=[0, 1, 2], sr=1000, click=click)
    assert yout.shape[0] == click.shape[0]
    assert np.allclose(yout[..., :100], click)
    assert np.allclose(yout[..., 1000:1100], click)
    assert np.allclose(yout[..., 2000:2100], click)

def test_nnls_multi(s_multi):
    if False:
        i = 10
        return i + 15
    (S, sr) = s_multi
    S = S[..., :int(S.shape[-1] / 2)]
    mel_basis = librosa.filters.mel(sr=sr, n_fft=2 * S.shape[-2] - 1, n_mels=256)
    M = np.einsum('...ft,mf->...mt', S, mel_basis)
    S_recover = librosa.util.nnls(mel_basis, M)
    M0 = np.einsum('...ft,mf->...mt', S[0], mel_basis)
    S0_recover = librosa.util.nnls(mel_basis, M0)
    M1 = np.einsum('...ft,mf->...mt', S[1], mel_basis)
    S1_recover = librosa.util.nnls(mel_basis, M1)
    assert np.allclose(S_recover[0], S0_recover, atol=1e-05, rtol=1e-05), np.max(np.abs(S_recover[0] - S0_recover))
    assert np.allclose(S_recover[1], S1_recover, atol=1e-05, rtol=1e-05), np.max(np.abs(S_recover[1] - S1_recover))
    assert not np.allclose(S0_recover, S1_recover)

@pytest.mark.parametrize('power', [1, 2])
@pytest.mark.parametrize('n_fft', [1024, 2048])
def test_mel_to_stft_multi(power, n_fft):
    if False:
        return 10
    srand()
    mel_basis = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=128)
    stft_orig = np.random.randn(2, n_fft // 2 + 1, 4) ** power
    mels = np.einsum('...ft,mf->...mt', stft_orig, mel_basis)
    stft = librosa.feature.inverse.mel_to_stft(mels, power=power, n_fft=n_fft)
    mels0 = np.einsum('...ft,mf->...mt', stft_orig[0], mel_basis)
    stft0 = librosa.feature.inverse.mel_to_stft(mels0, power=power, n_fft=n_fft)
    mels1 = np.einsum('...ft,mf->...mt', stft_orig[1], mel_basis)
    stft1 = librosa.feature.inverse.mel_to_stft(mels1, power=power, n_fft=n_fft)
    assert np.allclose(stft[0], stft0)
    assert np.allclose(stft[1], stft1)
    assert not np.allclose(stft0, stft1)

@pytest.mark.parametrize('n_mfcc', [13, 20])
@pytest.mark.parametrize('n_mels', [64, 128])
@pytest.mark.parametrize('dct_type', [2, 3])
def test_mfcc_to_mel_multi(s_multi, n_mfcc, n_mels, dct_type):
    if False:
        for i in range(10):
            print('nop')
    (S, sr) = s_multi
    mfcc0 = librosa.feature.mfcc(S=librosa.core.amplitude_to_db(S=S[0], top_db=None))
    mfcc1 = librosa.feature.mfcc(S=librosa.core.amplitude_to_db(S=S[1], top_db=None))
    mfcc = librosa.feature.mfcc(S=librosa.core.amplitude_to_db(S=S, top_db=None))
    mel_recover = librosa.feature.inverse.mfcc_to_mel(mfcc, n_mels=n_mels, dct_type=dct_type)
    mel_recover0 = librosa.feature.inverse.mfcc_to_mel(mfcc0, n_mels=n_mels, dct_type=dct_type)
    mel_recover1 = librosa.feature.inverse.mfcc_to_mel(mfcc1, n_mels=n_mels, dct_type=dct_type)
    assert np.allclose(mel_recover[0], mel_recover0)
    assert np.allclose(mel_recover[1], mel_recover1)
    assert not np.allclose(mel_recover0, mel_recover1)

def test_trim_multichannel(y_multi):
    if False:
        return 10
    (y, sr) = y_multi
    y = y * np.array([[1e-06, 1000000.0]]).T
    (yt, ival) = librosa.effects.trim(y)
    (yt0, ival0) = librosa.effects.trim(y[0])
    (yt1, ival1) = librosa.effects.trim(y[1])
    assert ival[0] == max(ival0[0], ival1[0])
    assert ival[1] == min(ival0[1], ival1[1])

@pytest.mark.parametrize('res_type', ('scipy', 'polyphase', 'sinc_fastest', 'kaiser_fast', 'soxr_qq'))
def test_resample_multichannel(y_multi, res_type):
    if False:
        while True:
            i = 10
    (y, sr) = y_multi
    y_res = librosa.resample(y=y, orig_sr=sr, target_sr=sr // 2, res_type=res_type)
    y0_res = librosa.resample(y=y[0], orig_sr=sr, target_sr=sr // 2, res_type=res_type)
    y1_res = librosa.resample(y=y[1], orig_sr=sr, target_sr=sr // 2, res_type=res_type)
    assert np.allclose(y_res[0], y0_res)
    assert np.allclose(y_res[1], y1_res)
    assert y_res[0].shape == y0_res.shape

@pytest.mark.parametrize('res_type', ('scipy', 'polyphase', 'sinc_fastest', 'kaiser_fast', 'soxr_qq'))
@pytest.mark.parametrize('x', [np.zeros((2, 2, 2, 22050))])
def test_resample_highdim(x, res_type):
    if False:
        print('Hello World!')
    y = librosa.resample(x, orig_sr=22050, target_sr=11025, res_type=res_type)

@pytest.mark.parametrize('res_type', ('scipy', 'polyphase', 'sinc_fastest', 'kaiser_fast', 'soxr_qq'))
@pytest.mark.parametrize('x, axis', [(np.zeros((2, 2, 2, 22050)), -1), (np.zeros((22050, 2, 3)), 0)])
def test_resample_highdim_axis(x, axis, res_type):
    if False:
        for i in range(10):
            print('nop')
    y = librosa.resample(x, orig_sr=22050, target_sr=11025, axis=axis, res_type=res_type)
    assert y.shape[axis] == 11025
    assert y.ndim == x.ndim

@pytest.mark.parametrize('dynamic', [False, True])
def test_f0_harmonics(y_multi, dynamic):
    if False:
        i = 10
        return i + 15
    (y, sr) = y_multi
    (Df, _, S) = librosa.reassigned_spectrogram(y, sr=sr, fill_nan=True)
    freqs = librosa.fft_frequencies(sr=sr)
    harmonics = np.array([1, 2, 3])
    f0 = 100 + 30 * np.random.random_sample(size=(S.shape[0], S.shape[-1]))
    if dynamic:
        out = librosa.f0_harmonics(S, freqs=Df, f0=f0, harmonics=harmonics)
        out0 = librosa.f0_harmonics(S[0], freqs=Df[0], f0=f0[0], harmonics=harmonics)
        out1 = librosa.f0_harmonics(S[1], freqs=Df[1], f0=f0[1], harmonics=harmonics)
    else:
        out = librosa.f0_harmonics(S, freqs=freqs, f0=f0, harmonics=harmonics)
        out0 = librosa.f0_harmonics(S[0], freqs=freqs, f0=f0[0], harmonics=harmonics)
        out1 = librosa.f0_harmonics(S[1], freqs=freqs, f0=f0[1], harmonics=harmonics)
    assert np.allclose(out[0], out0)
    assert np.allclose(out[1], out1)
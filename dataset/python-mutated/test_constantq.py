"""
CREATED:2015-03-01 by Eric Battenberg <ebattenberg@gmail.com>
unit tests for librosa core.constantq
"""
from __future__ import division
import warnings
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass
from typing import Optional
import librosa
import numpy as np
import scipy.stats
import pytest
from test_core import srand

def __test_cqt_size(y, sr, hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, sparsity, res_type):
    if False:
        i = 10
        return i + 15
    cqt_output = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, tuning=tuning, filter_scale=filter_scale, norm=norm, sparsity=sparsity, res_type=res_type))
    assert cqt_output.shape[0] == n_bins
    return cqt_output

def make_signal(sr, duration, fmin: Optional[str]='C1', fmax: Optional[str]='C8'):
    if False:
        while True:
            i = 10
    'Generates a linear sine sweep'
    if fmin is None:
        fmin_normfreq = 0.01
    else:
        fmin_normfreq = librosa.note_to_hz(fmin) / sr
    if fmax is None:
        fmax_normfreq = 0.5
    else:
        fmax_normfreq = librosa.note_to_hz(fmax) / sr
    return np.sin(np.cumsum(2 * np.pi * np.logspace(np.log10(fmin_normfreq), np.log10(fmax_normfreq), num=int(duration * sr))))

@pytest.fixture(scope='module')
def sr_cqt():
    if False:
        while True:
            i = 10
    return 11025

@pytest.fixture(scope='module')
def y_cqt(sr_cqt):
    if False:
        print('Hello World!')
    return make_signal(sr_cqt, 2.0)

@pytest.fixture(scope='module')
def y_cqt_110(sr_cqt):
    if False:
        return 10
    return librosa.tone(110.0, sr=sr_cqt, duration=0.75)

@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize('hop_length', [-1, 0])
@pytest.mark.parametrize('bpo', [12, 24])
def test_cqt_bad_hop(y_cqt, sr_cqt, hop_length, bpo):
    if False:
        return 10
    librosa.cqt(y=y_cqt, sr=sr_cqt, hop_length=hop_length, n_bins=bpo * 6, bins_per_octave=bpo)

@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize('bpo', [12, 24])
def test_cqt_exceed_passband(y_cqt, sr_cqt, bpo):
    if False:
        print('Hello World!')
    librosa.cqt(y=y_cqt, sr=sr_cqt, fmin=500, n_bins=4 * bpo, bins_per_octave=bpo)

@pytest.mark.parametrize('fmin', [None, librosa.note_to_hz('C2')])
@pytest.mark.parametrize('n_bins', [1, 12, 24, 76])
@pytest.mark.parametrize('bins_per_octave', [12, 24])
@pytest.mark.parametrize('tuning', [None, 0, 0.25])
@pytest.mark.parametrize('filter_scale', [1])
@pytest.mark.parametrize('norm', [1])
@pytest.mark.parametrize('res_type', ['polyphase'])
@pytest.mark.parametrize('hop_length', [512, 2000])
@pytest.mark.parametrize('sparsity', [0.01])
def test_cqt(y_cqt_110, sr_cqt, hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, res_type, sparsity):
    if False:
        return 10
    C = librosa.cqt(y=y_cqt_110, sr=sr_cqt, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, tuning=tuning, filter_scale=filter_scale, norm=norm, sparsity=sparsity, res_type=res_type)
    assert np.iscomplexobj(C)
    assert C.shape[0] == n_bins
    if fmin is None:
        fmin = librosa.note_to_hz('C1')
    if 110 <= fmin * 2 ** (n_bins / bins_per_octave):
        peaks = np.argmax(np.abs(C), axis=0)
        common_peak = np.argmax(np.bincount(peaks))
        peak_frequency = fmin * 2 ** (common_peak / bins_per_octave)
        assert np.isclose(peak_frequency, 110)

@pytest.mark.parametrize('fmin', [librosa.note_to_hz('C1')])
@pytest.mark.parametrize('bins_per_octave', [12])
@pytest.mark.parametrize('n_bins', [88])
def test_cqt_early_downsample(y_cqt_110, sr_cqt, n_bins, fmin, bins_per_octave):
    if False:
        while True:
            i = 10
    C = librosa.cqt(y=y_cqt_110, sr=sr_cqt, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, res_type=None)
    assert np.iscomplexobj(C)
    assert C.shape[0] == n_bins
    if fmin is None:
        fmin = librosa.note_to_hz('C1')
    if 110 <= fmin * 2 ** (n_bins / bins_per_octave):
        peaks = np.argmax(np.abs(C), axis=0)
        common_peak = np.argmax(np.bincount(peaks))
        peak_frequency = fmin * 2 ** (common_peak / bins_per_octave)
        assert np.isclose(peak_frequency, 110)

@pytest.mark.parametrize('hop_length', [256, 512])
def test_cqt_frame_rate(y_cqt_110, sr_cqt, hop_length):
    if False:
        i = 10
        return i + 15
    C = librosa.cqt(y=y_cqt_110, sr=sr_cqt, hop_length=hop_length, res_type='polyphase')
    if hop_length == 256:
        assert C.shape[1] == 33
    elif hop_length == 512:
        assert C.shape[1] == 17
    else:
        assert False

def test_cqt_odd_hop(y_cqt_110, sr_cqt):
    if False:
        while True:
            i = 10
    C = librosa.cqt(y=y_cqt_110, sr=sr_cqt, hop_length=1001, res_type='polyphase')

def test_icqt_odd_hop(y_cqt_110, sr_cqt):
    if False:
        print('Hello World!')
    C = librosa.cqt(y=y_cqt_110, sr=sr_cqt, hop_length=1001, res_type='polyphase')
    yi = librosa.icqt(C, sr=sr_cqt, hop_length=1001, res_type='polyphase', length=len(y_cqt_110))

@pytest.mark.parametrize('fmin', [None, librosa.note_to_hz('C2')])
@pytest.mark.parametrize('n_bins', [1, 12, 24])
@pytest.mark.parametrize('gamma', [None, 0, 2.5])
@pytest.mark.parametrize('bins_per_octave', [12, 24])
@pytest.mark.parametrize('tuning', [0])
@pytest.mark.parametrize('filter_scale', [1])
@pytest.mark.parametrize('norm', [1])
@pytest.mark.parametrize('res_type', ['polyphase'])
@pytest.mark.parametrize('sparsity', [0.01])
@pytest.mark.parametrize('hop_length', [512])
def test_vqt(y_cqt_110, sr_cqt, hop_length, fmin, n_bins, gamma, bins_per_octave, tuning, filter_scale, norm, res_type, sparsity):
    if False:
        while True:
            i = 10
    C = librosa.vqt(y=y_cqt_110, sr=sr_cqt, hop_length=hop_length, fmin=fmin, n_bins=n_bins, gamma=gamma, bins_per_octave=bins_per_octave, tuning=tuning, filter_scale=filter_scale, norm=norm, sparsity=sparsity, res_type=res_type)
    assert np.iscomplexobj(C)
    assert C.shape[0] == n_bins
    if fmin is None:
        fmin = librosa.note_to_hz('C1')
    if 110 <= fmin * 2 ** (n_bins / bins_per_octave):
        peaks = np.argmax(np.abs(C), axis=0)
        common_peak = np.argmax(np.bincount(peaks))
        peak_frequency = fmin * 2 ** (common_peak / bins_per_octave)
        assert np.isclose(peak_frequency, 110)

@pytest.fixture(scope='module')
def y_hybrid():
    if False:
        print('Hello World!')
    return make_signal(11025, 5.0, None)

@pytest.mark.parametrize('sr', [11025])
@pytest.mark.parametrize('hop_length', [512, 2000])
@pytest.mark.parametrize('sparsity', [0.01])
@pytest.mark.parametrize('fmin', [None, librosa.note_to_hz('C2')])
@pytest.mark.parametrize('n_bins', [1, 12, 24, 48, 72, 74, 76])
@pytest.mark.parametrize('bins_per_octave', [12, 24])
@pytest.mark.parametrize('tuning', [None, 0, 0.25])
@pytest.mark.parametrize('resolution', [1])
@pytest.mark.parametrize('norm', [1])
@pytest.mark.parametrize('res_type', ['polyphase'])
def test_hybrid_cqt(y_hybrid, sr, hop_length, fmin, n_bins, bins_per_octave, tuning, resolution, norm, sparsity, res_type):
    if False:
        while True:
            i = 10
    C2 = librosa.hybrid_cqt(y_hybrid, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, tuning=tuning, filter_scale=resolution, norm=norm, sparsity=sparsity, res_type=res_type)
    C1 = np.abs(librosa.cqt(y_hybrid, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, tuning=tuning, filter_scale=resolution, norm=norm, sparsity=sparsity, res_type=res_type))
    assert C1.shape == C2.shape
    idx1 = C1 > 0.0001 * C1.max()
    idx2 = C2 > 0.0001 * C2.max()
    perc = 0.99
    thresh = 0.001
    idx = idx1 | idx2
    assert np.percentile(np.abs(C1[idx] - C2[idx]), perc) < thresh * max(C1.max(), C2.max())

@pytest.mark.parametrize('note_min', [12, 18, 24, 30, 36])
@pytest.mark.parametrize('sr', [22050])
@pytest.mark.parametrize('y', [np.sin(2 * np.pi * librosa.midi_to_hz(60) * np.arange(2 * 22050) / 22050.0)])
def test_cqt_position(y, sr, note_min: int):
    if False:
        while True:
            i = 10
    C = np.abs(librosa.cqt(y, sr=sr, fmin=float(librosa.midi_to_hz(note_min)))) ** 2
    Cbar = np.median(C, axis=1)
    idx = int(np.argmax(Cbar))
    assert idx == 60 - note_min
    Cscale = Cbar / Cbar[idx]
    Cscale[idx] = np.nan
    assert np.nanmax(Cscale) < 0.6, Cscale
    Cscale[idx - 1:idx + 2] = np.nan
    assert np.nanmax(Cscale) < 0.05, Cscale

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cqt_fail_short_early():
    if False:
        i = 10
        return i + 15
    y = np.zeros(16)
    librosa.cqt(y, sr=44100, n_bins=36)

@pytest.fixture(scope='module', params=[11025, 16384, 22050, 32000, 44100])
def sr_impulse(request):
    if False:
        return 10
    return request.param

@pytest.fixture(scope='module', params=range(1, 9))
def hop_impulse(request):
    if False:
        while True:
            i = 10
    return 64 * request.param

@pytest.fixture(scope='module')
def y_impulse(sr_impulse, hop_impulse):
    if False:
        while True:
            i = 10
    x = np.zeros(sr_impulse)
    center = int(len(x) / (2.0 * float(hop_impulse)) * hop_impulse)
    x[center] = 1
    return x

def test_cqt_impulse(y_impulse, sr_impulse, hop_impulse):
    if False:
        i = 10
        return i + 15
    C = np.abs(librosa.cqt(y=y_impulse, sr=sr_impulse, hop_length=hop_impulse))
    response = np.mean(C ** 2, axis=1)
    continuity = np.abs(np.diff(response))
    assert np.max(continuity) < 0.0005, continuity

def test_hybrid_cqt_impulse(y_impulse, sr_impulse, hop_impulse):
    if False:
        for i in range(10):
            print('nop')
    hcqt = librosa.hybrid_cqt(y=y_impulse, sr=sr_impulse, hop_length=hop_impulse, tuning=0)
    response = np.mean(np.abs(hcqt) ** 2, axis=1)
    continuity = np.abs(np.diff(response))
    assert np.max(continuity) < 0.0005, continuity

@pytest.fixture(scope='module')
def sr_white():
    if False:
        print('Hello World!')
    return 22050

@pytest.fixture(scope='module')
def y_white(sr_white):
    if False:
        i = 10
        return i + 15
    srand()
    return np.random.randn(10 * sr_white)

@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('fmin', list(librosa.note_to_hz(['C1', 'C2'])))
@pytest.mark.parametrize('n_bins', [24, 36])
def test_cqt_white_noise(y_white, sr_white, fmin, n_bins, scale):
    if False:
        return 10
    C = np.abs(librosa.cqt(y=y_white, sr=sr_white, fmin=fmin, n_bins=n_bins, scale=scale))
    if not scale:
        lengths = librosa.filters.constant_q_lengths(sr=sr_white, fmin=fmin, n_bins=n_bins)
        C /= np.sqrt(lengths[:, np.newaxis])
    assert np.allclose(np.mean(C, axis=1), 1.0, atol=0.25), np.mean(C, axis=1)
    assert np.allclose(np.std(C, axis=1), 0.5, atol=0.5), np.std(C, axis=1)

@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('fmin', list(librosa.note_to_hz(['C1', 'C2'])))
@pytest.mark.parametrize('n_bins', [72, 84])
def test_hybrid_cqt_white_noise(y_white, sr_white, fmin, n_bins, scale):
    if False:
        i = 10
        return i + 15
    C = librosa.hybrid_cqt(y=y_white, sr=sr_white, fmin=fmin, n_bins=n_bins, scale=scale)
    if not scale:
        lengths = librosa.filters.constant_q_lengths(sr=sr_white, fmin=fmin, n_bins=n_bins)
        C /= np.sqrt(lengths[:, np.newaxis])
    assert np.allclose(np.mean(C, axis=1), 1.0, atol=0.25), np.mean(C, axis=1)
    assert np.allclose(np.std(C, axis=1), 0.5, atol=0.5), np.std(C, axis=1)

@pytest.fixture(scope='module', params=[22050, 44100])
def sr_icqt(request):
    if False:
        i = 10
        return i + 15
    return request.param

@pytest.fixture(scope='module')
def y_icqt(sr_icqt):
    if False:
        print('Hello World!')
    return make_signal(sr_icqt, 1.5, fmin='C2', fmax='C4')

@pytest.mark.parametrize('over_sample', [1, 3])
@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('hop_length', [384, 512])
@pytest.mark.parametrize('length', [None, True])
@pytest.mark.parametrize('res_type', ['soxr_hq', 'polyphase'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_icqt(y_icqt, sr_icqt, scale, hop_length, over_sample, length, res_type, dtype):
    if False:
        i = 10
        return i + 15
    bins_per_octave = over_sample * 12
    n_bins = 7 * bins_per_octave
    C = librosa.cqt(y_icqt, sr=sr_icqt, n_bins=n_bins, bins_per_octave=bins_per_octave, scale=scale, hop_length=hop_length)
    if length:
        _len = len(y_icqt)
    else:
        _len = None
    yinv = librosa.icqt(C, sr=sr_icqt, scale=scale, hop_length=hop_length, bins_per_octave=bins_per_octave, length=_len, res_type=res_type, dtype=dtype)
    assert yinv.dtype == dtype
    if length:
        assert len(y_icqt) == len(yinv)
    else:
        yinv = librosa.util.fix_length(yinv, size=len(y_icqt))
    y_icqt = y_icqt[sr_icqt // 2:-sr_icqt // 2]
    yinv = yinv[sr_icqt // 2:-sr_icqt // 2]
    residual = np.abs(y_icqt - yinv)
    resnorm = np.sqrt(np.mean(residual ** 2))
    assert resnorm <= 0.1, resnorm

@pytest.fixture
def y_chirp():
    if False:
        return 10
    sr = 22050
    y = librosa.chirp(fmin=55, fmax=55 * 2 ** 3, length=sr // 8, sr=sr)
    return y

@pytest.mark.parametrize('hop_length', [512, 1024])
@pytest.mark.parametrize('window', ['hann', 'hamming'])
@pytest.mark.parametrize('use_length', [False, True])
@pytest.mark.parametrize('over_sample', [1, 3])
@pytest.mark.parametrize('res_type', ['polyphase'])
@pytest.mark.parametrize('pad_mode', ['reflect'])
@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('momentum', [0.99])
@pytest.mark.parametrize('random_state', [0])
@pytest.mark.parametrize('fmin', [40.0])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('init', [None])
def test_griffinlim_cqt(y_chirp, hop_length, window, use_length, over_sample, fmin, res_type, pad_mode, scale, momentum, init, random_state, dtype):
    if False:
        print('Hello World!')
    if use_length:
        length = len(y_chirp)
    else:
        length = None
    sr = 22050
    bins_per_octave = 12 * over_sample
    n_bins = 6 * bins_per_octave
    C = librosa.cqt(y_chirp, sr=sr, hop_length=hop_length, window=window, fmin=fmin, bins_per_octave=bins_per_octave, n_bins=n_bins, scale=scale, pad_mode=pad_mode, res_type=res_type)
    Cmag = np.abs(C)
    y_rec = librosa.griffinlim_cqt(Cmag, hop_length=hop_length, window=window, sr=sr, fmin=fmin, bins_per_octave=bins_per_octave, scale=scale, pad_mode=pad_mode, n_iter=2, momentum=momentum, random_state=random_state, length=length, res_type=res_type, init=init, dtype=dtype)
    y_inv = librosa.icqt(Cmag, sr=sr, fmin=fmin, hop_length=hop_length, window=window, bins_per_octave=bins_per_octave, scale=scale, length=length, res_type=res_type)
    if use_length:
        assert len(y_rec) == length
    assert y_rec.dtype == dtype
    assert np.all(np.isfinite(y_rec))

@pytest.mark.parametrize('momentum', [0, 0.95])
def test_griffinlim_cqt_momentum(y_chirp, momentum):
    if False:
        i = 10
        return i + 15
    C = librosa.cqt(y=y_chirp, sr=22050, res_type='polyphase')
    y_rec = librosa.griffinlim_cqt(np.abs(C), sr=22050, n_iter=2, momentum=momentum, res_type='polyphase')
    assert np.all(np.isfinite(y_rec))

@pytest.mark.parametrize('random_state', [None, 0, np.random.RandomState()])
def test_griffinlim_cqt_rng(y_chirp, random_state):
    if False:
        while True:
            i = 10
    C = librosa.cqt(y=y_chirp, sr=22050, res_type='polyphase')
    y_rec = librosa.griffinlim_cqt(np.abs(C), sr=22050, n_iter=2, random_state=random_state, res_type='polyphase')
    assert np.all(np.isfinite(y_rec))

@pytest.mark.parametrize('init', [None, 'random'])
def test_griffinlim_cqt_init(y_chirp, init):
    if False:
        for i in range(10):
            print('nop')
    C = librosa.cqt(y=y_chirp, sr=22050, res_type='polyphase')
    y_rec = librosa.griffinlim_cqt(np.abs(C), sr=22050, n_iter=2, init=init, res_type='polyphase')
    assert np.all(np.isfinite(y_rec))

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_badinit():
    if False:
        i = 10
        return i + 15
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, init='garbage')

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_badrng():
    if False:
        print('Hello World!')
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, random_state='garbage')

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_bad_momentum():
    if False:
        while True:
            i = 10
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, momentum=-1)

def test_griffinlim_cqt_momentum_warn():
    if False:
        print('Hello World!')
    x = np.zeros((33, 3))
    with pytest.warns(UserWarning):
        librosa.griffinlim_cqt(x, momentum=2)

@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_cqt_precision(y_cqt, sr_cqt, dtype):
    if False:
        return 10
    C = librosa.cqt(y=y_cqt, sr=sr_cqt, dtype=dtype)
    assert np.dtype(C.dtype) == np.dtype(dtype)

@pytest.mark.parametrize('n_bins_missing', range(-11, 11))
def test_cqt_partial_octave(y_cqt, sr_cqt, n_bins_missing):
    if False:
        print('Hello World!')
    librosa.cqt(y=y_cqt, sr=sr_cqt, n_bins=72 - n_bins_missing, bins_per_octave=12)

def test_vqt_provided_intervals(y_cqt, sr_cqt):
    if False:
        print('Hello World!')
    V1 = librosa.vqt(y=y_cqt, sr=sr_cqt, bins_per_octave=20, n_bins=60, intervals='equal')
    intervals = 2.0 ** (np.arange(20) / 20.0)
    V2 = librosa.vqt(y=y_cqt, sr=sr_cqt, n_bins=60, intervals=intervals)
    assert np.allclose(V1, V2)
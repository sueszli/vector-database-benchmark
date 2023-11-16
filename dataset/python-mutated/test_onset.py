import pytest
from contextlib import nullcontext as dnr
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass
import warnings
import numpy as np
import librosa
from test_core import srand
__EXAMPLE_FILE = os.path.join('tests', 'data', 'test1_22050.wav')

@pytest.fixture(scope='module')
def ysr():
    if False:
        return 10
    return librosa.load(__EXAMPLE_FILE)

@pytest.mark.parametrize('feature', [None, librosa.feature.melspectrogram, librosa.feature.chroma_stft])
@pytest.mark.parametrize('n_fft', [512, 2048])
@pytest.mark.parametrize('hop_length', [256, 512])
@pytest.mark.parametrize('lag', [1, 2])
@pytest.mark.parametrize('max_size', [1, 2])
@pytest.mark.parametrize('detrend', [False, True])
@pytest.mark.parametrize('center', [False, True])
@pytest.mark.parametrize('aggregate', [None, np.mean, np.max])
def test_onset_strength_audio(ysr, feature, n_fft, hop_length, lag, max_size, detrend, center, aggregate):
    if False:
        i = 10
        return i + 15
    (y, sr) = ysr
    oenv = librosa.onset.onset_strength(y=y, sr=sr, S=None, detrend=detrend, center=center, aggregate=aggregate, feature=feature, n_fft=n_fft, hop_length=hop_length, lag=lag, max_size=max_size)
    assert oenv.ndim == 1
    S = librosa.feature.melspectrogram(y=y, n_fft=n_fft, hop_length=hop_length)
    target_shape = S.shape[-1]
    if not detrend:
        assert np.all(oenv >= 0)
    assert oenv.shape[-1] == target_shape

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_badlag(ysr):
    if False:
        print('Hello World!')
    (y, sr) = ysr
    librosa.onset.onset_strength(y=y, sr=sr, lag=0)

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_badmax(ysr):
    if False:
        while True:
            i = 10
    (y, sr) = ysr
    librosa.onset.onset_strength(y=y, sr=sr, max_size=0)

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_noinput():
    if False:
        print('Hello World!')
    librosa.onset.onset_strength(y=None, S=None)

@pytest.fixture(scope='module')
def melspec_sr(ysr):
    if False:
        i = 10
        return i + 15
    (y, sr) = ysr
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    return (S, sr)

@pytest.mark.parametrize('feature', [None, librosa.feature.melspectrogram, librosa.feature.chroma_stft])
@pytest.mark.parametrize('n_fft', [512, 2048])
@pytest.mark.parametrize('hop_length', [256, 512])
@pytest.mark.parametrize('detrend', [False, True])
@pytest.mark.parametrize('center', [False, True])
@pytest.mark.parametrize('aggregate', [None, np.mean, np.max])
def test_onset_strength_spectrogram(melspec_sr, feature, n_fft, hop_length, detrend, center, aggregate):
    if False:
        i = 10
        return i + 15
    (S, sr) = melspec_sr
    oenv = librosa.onset.onset_strength(y=None, sr=sr, S=S, detrend=detrend, center=center, aggregate=aggregate, feature=feature, n_fft=n_fft, hop_length=hop_length)
    assert oenv.ndim == 1
    target_shape = S.shape[-1]
    if not detrend:
        assert np.all(oenv >= 0)
    assert oenv.shape[-1] == target_shape

@pytest.mark.parametrize('lag', [1, 2, 3])
@pytest.mark.parametrize('aggregate', [np.mean, np.max])
def test_onset_strength_multi_noagg(melspec_sr, lag, aggregate):
    if False:
        i = 10
        return i + 15
    (S, sr) = melspec_sr
    odf_multi = librosa.onset.onset_strength_multi(S=S, lag=lag, max_size=1, aggregate=False)
    odf_mean = librosa.onset.onset_strength_multi(S=S, lag=lag, max_size=1, aggregate=aggregate)
    assert odf_multi.shape == S.shape
    assert np.allclose(odf_mean, aggregate(odf_multi, axis=0))

@pytest.fixture(scope='module')
def channels(melspec_sr):
    if False:
        return 10
    (S, _) = melspec_sr
    return np.linspace(0, S.shape[0], num=5, dtype=int)

@pytest.mark.parametrize('lag', [1, 2, 3])
def test_onset_strength_multi(melspec_sr, lag, channels):
    if False:
        i = 10
        return i + 15
    (S, sr) = melspec_sr
    odf_multi = librosa.onset.onset_strength_multi(S=S, lag=lag, max_size=1, channels=channels)
    assert len(odf_multi) == len(channels) - 1
    for (i, (s, t)) in enumerate(zip(channels, channels[1:])):
        odf_single = librosa.onset.onset_strength(S=S[s:t], lag=lag, max_size=1)
        assert np.allclose(odf_single, odf_multi[i])

@pytest.fixture(scope='module', params=[64, 512, 2048])
def hop(request):
    if False:
        while True:
            i = 10
    return request.param

@pytest.fixture(scope='module', params=[False, True], ids=['audio', 'oenv'])
def oenv(ysr, hop, request):
    if False:
        print('Hello World!')
    if request.param:
        (y, sr) = ysr
        return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    else:
        return None

@pytest.mark.parametrize('bt', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
def test_onset_detect_real(ysr, oenv, hop, bt, normalize):
    if False:
        while True:
            i = 10
    (y, sr) = ysr
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, hop_length=hop, backtrack=bt, normalize=normalize)
    if bt:
        assert np.all(onsets >= 0)
    else:
        assert np.all(onsets > 0)
    assert np.all(onsets < len(y) * sr // hop)
    if oenv is not None:
        assert np.all(onsets < len(oenv))

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_detect_nosignal():
    if False:
        while True:
            i = 10
    librosa.onset.onset_detect(y=None, onset_envelope=None)

@pytest.mark.parametrize('sr', [4000])
@pytest.mark.parametrize('y', [np.zeros(4000), np.ones(4000), -np.ones(4000)])
@pytest.mark.parametrize('hop_length', [64, 512, 2048])
def test_onset_detect_const(y, sr, hop_length):
    if False:
        i = 10
        return i + 15
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=None, hop_length=hop_length)
    assert len(onsets) == 0 or (y[0] != 0 and len(onsets) == 1)

@pytest.mark.parametrize('units, ctx', [('frames', dnr()), ('time', dnr()), ('samples', dnr()), ('bad units', pytest.raises(librosa.ParameterError))])
@pytest.mark.parametrize('hop_length', [512, 1024])
def test_onset_units(ysr, hop_length, units, ctx):
    if False:
        while True:
            i = 10
    (y, sr) = ysr
    with ctx:
        b1 = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        b2 = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units=units)
        t1 = librosa.frames_to_time(b1, sr=sr, hop_length=hop_length)
        if units == 'time':
            t2 = b2
        elif units == 'samples':
            t2 = librosa.samples_to_time(b2, sr=sr)
        elif units == 'frames':
            t2 = librosa.frames_to_time(b2, sr=sr, hop_length=hop_length)
        else:
            assert False
        assert np.allclose(t1, t2)

@pytest.fixture(scope='module', params=[False, True], ids=['oenv', 'rms'])
def energy(ysr, hop, request):
    if False:
        for i in range(10):
            print('nop')
    (y, sr) = ysr
    if request.param:
        return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    else:
        return librosa.feature.rms(y=y, hop_length=hop)

def test_onset_backtrack(ysr, oenv, hop, energy):
    if False:
        for i in range(10):
            print('nop')
    (y, sr) = ysr
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, hop_length=hop, backtrack=False)
    onsets_bt = librosa.onset.onset_backtrack(onsets, energy)
    assert np.all(onsets_bt >= 0)
    assert np.all(onsets_bt <= onsets)
    assert np.all(energy[onsets_bt] <= energy[np.maximum(0, onsets_bt - 1)])

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_noagg():
    if False:
        while True:
            i = 10
    S = np.zeros((3, 3))
    librosa.onset.onset_strength(S=S, aggregate=False)

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_badref():
    if False:
        return 10
    S = np.zeros((3, 3))
    librosa.onset.onset_strength(S=S, ref=S[:, :2])

def test_onset_strength_multi_ref():
    if False:
        for i in range(10):
            print('nop')
    srand()
    S = 1 + np.abs(np.random.randn(1025, 10))
    null_ref = np.zeros_like(S)
    onsets = librosa.onset.onset_strength_multi(S=S, ref=null_ref, aggregate=False, center=False)
    assert np.allclose(onsets[:, 1:], S[:, 1:])

def test_onset_detect_inplace_normalize():
    if False:
        return 10
    oenv_in = np.ones(50)
    oenv_in[10] = 2
    oenv_orig = oenv_in.copy()
    librosa.onset.onset_detect(onset_envelope=oenv_in, normalize=True)
    assert np.allclose(oenv_in, oenv_orig) and oenv_in is not oenv_orig
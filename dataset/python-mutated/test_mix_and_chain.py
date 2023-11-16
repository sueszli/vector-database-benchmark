import pytest
import numpy as np
from pedalboard import Pedalboard, Compressor, Delay, Distortion, Gain, Mix, Chain, PitchShift, Reverb
from pedalboard_native._internal import AddLatency
NUM_SECONDS = 4

def test_chain_access_like_list():
    if False:
        for i in range(10):
            print('nop')
    a = Gain(6)
    b = Gain(-6)
    c = Gain(1)
    d = Chain([b, c])
    e = Gain(-1)
    pb = Pedalboard([a, d, e])
    assert pb[0] is a
    assert pb[-3] is a
    assert pb[1] is d
    assert pb[-2] is d
    assert pb[1][0] is b
    assert pb[1][-2] is b
    assert pb[1][1] is c
    assert pb[1][-1] is c
    assert pb[2] is e
    assert pb[-1] is e

def test_nested_chain():
    if False:
        return 10
    sr = 44100
    _input = np.random.rand(int(NUM_SECONDS * sr), 2).astype(np.float32)
    pb = Pedalboard([Gain(6), Chain([Gain(-6), Gain(1)]), Gain(-1)])
    output = pb(_input, sr)
    assert isinstance(pb[0], Gain)
    assert pb[0].gain_db == 6
    assert isinstance(pb[1], Chain)
    assert pb[1][0].gain_db == -6
    assert pb[1][1].gain_db == 1
    assert isinstance(pb[2], Gain)
    assert pb[2].gain_db == -1
    np.testing.assert_allclose(_input, output, rtol=0.01)

def test_nested_list_raises_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError):
        Pedalboard([Gain(6), [Gain(-6), Gain(1)], Gain(-1)])

def test_nested_mix():
    if False:
        i = 10
        return i + 15
    sr = 44100
    _input = np.random.rand(int(NUM_SECONDS * sr), 2).astype(np.float32)
    pb = Pedalboard([Gain(6), Mix([Gain(-6), Gain(-6)]), Gain(-6)])
    output = pb(_input, sr)
    assert isinstance(pb[0], Gain)
    assert pb[0].gain_db == 6
    assert isinstance(pb[1], Mix)
    assert set([plugin.gain_db for plugin in pb[1]]) == set([-6, -6])
    assert isinstance(pb[2], Gain)
    assert pb[2].gain_db == -6
    np.testing.assert_allclose(_input, output, rtol=0.01)

def test_deep_nesting():
    if False:
        for i in range(10):
            print('nop')
    sr = 44100
    _input = np.random.rand(int(NUM_SECONDS * sr), 2).astype(np.float32)
    pb = Pedalboard([Mix([Chain([Gain(1) for _ in range(6)]), Chain([Gain(1) for _ in range(6)])]), Mix([Chain([Gain(-1) for _ in range(24)]), Chain([Gain(-1) for _ in range(24)])])])
    output = pb(_input, sr)
    np.testing.assert_allclose(_input * 0.5, output, rtol=0.01)

def test_nesting_pedalboards():
    if False:
        print('Hello World!')
    sr = 44100
    _input = np.random.rand(int(NUM_SECONDS * sr), 2).astype(np.float32)
    pb = Pedalboard([Mix([Pedalboard([Gain(1) for _ in range(6)]), Pedalboard([Gain(1) for _ in range(6)])]), Mix([Pedalboard([Gain(-1) for _ in range(24)]), Pedalboard([Gain(-1) for _ in range(24)])])])
    output = pb(_input, sr)
    np.testing.assert_allclose(_input * 0.5, output, rtol=0.01)

@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('buffer_size', [128, 8192, 22050, 65536])
@pytest.mark.parametrize('latency_a_seconds', [0.25, 1, NUM_SECONDS / 2])
@pytest.mark.parametrize('latency_b_seconds', [0.25, 1, NUM_SECONDS / 2])
def test_mix_latency_compensation(sample_rate, buffer_size, latency_a_seconds, latency_b_seconds):
    if False:
        i = 10
        return i + 15
    noise = np.random.rand(int(NUM_SECONDS * sample_rate))
    pb = Pedalboard([Mix([AddLatency(int(latency_a_seconds * sample_rate)), AddLatency(int(latency_b_seconds * sample_rate))])])
    output = pb(noise, sample_rate, buffer_size=buffer_size)
    np.testing.assert_allclose(output, noise * 2, rtol=0.01)

@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('buffer_size', [128, 8192, 65536])
@pytest.mark.parametrize('latency_a_seconds', [0.25, 1, 2, 10])
@pytest.mark.parametrize('latency_b_seconds', [0.25, 1, 2, 10])
def test_chain_latency_compensation(sample_rate, buffer_size, latency_a_seconds, latency_b_seconds):
    if False:
        return 10
    noise = np.random.rand(int(NUM_SECONDS * sample_rate))
    pb = Pedalboard([Chain([AddLatency(int(latency_a_seconds * sample_rate)), AddLatency(int(latency_b_seconds * sample_rate))]), Chain([AddLatency(int(latency_a_seconds * sample_rate)), AddLatency(int(latency_b_seconds * sample_rate))])])
    output = pb(noise, sample_rate, buffer_size=buffer_size)
    np.testing.assert_allclose(output, noise, rtol=0.01)

@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('buffer_size', [128, 8192, 65536])
def test_readme_example_does_not_crash(sample_rate, buffer_size):
    if False:
        while True:
            i = 10
    noise = np.random.rand(int(NUM_SECONDS * sample_rate))
    passthrough = Gain(gain_db=0)
    delay_and_pitch_shift = Pedalboard([Delay(delay_seconds=0.25, mix=1.0), PitchShift(semitones=7), Gain(gain_db=-3)])
    delay_longer_and_more_pitch_shift = Pedalboard([Delay(delay_seconds=0.25, mix=1.0), PitchShift(semitones=12), Gain(gain_db=-6)])
    original_plus_delayed_harmonies = Pedalboard(Mix([passthrough, delay_and_pitch_shift, delay_longer_and_more_pitch_shift]))
    original_plus_delayed_harmonies(noise, sample_rate=sample_rate, buffer_size=buffer_size)
    original_plus_delayed_harmonies = Pedalboard([Compressor(), Mix([Pedalboard([passthrough, Distortion(drive_db=36)]), Pedalboard([delay_and_pitch_shift, Reverb(room_size=1)]), delay_longer_and_more_pitch_shift]), Reverb()])
    original_plus_delayed_harmonies(noise, sample_rate=sample_rate, buffer_size=buffer_size)

@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('buffer_size', [128, 8192, 65536])
def test_pedalboard_is_a_plugin(sample_rate, buffer_size):
    if False:
        while True:
            i = 10
    noise = np.random.rand(int(NUM_SECONDS * sample_rate))
    passthrough = Gain(gain_db=0)
    delay_and_pitch_shift = Chain([Delay(delay_seconds=0.25, mix=1.0), PitchShift(semitones=7), Gain(gain_db=-3)])
    delay_longer_and_more_pitch_shift = Chain([Delay(delay_seconds=0.5, mix=1.0), PitchShift(semitones=12), Gain(gain_db=-6)])
    original_plus_delayed_harmonies = Pedalboard([Mix([passthrough, delay_and_pitch_shift, delay_longer_and_more_pitch_shift])])
    original_plus_delayed_harmonies(noise, sample_rate=sample_rate, buffer_size=buffer_size)
    original_plus_delayed_harmonies = Pedalboard([Compressor(), Mix([Chain([passthrough, Distortion(drive_db=36)]), Chain([delay_and_pitch_shift, Reverb(room_size=1)]), delay_longer_and_more_pitch_shift]), Reverb()])
    original_plus_delayed_harmonies(noise, sample_rate=sample_rate, buffer_size=buffer_size)

@pytest.mark.parametrize('cls', [Mix, Chain, Pedalboard])
def test_empty_list_is_valid_constructor_arg(cls):
    if False:
        print('Hello World!')
    assert len(cls([])) == 0

@pytest.mark.parametrize('cls', [Mix, Chain, Pedalboard])
def test_no_arg_constructor(cls):
    if False:
        while True:
            i = 10
    assert len(cls()) == 0

@pytest.mark.parametrize('cls', [Mix, Chain, Pedalboard])
def test_none_as_argument(cls):
    if False:
        while True:
            i = 10
    container = cls([None])
    assert len(container) == 1
    assert container[0] is None
    sample_rate = 44100
    noise = np.random.rand(int(NUM_SECONDS * sample_rate))
    output = container(noise, sample_rate)
    np.testing.assert_allclose(noise, output)
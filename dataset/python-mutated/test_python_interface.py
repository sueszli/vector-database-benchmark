import pytest
import numpy as np
from pedalboard import Pedalboard, Gain

@pytest.mark.parametrize('shape', [(44100,), (44100, 1), (44100, 2), (1, 4), (2, 4)])
def test_no_transforms(shape, sr=44100):
    if False:
        return 10
    _input = np.random.rand(*shape).astype(np.float32)
    output = Pedalboard([]).process(_input, sr)
    assert _input.shape == output.shape
    assert np.allclose(_input, output, rtol=0.0001)

def test_fail_on_invalid_plugin():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        Pedalboard(['I want a reverb please'])

def test_fail_on_invalid_sample_rate():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):
        Pedalboard([]).process([], 'fourty four one hundred')

def test_fail_on_invalid_buffer_size():
    if False:
        return 10
    with pytest.raises(TypeError):
        Pedalboard([]).process([], 44100, 'very big buffer please')

def test_repr():
    if False:
        i = 10
        return i + 15
    gain = Gain(-6)
    value = repr(Pedalboard([gain]))
    assert 'Pedalboard' in value
    assert ' 1 ' in value
    assert repr(gain) in value
    gain2 = Gain(-6)
    value = repr(Pedalboard([gain, gain2]))
    assert 'Pedalboard' in value
    assert ' 2 ' in value
    assert repr(gain) in value
    assert repr(gain2) in value

def test_is_list_like():
    if False:
        return 10
    gain = Gain(-6)
    assert len(Pedalboard([gain])) == 1
    assert len(Pedalboard([gain, Gain(-6)])) == 2
    pb = Pedalboard([gain])
    assert len(pb) == 1
    pb.append(Gain())
    assert len(pb) == 2
    with pytest.raises(TypeError):
        pb.append('not a plugin')
    del pb[1]
    assert len(pb) == 1
    assert pb[0] is gain
    pb[0] = gain
    assert pb[0] is gain
    with pytest.raises(TypeError):
        pb[0] = 'not a plugin'
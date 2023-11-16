import time
import pytest
import platform
import pedalboard
INPUT_DEVICE_NAMES_TO_SKIP = {'iPhone Microphone'}
INPUT_DEVICE_NAMES = [n for n in pedalboard.io.AudioStream.input_device_names if not any((substr in n for substr in INPUT_DEVICE_NAMES_TO_SKIP))]
ACCEPTABLE_ERRORS_ON_CI = {'No driver'}

@pytest.mark.parametrize('input_device_name', INPUT_DEVICE_NAMES)
@pytest.mark.parametrize('output_device_name', pedalboard.io.AudioStream.output_device_names)
@pytest.mark.skipif(platform.system() == 'Linux', reason='AudioStream not supported on Linux yet.')
def test_create_stream(input_device_name: str, output_device_name: str):
    if False:
        for i in range(10):
            print('nop')
    try:
        stream = pedalboard.io.AudioStream(input_device_name, output_device_name, allow_feedback=True)
    except Exception as e:
        if any((substr in str(e) for substr in ACCEPTABLE_ERRORS_ON_CI)):
            return
        raise
    assert stream is not None
    assert input_device_name in repr(stream)
    assert output_device_name in repr(stream)
    assert not stream.running
    assert isinstance(stream.plugins, pedalboard.Chain)
    with stream:
        assert stream.running
        stream.plugins.append(pedalboard.Gain(gain_db=-120))
        for _ in range(0, 100):
            time.sleep(0.01)
            stream.plugins.append(pedalboard.Gain(gain_db=-120))
        for i in reversed(range(len(stream.plugins))):
            time.sleep(0.01)
            del stream.plugins[i]
        assert stream.running
    assert not stream.running

@pytest.mark.skipif(platform.system() != 'Linux', reason='Test platform is not Linux.')
def test_create_stream_fails_on_linux():
    if False:
        i = 10
        return i + 15
    with pytest.raises(RuntimeError):
        pedalboard.io.AudioStream('input', 'output')
import sys
import warnings
from typing import List
import paddle
from . import backend, wave_backend

def _check_version(version: str) -> bool:
    if False:
        while True:
            i = 10
    ver_arr = version.split('.')
    v0 = int(ver_arr[0])
    v1 = int(ver_arr[1])
    v2 = int(ver_arr[2])
    if v0 < 1:
        return False
    if v0 == 1 and v1 == 0 and (v2 <= 1):
        return False
    return True

def list_available_backends() -> List[str]:
    if False:
        print('Hello World!')
    'List available backends, the backends in paddleaudio and the default backend.\n\n    Returns:\n        List[str]: The list of available backends.\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> sample_rate = 16000\n            >>> wav_duration = 0.5\n            >>> num_channels = 1\n            >>> num_frames = sample_rate * wav_duration\n            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1\n            >>> waveform = wav_data.tile([num_channels, 1])\n            >>> wav_path = "./test.wav"\n\n            >>> current_backend = paddle.audio.backends.get_current_backend()\n            >>> print(current_backend)\n            wave_backend\n\n            >>> backends = paddle.audio.backends.list_available_backends()\n            >>> # default backends is [\'wave_backend\']\n            >>> # backends is [\'wave_backend\', \'soundfile\'], if have installed paddleaudio >= 1.0.2\n            >>> if \'soundfile\' in backends:\n            ...     paddle.audio.backends.set_backend(\'soundfile\')\n            ...\n            >>> paddle.audio.save(wav_path, waveform, sample_rate)\n\n    '
    backends = []
    try:
        import paddleaudio
    except ImportError:
        package = 'paddleaudio'
        warn_msg = 'Failed importing {}. \nonly wave_banckend(only can deal with PCM16 WAV) supportted.\nif want soundfile_backend(more audio type suppported),\nplease manually installed (usually with `pip install {} >= 1.0.2`). '.format(package, package)
        warnings.warn(warn_msg)
    if 'paddleaudio' in sys.modules:
        version = paddleaudio.__version__
        if not _check_version(version):
            err_msg = f'the version of paddleaudio installed is {version},\nplease ensure the paddleaudio >= 1.0.2.'
            raise ImportError(err_msg)
        backends = paddleaudio.backends.list_audio_backends()
    backends.append('wave_backend')
    return backends

def get_current_backend() -> str:
    if False:
        i = 10
        return i + 15
    'Get the name of the current audio backend\n\n    Returns:\n        str: The name of the current backend,\n        the wave_backend or backend imported from paddleaudio\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> sample_rate = 16000\n            >>> wav_duration = 0.5\n            >>> num_channels = 1\n            >>> num_frames = sample_rate * wav_duration\n            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1\n            >>> waveform = wav_data.tile([num_channels, 1])\n            >>> wav_path = "./test.wav"\n\n            >>> current_backend = paddle.audio.backends.get_current_backend()\n            >>> print(current_backend)\n            wave_backend\n\n            >>> backends = paddle.audio.backends.list_available_backends()\n            >>> # default backends is [\'wave_backend\']\n            >>> # backends is [\'wave_backend\', \'soundfile\'], if have installed paddleaudio >= 1.0.2\n\n            >>> if \'soundfile\' in backends:\n            ...     paddle.audio.backends.set_backend(\'soundfile\')\n            ...\n            >>> paddle.audio.save(wav_path, waveform, sample_rate)\n\n    '
    current_backend = None
    if 'paddleaudio' in sys.modules:
        import paddleaudio
        current_backend = paddleaudio.backends.get_audio_backend()
        if paddle.audio.load == paddleaudio.load:
            return current_backend
    return 'wave_backend'

def set_backend(backend_name: str):
    if False:
        print('Hello World!')
    'Set the backend by one of the list_audio_backend return.\n\n    Args:\n        backend (str): one of the list_audio_backend. "wave_backend" is the default. "soundfile" imported from paddleaudio.\n\n    Returns:\n        None\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> sample_rate = 16000\n            >>> wav_duration = 0.5\n            >>> num_channels = 1\n            >>> num_frames = sample_rate * wav_duration\n            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1\n            >>> waveform = wav_data.tile([num_channels, 1])\n            >>> wav_path = "./test.wav"\n\n            >>> current_backend = paddle.audio.backends.get_current_backend()\n            >>> print(current_backend)\n            wave_backend\n\n            >>> backends = paddle.audio.backends.list_available_backends()\n            >>> # default backends is [\'wave_backend\']\n            >>> # backends is [\'wave_backend\', \'soundfile\'], if have installed paddleaudio >= 1.0.2\n\n            >>> if \'soundfile\' in backends:\n            ...     paddle.audio.backends.set_backend(\'soundfile\')\n            ...\n            >>> paddle.audio.save(wav_path, waveform, sample_rate)\n\n    '
    if backend_name not in list_available_backends():
        raise NotImplementedError()
    if backend_name == 'wave_backend':
        module = wave_backend
    else:
        import paddleaudio
        paddleaudio.backends.set_audio_backend(backend_name)
        module = paddleaudio
    for func in ['save', 'load', 'info']:
        setattr(backend, func, getattr(module, func))
        setattr(paddle.audio, func, getattr(module, func))

def _init_set_audio_backend():
    if False:
        for i in range(10):
            print('nop')
    for func in ['save', 'load', 'info']:
        setattr(backend, func, getattr(wave_backend, func))
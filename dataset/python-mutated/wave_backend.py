import wave
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import paddle
from .backend import AudioInfo

def _error_message():
    if False:
        while True:
            i = 10
    package = 'paddleaudio'
    warn_msg = f"only PCM16 WAV supportted. \nif want support more other audio types, please manually installed (usually with `pip install {package}`). \n and use paddle.audio.backends.set_backend('soundfile') to set audio backend"
    return warn_msg

def info(filepath: str) -> AudioInfo:
    if False:
        i = 10
        return i + 15
    'Get signal information of input audio file.\n\n    Args:\n       filepath: audio path or file object.\n\n    Returns:\n        AudioInfo: info of the given audio.\n\n    Example:\n        .. code-block:: python\n\n            >>> import os\n            >>> import paddle\n\n            >>> sample_rate = 16000\n            >>> wav_duration = 0.5\n            >>> num_channels = 1\n            >>> num_frames = sample_rate * wav_duration\n            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1\n            >>> waveform = wav_data.tile([num_channels, 1])\n            >>> base_dir = os.getcwd()\n            >>> filepath = os.path.join(base_dir, "test.wav")\n\n            >>> paddle.audio.save(filepath, waveform, sample_rate)\n            >>> wav_info = paddle.audio.info(filepath)\n    '
    if hasattr(filepath, 'read'):
        file_obj = filepath
    else:
        file_obj = open(filepath, 'rb')
    try:
        file_ = wave.open(file_obj)
    except wave.Error:
        file_obj.seek(0)
        file_obj.close()
        err_msg = _error_message()
        raise NotImplementedError(err_msg)
    channels = file_.getnchannels()
    sample_rate = file_.getframerate()
    sample_frames = file_.getnframes()
    bits_per_sample = file_.getsampwidth() * 8
    encoding = 'PCM_S'
    file_obj.close()
    return AudioInfo(sample_rate, sample_frames, channels, bits_per_sample, encoding)

def load(filepath: Union[str, Path], frame_offset: int=0, num_frames: int=-1, normalize: bool=True, channels_first: bool=True) -> Tuple[paddle.Tensor, int]:
    if False:
        return 10
    'Load audio data from file. load the audio content start form frame_offset, and get num_frames.\n\n    Args:\n        frame_offset: from 0 to total frames,\n        num_frames: from -1 (means total frames) or number frames which want to read,\n        normalize:\n            if True: return audio which norm to (-1, 1), dtype=float32\n            if False: return audio with raw data, dtype=int16\n\n        channels_first:\n            if True: return audio with shape (channels, time)\n\n    Return:\n        Tuple[paddle.Tensor, int]: (audio_content, sample rate)\n\n    Examples:\n        .. code-block:: python\n\n            >>> import os\n            >>> import paddle\n\n            >>> sample_rate = 16000\n            >>> wav_duration = 0.5\n            >>> num_channels = 1\n            >>> num_frames = sample_rate * wav_duration\n            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1\n            >>> waveform = wav_data.tile([num_channels, 1])\n            >>> base_dir = os.getcwd()\n            >>> filepath = os.path.join(base_dir, "test.wav")\n\n            >>> paddle.audio.save(filepath, waveform, sample_rate)\n            >>> wav_data_read, sr = paddle.audio.load(filepath)\n    '
    if hasattr(filepath, 'read'):
        file_obj = filepath
    else:
        file_obj = open(filepath, 'rb')
    try:
        file_ = wave.open(file_obj)
    except wave.Error:
        file_obj.seek(0)
        file_obj.close()
        err_msg = _error_message()
        raise NotImplementedError(err_msg)
    channels = file_.getnchannels()
    sample_rate = file_.getframerate()
    frames = file_.getnframes()
    audio_content = file_.readframes(frames)
    file_obj.close()
    audio_as_np16 = np.frombuffer(audio_content, dtype=np.int16)
    audio_as_np32 = audio_as_np16.astype(np.float32)
    if normalize:
        audio_norm = audio_as_np32 / 2 ** 15
    else:
        audio_norm = audio_as_np32
    waveform = np.reshape(audio_norm, (frames, channels))
    if num_frames != -1:
        waveform = waveform[frame_offset:frame_offset + num_frames, :]
    waveform = paddle.to_tensor(waveform)
    if channels_first:
        waveform = paddle.transpose(waveform, perm=[1, 0])
    return (waveform, sample_rate)

def save(filepath: str, src: paddle.Tensor, sample_rate: int, channels_first: bool=True, encoding: Optional[str]=None, bits_per_sample: Optional[int]=16):
    if False:
        i = 10
        return i + 15
    '\n    Save audio tensor to file.\n\n    Args:\n        filepath: saved path\n        src: the audio tensor\n        sample_rate: the number of samples of audio per second.\n        channels_first: src channel information\n            if True, means input tensor is (channels, time)\n            if False, means input tensor is (time, channels)\n        encoding: audio encoding format, wave_backend only support PCM16 now.\n        bits_per_sample: bits per sample, wave_backend only support 16 bits now.\n\n    Returns:\n        None\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> sample_rate = 16000\n            >>> wav_duration = 0.5\n            >>> num_channels = 1\n            >>> num_frames = sample_rate * wav_duration\n            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1\n            >>> waveform = wav_data.tile([num_channels, 1])\n            >>> filepath = "./test.wav"\n\n            >>> paddle.audio.save(filepath, waveform, sample_rate)\n    '
    assert src.ndim == 2, 'Expected 2D tensor'
    audio_numpy = src.numpy()
    if channels_first:
        audio_numpy = np.transpose(audio_numpy)
    channels = audio_numpy.shape[1]
    if bits_per_sample not in (None, 16):
        raise ValueError('Invalid bits_per_sample, only support 16 bit')
    sample_width = int(bits_per_sample / 8)
    if src.dtype == paddle.float32:
        audio_numpy = (audio_numpy * 2 ** 15).astype('<h')
    with wave.open(filepath, 'w') as f:
        f.setnchannels(channels)
        f.setsampwidth(sample_width)
        f.setframerate(sample_rate)
        f.writeframes(audio_numpy.tobytes())
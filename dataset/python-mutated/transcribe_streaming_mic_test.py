import os
import re
import threading
import time
from unittest import mock
import pytest
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

class MockPyAudio:

    def __init__(self: object, audio_filename: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.audio_filename = audio_filename

    def __call__(self: object, *args: object) -> object:
        if False:
            i = 10
            return i + 15
        return self

    def open(self: object, stream_callback: object, rate: int, *args: object, **kwargs: object) -> object:
        if False:
            return 10
        self.rate = rate
        self.closed = threading.Event()
        self.stream_thread = threading.Thread(target=self.stream_audio, args=(self.audio_filename, stream_callback, self.closed))
        self.stream_thread.start()
        return self

    def close(self: object) -> None:
        if False:
            return 10
        self.closed.set()

    def stop_stream(self: object) -> None:
        if False:
            return 10
        pass

    def terminate(self: object) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def stream_audio(self: object, audio_filename: str, callback: object, closed: object, num_frames: int=512) -> None:
        if False:
            while True:
                i = 10
        with open(audio_filename, 'rb') as audio_file:
            while not closed.is_set():
                time.sleep(num_frames / float(self.rate))
                num_bytes = 2 * num_frames
                chunk = audio_file.read(num_bytes) or b'\x00' * num_bytes
                callback(chunk, None, None, None)

@mock.patch.dict('sys.modules', pyaudio=mock.MagicMock(PyAudio=MockPyAudio(os.path.join(RESOURCES, 'quit.raw'))))
def test_main(capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    import transcribe_streaming_mic
    transcribe_streaming_mic.main()
    (out, err) = capsys.readouterr()
    assert re.search('quit', out, re.DOTALL | re.I)
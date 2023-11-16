from threading import Event
from streamlink.buffers import Buffer
from streamlink.stream.stream import StreamIO

class FilteredStream(StreamIO):
    """StreamIO mixin for being able to pause read calls while filtering content"""
    buffer: Buffer

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._event_filter = Event()
        self._event_filter.set()
        super().__init__(*args, **kwargs)

    def read(self, *args, **kwargs) -> bytes:
        if False:
            i = 10
            return i + 15
        read = super().read
        while True:
            try:
                return read(*args, **kwargs)
            except OSError:
                self._event_filter.wait()
                if self.buffer.closed:
                    return b''
                if self.buffer.length > 0:
                    continue
                raise

    def close(self) -> None:
        if False:
            print('Hello World!')
        super().close()
        self._event_filter.set()

    def is_paused(self) -> bool:
        if False:
            return 10
        return not self._event_filter.is_set()

    def pause(self) -> None:
        if False:
            return 10
        self._event_filter.clear()

    def resume(self) -> None:
        if False:
            while True:
                i = 10
        self._event_filter.set()

    def filter_wait(self, timeout=None):
        if False:
            print('Hello World!')
        return self._event_filter.wait(timeout)
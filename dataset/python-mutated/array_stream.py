from .stream import Stream
from .lv_types import StreamItem
import uuid

class ArrayStream(Stream):

    def __init__(self, array, stream_name: str=None, console_debug: bool=False):
        if False:
            while True:
                i = 10
        super(ArrayStream, self).__init__(stream_name=stream_name, console_debug=console_debug)
        self.stream_name = stream_name
        self.array = array

    def load(self, from_stream: 'Stream'=None):
        if False:
            i = 10
            return i + 15
        if self.array is not None:
            self.write(self.array)
        super(ArrayStream, self).load()
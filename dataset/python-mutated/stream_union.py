from .stream import Stream
from typing import Iterator

class StreamUnion(Stream):

    def __init__(self, child_streams: Iterator[Stream], for_write: bool, stream_name: str=None, console_debug: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(StreamUnion, self).__init__(stream_name=stream_name, console_debug=console_debug)
        self.child_streams = child_streams
        if for_write:
            for child_stream in child_streams:
                child_stream.subscribe(self)
        else:
            for child_stream in child_streams:
                self.subscribe(child_stream)
import sys
from typing import TextIO

class StreamWrapper:
    """
    Used by logger to wrap stderr & stdout streams. Handles UnicodeDecodeError if console encoding is not utf-8.
    """

    def __init__(self, stream: TextIO):
        if False:
            print('Hello World!')
        self.stream = stream

    def flush(self):
        if False:
            print('Hello World!')
        try:
            self.stream.flush()
        except:
            pass

    def write(self, s: str):
        if False:
            return 10
        try:
            self.stream.write(s)
        except UnicodeEncodeError:
            encoding = self.stream.encoding
            s2 = s.encode(encoding, errors='backslashreplace').decode(encoding)
            self.stream.write(s2)

    def close(self):
        if False:
            return 10
        try:
            self.stream.close()
        except:
            pass
stdout_wrapper = StreamWrapper(sys.stdout)
stderr_wrapper = StreamWrapper(sys.stderr)
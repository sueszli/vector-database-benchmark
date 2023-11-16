class ListO(object):
    """A StringO write to list."""

    def __init__(self, buffer=None):
        if False:
            i = 10
            return i + 15
        self._buffer = buffer
        if self._buffer is None:
            self._buffer = []

    def isatty(self):
        if False:
            i = 10
            return i + 15
        return False

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def seek(self, n, mode=0):
        if False:
            i = 10
            return i + 15
        pass

    def readline(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def reset(self):
        if False:
            return 10
        pass

    def write(self, x):
        if False:
            return 10
        self._buffer.append(x)

    def writelines(self, x):
        if False:
            while True:
                i = 10
        self._buffer.extend(x)
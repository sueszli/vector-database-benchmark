from antlr4.Token import Token

class InputStream(object):
    __slots__ = ('name', 'strdata', '_index', 'data', '_size')

    def __init__(self, data: str):
        if False:
            print('Hello World!')
        self.name = '<empty>'
        self.strdata = data
        self._loadString()

    def _loadString(self):
        if False:
            for i in range(10):
                print('nop')
        self._index = 0
        self.data = [ord(c) for c in self.strdata]
        self._size = len(self.data)

    @property
    def index(self):
        if False:
            return 10
        return self._index

    @property
    def size(self):
        if False:
            return 10
        return self._size

    def reset(self):
        if False:
            while True:
                i = 10
        self._index = 0

    def consume(self):
        if False:
            i = 10
            return i + 15
        if self._index >= self._size:
            assert self.LA(1) == Token.EOF
            raise Exception('cannot consume EOF')
        self._index += 1

    def LA(self, offset: int):
        if False:
            while True:
                i = 10
        if offset == 0:
            return 0
        if offset < 0:
            offset += 1
        pos = self._index + offset - 1
        if pos < 0 or pos >= self._size:
            return Token.EOF
        return self.data[pos]

    def LT(self, offset: int):
        if False:
            while True:
                i = 10
        return self.LA(offset)

    def mark(self):
        if False:
            for i in range(10):
                print('nop')
        return -1

    def release(self, marker: int):
        if False:
            while True:
                i = 10
        pass

    def seek(self, _index: int):
        if False:
            print('Hello World!')
        if _index <= self._index:
            self._index = _index
            return
        self._index = min(_index, self._size)

    def getText(self, start: int, stop: int):
        if False:
            for i in range(10):
                print('nop')
        if stop >= self._size:
            stop = self._size - 1
        if start >= self._size:
            return ''
        else:
            return self.strdata[start:stop + 1]

    def __str__(self):
        if False:
            print('Hello World!')
        return self.strdata
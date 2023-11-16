import io
import os

class Unreader(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.buf = io.BytesIO()

    def chunk(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def read(self, size=None):
        if False:
            return 10
        if size is not None and (not isinstance(size, int)):
            raise TypeError('size parameter must be an int or long.')
        if size is not None:
            if size == 0:
                return b''
            if size < 0:
                size = None
        self.buf.seek(0, os.SEEK_END)
        if size is None and self.buf.tell():
            ret = self.buf.getvalue()
            self.buf = io.BytesIO()
            return ret
        if size is None:
            d = self.chunk()
            return d
        while self.buf.tell() < size:
            chunk = self.chunk()
            if not chunk:
                ret = self.buf.getvalue()
                self.buf = io.BytesIO()
                return ret
            self.buf.write(chunk)
        data = self.buf.getvalue()
        self.buf = io.BytesIO()
        self.buf.write(data[size:])
        return data[:size]

    def unread(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.buf.seek(0, os.SEEK_END)
        self.buf.write(data)

class SocketUnreader(Unreader):

    def __init__(self, sock, max_chunk=8192):
        if False:
            print('Hello World!')
        super().__init__()
        self.sock = sock
        self.mxchunk = max_chunk

    def chunk(self):
        if False:
            print('Hello World!')
        return self.sock.recv(self.mxchunk)

class IterUnreader(Unreader):

    def __init__(self, iterable):
        if False:
            print('Hello World!')
        super().__init__()
        self.iter = iter(iterable)

    def chunk(self):
        if False:
            return 10
        if not self.iter:
            return b''
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = None
            return b''
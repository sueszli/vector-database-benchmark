try:
    import deflate, io
    io.IOBase
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit
if not hasattr(deflate.DeflateIO, 'write'):
    print('SKIP')
    raise SystemExit
formats = (deflate.RAW, deflate.ZLIB, deflate.GZIP)

class Stream(io.IOBase):

    def readinto(self, buf):
        if False:
            for i in range(10):
                print('nop')
        print('Stream.readinto', len(buf))
        return -1
try:
    deflate.DeflateIO(Stream()).read()
except OSError as er:
    print(repr(er))

class Stream(io.IOBase):

    def write(self, buf):
        if False:
            i = 10
            return i + 15
        print('Stream.write', buf)
        return -1
for format in formats:
    try:
        deflate.DeflateIO(Stream(), format).write('a')
    except OSError as er:
        print(repr(er))

class Stream(io.IOBase):

    def write(self, buf):
        if False:
            print('Hello World!')
        print('Stream.write', buf)
        return -1

    def ioctl(self, cmd, arg):
        if False:
            print('Hello World!')
        print('Stream.ioctl', cmd, arg)
        return 0
try:
    d = deflate.DeflateIO(Stream(), deflate.RAW, 0, True)
    d.close()
    d.write('a')
except OSError as er:
    print(repr(er))

class Stream(io.IOBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.num_writes = 0

    def write(self, buf):
        if False:
            while True:
                i = 10
        print('Stream.write', buf)
        if self.num_writes >= 4:
            return -1
        self.num_writes += 1
        return len(buf)
for format in formats:
    d = deflate.DeflateIO(Stream(), format)
    d.write('a')
    try:
        d.close()
    except OSError as er:
        print(repr(er))
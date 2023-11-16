try:
    import io, json
except ImportError:
    print('SKIP')
    raise SystemExit
if not hasattr(io, 'IOBase'):
    print('SKIP')
    raise SystemExit

class S(io.IOBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.buf = ''

    def write(self, buf):
        if False:
            i = 10
            return i + 15
        if type(buf) == bytearray:
            buf = str(buf, 'ascii')
        self.buf += buf
        return len(buf)
s = S()
json.dump([123, {}], s)
print(s.buf)
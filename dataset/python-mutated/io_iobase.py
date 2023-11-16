import io
try:
    io.IOBase
except AttributeError:
    print('SKIP')
    raise SystemExit

class MyIO(io.IOBase):

    def write(self, buf):
        if False:
            for i in range(10):
                print('nop')
        print('write', len(buf))
        return len(buf)
print('test', file=MyIO())
try:
    import io
    import ssl
except ImportError:
    print('SKIP')
    raise SystemExit

class TestSocket(io.IOBase):

    def write(self, buf):
        if False:
            print('Hello World!')
        return len(buf)

    def readinto(self, buf):
        if False:
            return 10
        return 0

    def ioctl(self, cmd, arg):
        if False:
            for i in range(10):
                print('nop')
        print('TestSocket.ioctl', cmd, arg)
        return 0

    def setblocking(self, value):
        if False:
            return 10
        print('TestSocket.setblocking({})'.format(value))
try:
    ss = ssl.wrap_socket(TestSocket(), server_hostname='test.example.com')
except OSError as er:
    print('OSError: client')
ss = ssl.wrap_socket(TestSocket(), server_side=1, do_handshake=0)
print(ss)
ss.setblocking(False)
ss.setblocking(True)
try:
    ss.write(b'aaaa')
except OSError:
    pass
try:
    ss.read(8)
except OSError:
    pass
ss.close()
ss.close()
try:
    ss.read(10)
except OSError as er:
    print('OSError: read')
try:
    ss.write(b'aaaa')
except OSError as er:
    print('OSError: write')
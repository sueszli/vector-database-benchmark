try:
    import sys, io, os
    io.IOBase
    os.mount
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class UserFile(io.IOBase):

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.data = memoryview(data)
        self.pos = 0

    def readinto(self, buf):
        if False:
            return 10
        n = min(len(buf), len(self.data) - self.pos)
        buf[:n] = self.data[self.pos:self.pos + n]
        self.pos += n
        return n

    def ioctl(self, req, arg):
        if False:
            for i in range(10):
                print('nop')
        return 0

class UserFS:

    def __init__(self, files):
        if False:
            return 10
        self.files = files

    def mount(self, readonly, mksfs):
        if False:
            i = 10
            return i + 15
        pass

    def umount(self):
        if False:
            i = 10
            return i + 15
        pass

    def stat(self, path):
        if False:
            print('Hello World!')
        if path in self.files:
            return (32768, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        raise OSError

    def open(self, path, mode):
        if False:
            i = 10
            return i + 15
        return UserFile(self.files[path])
user_files = {'/mod0.mpy': b'', '/mod1.mpy': b'M', '/mod2.mpy': b'M\x00\x00\x00'}
os.mount(UserFS(user_files), '/userfs')
sys.path.append('/userfs')
for i in range(len(user_files)):
    mod = 'mod%u' % i
    try:
        __import__(mod)
    except ValueError as er:
        print(mod, 'ValueError', er)
os.umount('/userfs')
sys.path.pop()
try:
    import sys, io, os
    sys.implementation._mpy
    io.IOBase
    os.mount
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit
mpy_arch = sys.implementation._mpy >> 8
if mpy_arch >> 2 == 0:
    print('SKIP')
    raise SystemExit

class UserFile(io.IOBase):

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        self.data = memoryview(data)
        self.pos = 0

    def readinto(self, buf):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
        self.files = files

    def mount(self, readonly, mksfs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def umount(self):
        if False:
            i = 10
            return i + 15
        pass

    def stat(self, path):
        if False:
            return 10
        if path in self.files:
            return (32768, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        raise OSError

    def open(self, path, mode):
        if False:
            for i in range(10):
                print('nop')
        return UserFile(self.files[path])
valid_header = bytes([77, 6, mpy_arch, 31])
user_files = {'/mod0.mpy': bytes([77, 6, 252 | mpy_arch, 31]), '/mod1.mpy': valid_header + b'\x02\x00\x0emod1.py\x00\nouter\x00,\x00\x02\x01Qc\x02B\x00\x00\x00\x00\x00\x00\x00\x00\x00C\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', '/mod2.mpy': valid_header + b'\x02\x00\x0emod2.py\x00\nouter\x00,\x00\x02\x01Qc\x01"\x00\x00\x00\x00p\x06\x04rodata\x03\x01\x00'}
os.mount(UserFS(user_files), '/userfs')
sys.path.append('/userfs')
for i in range(len(user_files)):
    mod = 'mod%u' % i
    try:
        __import__(mod)
        print(mod, 'OK')
    except ValueError as er:
        print(mod, 'ValueError', er)
os.umount('/userfs')
sys.path.pop()
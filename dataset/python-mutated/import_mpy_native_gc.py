try:
    import gc, sys, io, os
    sys.implementation._mpy
    io.IOBase
    os.mount
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class UserFile(io.IOBase):

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
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
            while True:
                i = 10
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
features0_file_contents = {2054: b'M\x06\n\x1f\x02\x004build/features0.native.mpy\x00\x12factorial\x00\x8a\x02\xe9/\x00\x00\x00SH\x8b\x1d\x83\x00\x00\x00\xbe\x02\x00\x00\x00\xffS\x18\xbf\x01\x00\x00\x00H\x85\xc0u\x0cH\x8bC \xbe\x02\x00\x00\x00[\xff\xe0H\x0f\xaf\xf8H\xff\xc8\xeb\xe6ATUSH\x8b\x1dQ\x00\x00\x00H\x8bG\x08L\x8bc(H\x8bx\x08A\xff\xd4H\x8d5+\x00\x00\x00H\x89\xc5H\x8b\x059\x00\x00\x00\x0f\xb7x\x02\xffShH\x89\xefA\xff\xd4H\x8b\x03[]A\\\xc3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x11$\r&\xa3 \x01"\xff', 4102: b'M\x06\x12\x1f\x02\x004build/features0.native.mpy\x00\x12factorial\x00\x88\x02\x18\xe0\x00\x00\x10\xb5\tK\tJ{D\x9cX\x02!\xe3h\x98G\x03\x00\x01 \x00+\x02\xd0XC\x01;\xfa\xe7\x02!#i\x98G\x10\xbd\xc0Fj\x00\x00\x00\x00\x00\x00\x00\xf8\xb5\nN\nK~D\xf4XChgiXh\xb8G\x05\x00\x07K\x08I\xf3XyDX\x88ck\x98G(\x00\xb8G h\xf8\xbd\xc0F:\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x11<\r>\xa38\x01:\xff'}
for arch in (5126, 6150, 7174, 8198):
    features0_file_contents[arch] = features0_file_contents[4102]
sys_implementation_mpy = sys.implementation._mpy & ~(3 << 8)
if sys_implementation_mpy not in features0_file_contents:
    print('SKIP')
    raise SystemExit
user_files = {'/features0.mpy': features0_file_contents[sys_implementation_mpy]}
os.mount(UserFS(user_files), '/userfs')
sys.path.append('/userfs')
gc.collect()
from features0 import factorial
del sys.modules['features0']
gc.collect()
for i in range(1000):
    []
print(factorial(10))
os.umount('/userfs')
sys.path.pop()
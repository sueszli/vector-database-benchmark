try:
    import io, os
    execfile
    io.IOBase
    os.mount
except (ImportError, NameError, AttributeError):
    print('SKIP')
    raise SystemExit

class File(io.IOBase):

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        self.data = data
        self.off = 0

    def ioctl(self, request, arg):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def readinto(self, buf):
        if False:
            print('Hello World!')
        buf[:] = memoryview(self.data)[self.off:self.off + len(buf)]
        self.off += len(buf)
        return len(buf)

class Filesystem:

    def __init__(self, files):
        if False:
            return 10
        self.files = files

    def mount(self, readonly, mkfs):
        if False:
            while True:
                i = 10
        print('mount', readonly, mkfs)

    def umount(self):
        if False:
            return 10
        print('umount')

    def open(self, file, mode):
        if False:
            for i in range(10):
                print('nop')
        print('open', file, mode)
        if file not in self.files:
            raise OSError(2)
        return File(self.files[file])
try:
    import io, os
    os.umount('/')
except OSError:
    pass
for path in os.listdir('/'):
    os.umount('/' + path)
files = {'/test.py': 'print(123)'}
fs = Filesystem(files)
os.mount(fs, '/test_mnt')
try:
    import io, os
    execfile('/test_mnt/noexist.py')
except OSError:
    print('OSError')
execfile('/test_mnt/test.py')
try:
    execfile(b'aaa')
except TypeError:
    print('TypeError')
os.umount(fs)
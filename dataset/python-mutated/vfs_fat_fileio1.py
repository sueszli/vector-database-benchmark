try:
    import errno
    import os
except ImportError:
    print('SKIP')
    raise SystemExit
try:
    os.VfsFat
except AttributeError:
    print('SKIP')
    raise SystemExit

class RAMFS:
    SEC_SIZE = 512

    def __init__(self, blocks):
        if False:
            i = 10
            return i + 15
        self.data = bytearray(blocks * self.SEC_SIZE)

    def readblocks(self, n, buf):
        if False:
            return 10
        for i in range(len(buf)):
            buf[i] = self.data[n * self.SEC_SIZE + i]

    def writeblocks(self, n, buf):
        if False:
            return 10
        for i in range(len(buf)):
            self.data[n * self.SEC_SIZE + i] = buf[i]

    def ioctl(self, op, arg):
        if False:
            print('Hello World!')
        if op == 4:
            return len(self.data) // self.SEC_SIZE
        if op == 5:
            return self.SEC_SIZE
try:
    bdev = RAMFS(50)
    os.VfsFat.mkfs(bdev)
except MemoryError:
    print('SKIP')
    raise SystemExit
vfs = os.VfsFat(bdev)
os.mount(vfs, '/ramdisk')
os.chdir('/ramdisk')
f = open('foo_file.txt', 'w')
print(str(f)[:17], str(f)[-1:])
f.write('hello!')
f.flush()
f.close()
f.close()
try:
    f.write('world!')
except OSError as e:
    print(e.errno == errno.EINVAL)
try:
    f.read()
except OSError as e:
    print(e.errno == errno.EINVAL)
try:
    f.flush()
except OSError as e:
    print(e.errno == errno.EINVAL)
try:
    open('foo_file.txt', 'x')
except OSError as e:
    print(e.errno == errno.EEXIST)
with open('foo_file.txt', 'a') as f:
    f.write('world!')
with open('foo_file.txt') as f2:
    print(f2.read())
    print(f2.tell())
    f2.seek(0, 0)
    print(f2.read(1))
    f2.seek(0, 1)
    print(f2.read(1))
    f2.seek(2, 1)
    print(f2.read(1))
    f2.seek(-2, 2)
    print(f2.read(1))
vfs.mkdir('foo_dir')
try:
    vfs.rmdir('foo_file.txt')
except OSError as e:
    print(e.errno == 20)
vfs.remove('foo_file.txt')
print(list(vfs.ilistdir()))
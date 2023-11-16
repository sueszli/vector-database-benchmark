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
            return 10
        self.data = bytearray(blocks * self.SEC_SIZE)

    def readblocks(self, n, buf):
        if False:
            print('Hello World!')
        for i in range(len(buf)):
            buf[i] = self.data[n * self.SEC_SIZE + i]

    def writeblocks(self, n, buf):
        if False:
            return 10
        for i in range(len(buf)):
            self.data[n * self.SEC_SIZE + i] = buf[i]

    def ioctl(self, op, arg):
        if False:
            for i in range(10):
                print('nop')
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
print(b'FOO_FILETXT' not in bdev.data)
print(b'hello!' not in bdev.data)
vfs = os.VfsFat(bdev)
os.mount(vfs, '/ramdisk')
print('statvfs:', vfs.statvfs('/ramdisk'))
print('getcwd:', vfs.getcwd())
try:
    vfs.stat('no_file.txt')
except OSError as e:
    print(e.errno == errno.ENOENT)
with vfs.open('foo_file.txt', 'w') as f:
    f.write('hello!')
print(list(vfs.ilistdir()))
print('stat root:', vfs.stat('/')[:-3])
print('stat file:', vfs.stat('foo_file.txt')[:-3])
print(b'FOO_FILETXT' in bdev.data)
print(b'hello!' in bdev.data)
vfs.mkdir('foo_dir')
vfs.chdir('foo_dir')
print('getcwd:', vfs.getcwd())
print(list(vfs.ilistdir()))
with vfs.open('sub_file.txt', 'w') as f:
    f.write('subdir file')
try:
    vfs.chdir('sub_file.txt')
except OSError as e:
    print(e.errno == errno.ENOENT)
vfs.chdir('..')
print('getcwd:', vfs.getcwd())
os.umount(vfs)
vfs = os.VfsFat(bdev)
print(list(vfs.ilistdir(b'')))
try:
    vfs.ilistdir(b'no_exist')
except OSError as e:
    print('ENOENT:', e.errno == errno.ENOENT)
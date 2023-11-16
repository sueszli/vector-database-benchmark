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
            print('Hello World!')
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
            i = 10
            return i + 15
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
try:
    vfs.mkdir('foo_dir')
except OSError as e:
    print(e.errno == errno.EEXIST)
try:
    vfs.remove('foo_dir')
except OSError as e:
    print(e.errno == errno.EISDIR)
try:
    vfs.remove('no_file.txt')
except OSError as e:
    print(e.errno == errno.ENOENT)
try:
    vfs.rename('foo_dir', '/null/file')
except OSError as e:
    print(e.errno == errno.ENOENT)
with open('foo_dir/file-in-dir.txt', 'w+t') as f:
    f.write('data in file')
with open('foo_dir/file-in-dir.txt', 'r+b') as f:
    print(f.read())
with open('foo_dir/sub_file.txt', 'w') as f:
    f.write('subdir file')
try:
    vfs.rmdir('foo_dir')
except OSError as e:
    print(e.errno == errno.EACCES)
vfs.rename('foo_dir/file-in-dir.txt', 'foo_dir/file.txt')
print(list(vfs.ilistdir('foo_dir')))
vfs.rename('foo_dir/file.txt', 'moved-to-root.txt')
print(list(vfs.ilistdir()))
with open('temp', 'w') as f:
    f.write('new text')
vfs.rename('temp', 'moved-to-root.txt')
print(list(vfs.ilistdir()))
with open('moved-to-root.txt') as f:
    print(f.read())
vfs.remove('foo_dir/sub_file.txt')
vfs.rmdir('foo_dir')
print(list(vfs.ilistdir()))
try:
    bsize = vfs.statvfs('/ramdisk')[0]
    free = vfs.statvfs('/ramdisk')[2] + 1
    f = open('large_file.txt', 'wb')
    f.write(bytearray(bsize * free))
except OSError as e:
    print('ENOSPC:', e.errno == 28)
f.close()
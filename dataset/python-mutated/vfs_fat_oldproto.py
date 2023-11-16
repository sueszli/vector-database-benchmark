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

class RAMFS_OLD:
    SEC_SIZE = 512

    def __init__(self, blocks):
        if False:
            for i in range(10):
                print('nop')
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

    def sync(self):
        if False:
            print('Hello World!')
        pass

    def count(self):
        if False:
            while True:
                i = 10
        return len(self.data) // self.SEC_SIZE
try:
    bdev = RAMFS_OLD(50)
except MemoryError:
    print('SKIP')
    raise SystemExit
os.VfsFat.mkfs(bdev)
vfs = os.VfsFat(bdev)
os.mount(vfs, '/ramdisk')
with vfs.open('file.txt', 'w') as f:
    f.write('hello!')
print(list(vfs.ilistdir()))
with vfs.open('file.txt', 'r') as f:
    print(f.read())
vfs.remove('file.txt')
print(list(vfs.ilistdir()))
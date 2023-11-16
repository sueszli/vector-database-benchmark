try:
    import os
except ImportError:
    print('SKIP')
    raise SystemExit
try:
    os.VfsFat
except AttributeError:
    print('SKIP')
    raise SystemExit

class RAMBDevSparse:
    SEC_SIZE = 512

    def __init__(self, blocks):
        if False:
            return 10
        self.blocks = blocks
        self.data = {}

    def readblocks(self, n, buf):
        if False:
            i = 10
            return i + 15
        assert len(buf) == self.SEC_SIZE
        if n not in self.data:
            self.data[n] = bytearray(self.SEC_SIZE)
        buf[:] = self.data[n]

    def writeblocks(self, n, buf):
        if False:
            while True:
                i = 10
        mv = memoryview(buf)
        for off in range(0, len(buf), self.SEC_SIZE):
            s = n + off // self.SEC_SIZE
            if s not in self.data:
                self.data[s] = bytearray(self.SEC_SIZE)
            self.data[s][:] = mv[off:off + self.SEC_SIZE]

    def ioctl(self, op, arg):
        if False:
            i = 10
            return i + 15
        if op == 4:
            return self.blocks
        if op == 5:
            return self.SEC_SIZE
try:
    bdev = RAMBDevSparse(4 * 1024 * 1024 * 1024 // RAMBDevSparse.SEC_SIZE)
    os.VfsFat.mkfs(bdev)
except MemoryError:
    print('SKIP')
    raise SystemExit
vfs = os.VfsFat(bdev)
os.mount(vfs, '/ramdisk')
print('statvfs:', vfs.statvfs('/ramdisk'))
f = open('/ramdisk/test.txt', 'w')
f.write('test file')
f.close()
print('statvfs:', vfs.statvfs('/ramdisk'))
f = open('/ramdisk/test.txt')
print(f.read())
f.close()
os.umount(vfs)
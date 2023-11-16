try:
    import os
    os.VfsFat
    os.VfsLfs2
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class RAMBlockDevice:
    ERASE_BLOCK_SIZE = 512

    def __init__(self, blocks):
        if False:
            print('Hello World!')
        self.data = bytearray(blocks * self.ERASE_BLOCK_SIZE)

    def readblocks(self, block, buf, off=0):
        if False:
            print('Hello World!')
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            buf[i] = self.data[addr + i]

    def writeblocks(self, block, buf, off=None):
        if False:
            while True:
                i = 10
        if off is None:
            off = 0
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            self.data[addr + i] = buf[i]

    def ioctl(self, op, arg):
        if False:
            print('Hello World!')
        if op == 4:
            return len(self.data) // self.ERASE_BLOCK_SIZE
        if op == 5:
            return self.ERASE_BLOCK_SIZE
        if op == 6:
            return 0

def test(bdev, vfs_class):
    if False:
        i = 10
        return i + 15
    print('test', vfs_class)
    vfs_class.mkfs(bdev)
    vfs = vfs_class(bdev)
    print(vfs.statvfs('/'))
    f = vfs.open('test', 'w')
    for i in range(10):
        f.write('some data')
    f.close()
    print(list(vfs.ilistdir()))
    with vfs.open('test', 'r') as f:
        print(f.read())
try:
    import os
    bdev = RAMBlockDevice(50)
except MemoryError:
    print('SKIP')
    raise SystemExit
test(bdev, os.VfsFat)
test(bdev, os.VfsLfs2)
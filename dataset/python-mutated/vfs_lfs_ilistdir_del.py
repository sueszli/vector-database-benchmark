import gc
try:
    import os
    os.VfsLfs2
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class RAMBlockDevice:
    ERASE_BLOCK_SIZE = 1024

    def __init__(self, blocks):
        if False:
            print('Hello World!')
        self.data = bytearray(blocks * self.ERASE_BLOCK_SIZE)

    def readblocks(self, block, buf, off):
        if False:
            while True:
                i = 10
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            buf[i] = self.data[addr + i]

    def writeblocks(self, block, buf, off):
        if False:
            print('Hello World!')
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            self.data[addr + i] = buf[i]

    def ioctl(self, op, arg):
        if False:
            for i in range(10):
                print('nop')
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
    vfs_class.mkfs(bdev)
    vfs = vfs_class(bdev)
    vfs.mkdir('/test_d1')
    vfs.mkdir('/test_d2')
    vfs.mkdir('/test_d3')
    for i in range(10):
        print(i)
        idir = vfs.ilistdir('/')
        print(any(idir))
        for (dname, *_) in vfs.ilistdir('/'):
            vfs.rmdir(dname)
            break
        vfs.mkdir(dname)
        idir_emptied = vfs.ilistdir('/')
        l = list(idir_emptied)
        print(len(l))
        try:
            next(idir_emptied)
        except StopIteration:
            pass
        gc.collect()
        vfs.open('/test', 'w').close()
bdev = RAMBlockDevice(30)
test(bdev, os.VfsLfs2)
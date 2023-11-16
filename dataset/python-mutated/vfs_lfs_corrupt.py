try:
    import os
    os.VfsLfs1
    os.VfsLfs2
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class RAMBlockDevice:
    ERASE_BLOCK_SIZE = 1024

    def __init__(self, blocks):
        if False:
            return 10
        self.data = bytearray(blocks * self.ERASE_BLOCK_SIZE)
        self.ret = 0

    def readblocks(self, block, buf, off):
        if False:
            i = 10
            return i + 15
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            buf[i] = self.data[addr + i]
        return self.ret

    def writeblocks(self, block, buf, off):
        if False:
            while True:
                i = 10
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            self.data[addr + i] = buf[i]
        return self.ret

    def ioctl(self, op, arg):
        if False:
            i = 10
            return i + 15
        if op == 4:
            return len(self.data) // self.ERASE_BLOCK_SIZE
        if op == 5:
            return self.ERASE_BLOCK_SIZE
        if op == 6:
            return 0

def corrupt(bdev, block):
    if False:
        while True:
            i = 10
    addr = block * bdev.ERASE_BLOCK_SIZE
    for i in range(bdev.ERASE_BLOCK_SIZE):
        bdev.data[addr + i] = i & 255

def create_vfs(bdev, vfs_class):
    if False:
        return 10
    bdev.ret = 0
    vfs_class.mkfs(bdev)
    vfs = vfs_class(bdev)
    with vfs.open('f', 'w') as f:
        for i in range(100):
            f.write('test')
    return vfs

def test(bdev, vfs_class):
    if False:
        for i in range(10):
            print('nop')
    print('test', vfs_class)
    vfs = create_vfs(bdev, vfs_class)
    corrupt(bdev, 0)
    corrupt(bdev, 1)
    try:
        print(vfs.statvfs(''))
    except OSError:
        print('statvfs OSError')
    vfs = create_vfs(bdev, vfs_class)
    f = vfs.open('f', 'r')
    bdev.ret = -5
    try:
        f.read(10)
    except OSError:
        print('read OSError')
    vfs = create_vfs(bdev, vfs_class)
    f = vfs.open('f', 'a')
    bdev.ret = -5
    try:
        f.write('test')
    except OSError:
        print('write OSError')
    vfs = create_vfs(bdev, vfs_class)
    f = vfs.open('f', 'w')
    f.write('test')
    bdev.ret = -5
    try:
        f.close()
    except OSError:
        print('close OSError')
    vfs = create_vfs(bdev, vfs_class)
    f = vfs.open('f', 'w')
    f.write('test')
    bdev.ret = -5
    try:
        f.flush()
    except OSError:
        print('flush OSError')
    bdev.ret = 0
    f.close()
bdev = RAMBlockDevice(30)
test(bdev, os.VfsLfs1)
test(bdev, os.VfsLfs2)
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

    def readblocks(self, block, buf, off):
        if False:
            while True:
                i = 10
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            buf[i] = self.data[addr + i]

    def writeblocks(self, block, buf, off):
        if False:
            i = 10
            return i + 15
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
    try:
        vfs_class.mkfs(RAMBlockDevice(1))
    except OSError:
        print('mkfs OSError')
    try:
        vfs_class(bdev)
    except OSError:
        print('mount OSError')
    vfs_class.mkfs(bdev)
    vfs = vfs_class(bdev)
    with vfs.open('testfile', 'w') as f:
        f.write('test')
    vfs.mkdir('testdir')
    try:
        vfs.ilistdir('noexist')
    except OSError:
        print('ilistdir OSError')
    try:
        vfs.remove('noexist')
    except OSError:
        print('remove OSError')
    try:
        vfs.rmdir('noexist')
    except OSError:
        print('rmdir OSError')
    try:
        vfs.rename('noexist', 'somethingelse')
    except OSError:
        print('rename OSError')
    try:
        vfs.mkdir('testdir')
    except OSError:
        print('mkdir OSError')
    try:
        vfs.chdir('noexist')
    except OSError:
        print('chdir OSError')
    print(vfs.getcwd())
    try:
        vfs.chdir('testfile')
    except OSError:
        print('chdir OSError')
    print(vfs.getcwd())
    try:
        vfs.stat('noexist')
    except OSError:
        print('stat OSError')
    with vfs.open('testfile', 'r') as f:
        f.seek(1 << 30)
        try:
            f.seek(1 << 30, 1)
        except OSError:
            print('seek OSError')
bdev = RAMBlockDevice(30)
test(bdev, os.VfsLfs1)
test(bdev, os.VfsLfs2)
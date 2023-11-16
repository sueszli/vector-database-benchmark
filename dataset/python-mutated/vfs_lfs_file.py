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
            i = 10
            return i + 15
        self.data = bytearray(blocks * self.ERASE_BLOCK_SIZE)

    def readblocks(self, block, buf, off):
        if False:
            print('Hello World!')
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            buf[i] = self.data[addr + i]

    def writeblocks(self, block, buf, off):
        if False:
            return 10
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            self.data[addr + i] = buf[i]

    def ioctl(self, op, arg):
        if False:
            return 10
        if op == 4:
            return len(self.data) // self.ERASE_BLOCK_SIZE
        if op == 5:
            return self.ERASE_BLOCK_SIZE
        if op == 6:
            return 0

def test(bdev, vfs_class):
    if False:
        for i in range(10):
            print('nop')
    print('test', vfs_class)
    vfs_class.mkfs(bdev)
    vfs = vfs_class(bdev)
    f = vfs.open('test.txt', 'wt')
    print(f)
    f.write('littlefs')
    f.close()
    f.close()
    f = vfs.open('test.bin', 'wb')
    print(f)
    f.write('littlefs')
    f.flush()
    f.close()
    f = vfs.open('test.bin', 'ab')
    f.write('more')
    f.close()
    f = vfs.open('test2.bin', 'xb')
    f.close()
    try:
        vfs.open('test2.bin', 'x')
    except OSError:
        print('open OSError')
    with vfs.open('test.txt', '') as f:
        print(f.read())
    with vfs.open('test.txt', 'rt') as f:
        print(f.read())
    with vfs.open('test.bin', 'rb') as f:
        print(f.read())
    with vfs.open('test.bin', 'r+b') as f:
        print(f.read(8))
        f.write('MORE')
    with vfs.open('test.bin', 'rb') as f:
        print(f.read())
    f = vfs.open('test.txt', 'r')
    print(f.tell())
    f.seek(3, 0)
    print(f.tell())
    f.close()
    try:
        vfs.open('noexist', 'r')
    except OSError:
        print('open OSError')
    f1 = vfs.open('test.txt', '')
    f2 = vfs.open('test.bin', 'b')
    print(f1.read())
    print(f2.read())
    f1.close()
    f2.close()
bdev = RAMBlockDevice(30)
test(bdev, os.VfsLfs1)
test(bdev, os.VfsLfs2)
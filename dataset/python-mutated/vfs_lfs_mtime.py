try:
    import time, os
    time.time
    time.sleep
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
            print('Hello World!')
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
            return 10
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
    print('mtime=True')
    vfs = vfs_class(bdev, mtime=True)
    current_time = int(time.time())
    vfs.open('test1', 'wt').close()
    time.sleep(1)
    vfs.open('test2', 'wt').close()
    stat1 = vfs.stat('test1')
    stat2 = vfs.stat('test2')
    print(stat1[8] != 0, stat2[8] != 0)
    print(current_time <= stat1[8] <= current_time + 1)
    print(stat1[8] < stat2[8])
    time.sleep(1)
    vfs.open('test1', 'rt').close()
    print(vfs.stat('test1') == stat1)
    vfs.open('test1', 'wt').close()
    stat1_old = stat1
    stat1 = vfs.stat('test1')
    print(stat1_old[8] < stat1[8])
    vfs.umount()
    print('mtime=False')
    vfs = vfs_class(bdev, mtime=False)
    print(vfs.stat('test1') == stat1)
    print(vfs.stat('test2') == stat2)
    f = vfs.open('test1', 'wt')
    f.close()
    print(vfs.stat('test1') == stat1)
    vfs.umount()
    print('mtime=True')
    vfs = vfs_class(bdev, mtime=True)
    print(vfs.stat('test1') == stat1)
    print(vfs.stat('test2') == stat2)
    vfs.umount()
try:
    bdev = RAMBlockDevice(30)
except MemoryError:
    print('SKIP')
    raise SystemExit
test(bdev, os.VfsLfs2)
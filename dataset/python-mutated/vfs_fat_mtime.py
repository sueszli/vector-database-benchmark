try:
    import time, os
    time.time
    time.sleep
    os.VfsFat
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class RAMBlockDevice:
    ERASE_BLOCK_SIZE = 512

    def __init__(self, blocks):
        if False:
            for i in range(10):
                print('nop')
        self.data = bytearray(blocks * self.ERASE_BLOCK_SIZE)

    def readblocks(self, block, buf):
        if False:
            while True:
                i = 10
        addr = block * self.ERASE_BLOCK_SIZE
        for i in range(len(buf)):
            buf[i] = self.data[addr + i]

    def writeblocks(self, block, buf):
        if False:
            for i in range(10):
                print('nop')
        addr = block * self.ERASE_BLOCK_SIZE
        for i in range(len(buf)):
            self.data[addr + i] = buf[i]

    def ioctl(self, op, arg):
        if False:
            while True:
                i = 10
        if op == 4:
            return len(self.data) // self.ERASE_BLOCK_SIZE
        if op == 5:
            return self.ERASE_BLOCK_SIZE

def test(bdev, vfs_class):
    if False:
        i = 10
        return i + 15
    print('test', vfs_class)
    vfs_class.mkfs(bdev)
    vfs = vfs_class(bdev)
    current_time = int(time.time())
    vfs.open('test1', 'wt').close()
    time.sleep(2)
    vfs.open('test2', 'wt').close()
    stat1 = vfs.stat('test1')
    stat2 = vfs.stat('test2')
    print(stat1[8] != 0, stat2[8] != 0)
    print(stat1[8] < stat2[8])
    vfs.umount()
bdev = RAMBlockDevice(50)
test(bdev, os.VfsFat)
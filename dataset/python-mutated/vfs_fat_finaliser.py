try:
    import errno, os
    os.VfsFat
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class RAMBlockDevice:

    def __init__(self, blocks, sec_size=512):
        if False:
            print('Hello World!')
        self.sec_size = sec_size
        self.data = bytearray(blocks * self.sec_size)

    def readblocks(self, n, buf):
        if False:
            for i in range(10):
                print('nop')
        for i in range(len(buf)):
            buf[i] = self.data[n * self.sec_size + i]

    def writeblocks(self, n, buf):
        if False:
            while True:
                i = 10
        for i in range(len(buf)):
            self.data[n * self.sec_size + i] = buf[i]

    def ioctl(self, op, arg):
        if False:
            return 10
        if op == 4:
            return len(self.data) // self.sec_size
        if op == 5:
            return self.sec_size
try:
    import errno, os
    bdev = RAMBlockDevice(50)
except MemoryError:
    print('SKIP')
    raise SystemExit
os.VfsFat.mkfs(bdev)
vfs = os.VfsFat(bdev)
import micropython
micropython.heap_lock()
try:
    import errno, os
    vfs.open('x', 'r')
except MemoryError:
    print('MemoryError')
micropython.heap_unlock()
import gc
f = None
n = None
names = ['x%d' % i for i in range(5)]
for i in range(1024):
    []
for n in names:
    f = vfs.open(n, 'w')
    f.write(n)
    f = None
gc.collect()
for n in names[:-1]:
    with vfs.open(n, 'r') as f:
        print(f.read())
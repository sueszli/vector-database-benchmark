import errno
import os as os
try:
    os.VfsFat
except AttributeError:
    print('SKIP')
    raise SystemExit

class RAMFS:
    SEC_SIZE = 512

    def __init__(self, blocks):
        if False:
            i = 10
            return i + 15
        self.data = bytearray(blocks * self.SEC_SIZE)

    def readblocks(self, n, buf):
        if False:
            for i in range(10):
                print('nop')
        for i in range(len(buf)):
            buf[i] = self.data[n * self.SEC_SIZE + i]
        return 0

    def writeblocks(self, n, buf):
        if False:
            for i in range(10):
                print('nop')
        for i in range(len(buf)):
            self.data[n * self.SEC_SIZE + i] = buf[i]
        return 0

    def ioctl(self, op, arg):
        if False:
            print('Hello World!')
        if op == 4:
            return len(self.data) // self.SEC_SIZE
        if op == 5:
            return self.SEC_SIZE
try:
    bdev = RAMFS(50)
except MemoryError:
    print('SKIP')
    raise SystemExit
os.VfsFat.mkfs(bdev)
vfs = os.VfsFat(bdev)
os.mount(vfs, '/ramdisk')
os.chdir('/ramdisk')
vfs.label = 'labelæ'
print(vfs.label)
vfs.mkdir('fooaz')
print(os.listdir(''))
vfs.rmdir('fOOAZ')
vfs.mkdir('123456789fooaz')
print(os.listdir(''))
vfs.rmdir('123456789fOOAZ')
vfs.mkdir('extended_æ')
print(os.listdir(''))
try:
    vfs.rmdir('extended_Æ')
except OSError as e:
    print(e.errno == errno.ENOENT)
vfs.rmdir('extended_æ')
vfs.mkdir('emoji_😀')
vfs.rmdir('emoji_😀')
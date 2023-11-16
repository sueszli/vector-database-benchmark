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
            print('Hello World!')
        self.data = bytearray(blocks * self.ERASE_BLOCK_SIZE)

    def readblocks(self, block, buf, off=0):
        if False:
            print('Hello World!')
        addr = block * self.ERASE_BLOCK_SIZE + off
        for i in range(len(buf)):
            buf[i] = self.data[addr + i]

    def writeblocks(self, block, buf, off=0):
        if False:
            return 10
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

def test(vfs_class):
    if False:
        return 10
    print('test', vfs_class)
    bdev = RAMBlockDevice(30)
    try:
        os.mount(bdev, '/lfs')
    except Exception as er:
        print(repr(er))
    vfs_class.mkfs(bdev)
    vfs = vfs_class(bdev)
    os.mount(vfs, '/lfs')
    with open('/lfs/lfsmod.py', 'w') as f:
        f.write('print("hello from lfs")\n')
    import lfsmod
    os.mkdir('/lfs/lfspkg')
    with open('/lfs/lfspkg/__init__.py', 'w') as f:
        f.write('print("package")\n')
    import lfspkg
    os.mkdir('/lfs/subdir')
    os.chdir('/lfs/subdir')
    os.rename('/lfs/lfsmod.py', '/lfs/subdir/lfsmod2.py')
    import lfsmod2
    os.umount('/lfs')
    vfs = vfs_class(bdev)
    os.mount(vfs, '/lfs', readonly=True)
    with open('/lfs/subdir/lfsmod2.py') as f:
        print('lfsmod2.py:', f.read())
    try:
        open('/lfs/test_write', 'w')
    except OSError as er:
        print(repr(er))
    os.umount('/lfs')
    os.mount(bdev, '/lfs')
    os.umount('/lfs')
    sys.modules.clear()
import sys
sys.path.clear()
sys.path.append('/lfs')
sys.path.append('')
test(os.VfsLfs1)
test(os.VfsLfs2)
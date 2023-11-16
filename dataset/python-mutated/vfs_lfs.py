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
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
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

def print_stat(st, print_size=True):
    if False:
        print('Hello World!')
    print(st[:6], st[6] if print_size else -1, type(st[7]), type(st[8]), type(st[9]))

def test(bdev, vfs_class):
    if False:
        i = 10
        return i + 15
    print('test', vfs_class)
    vfs_class.mkfs(bdev)
    vfs = vfs_class(bdev)
    print(vfs.statvfs('/'))
    f = vfs.open('test', 'w')
    f.write('littlefs')
    f.close()
    print(vfs.statvfs('/'))
    print(list(vfs.ilistdir()))
    print(list(vfs.ilistdir('/')))
    print(list(vfs.ilistdir(b'/')))
    vfs.mkdir('testdir')
    print(list(vfs.ilistdir()))
    print(sorted(list(vfs.ilistdir('testdir'))))
    vfs.rmdir('testdir')
    print(list(vfs.ilistdir()))
    vfs.mkdir('testdir')
    print_stat(vfs.stat('test'))
    print_stat(vfs.stat('testdir'), False)
    with vfs.open('test', 'r') as f:
        print(f.read())
    with vfs.open('testbig', 'w') as f:
        data = 'large012' * 32 * 16
        print('data length:', len(data))
        for i in range(4):
            print('write', i)
            f.write(data)
    print(vfs.statvfs('/'))
    vfs.rename('testbig', 'testbig2')
    print(sorted(list(vfs.ilistdir())))
    vfs.chdir('testdir')
    vfs.rename('/testbig2', 'testbig2')
    print(sorted(list(vfs.ilistdir())))
    vfs.rename('testbig2', '/testbig2')
    vfs.chdir('/')
    print(sorted(list(vfs.ilistdir())))
    vfs.remove('testbig2')
    print(sorted(list(vfs.ilistdir())))
    vfs.mkdir('/testdir2')
    vfs.mkdir('/testdir/subdir')
    print(vfs.getcwd())
    vfs.chdir('/testdir')
    print(vfs.getcwd())
    vfs.open('test2', 'w').close()
    print_stat(vfs.stat('test2'))
    print_stat(vfs.stat('/testdir/test2'))
    vfs.remove('test2')
    vfs.chdir('/')
    print(vfs.getcwd())
    vfs.chdir('testdir')
    print(vfs.getcwd())
    vfs.chdir('..')
    print(vfs.getcwd())
    vfs.chdir('testdir/subdir')
    print(vfs.getcwd())
    vfs.chdir('../..')
    print(vfs.getcwd())
    vfs.chdir('/./testdir2')
    print(vfs.getcwd())
    vfs.chdir('../testdir')
    print(vfs.getcwd())
    vfs.chdir('../..')
    print(vfs.getcwd())
    vfs.chdir('.//testdir')
    print(vfs.getcwd())
    vfs.chdir('subdir/./')
    print(vfs.getcwd())
    vfs.chdir('/')
    print(vfs.getcwd())
    vfs.rmdir('testdir/subdir')
    vfs.rmdir('testdir')
    vfs.rmdir('testdir2')
bdev = RAMBlockDevice(30)
test(bdev, os.VfsLfs1)
test(bdev, os.VfsLfs2)
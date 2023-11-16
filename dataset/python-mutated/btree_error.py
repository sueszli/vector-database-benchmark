try:
    import btree, io, errno
    io.IOBase
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class Device(io.IOBase):

    def __init__(self, read_ret=0, ioctl_ret=0):
        if False:
            while True:
                i = 10
        self.read_ret = read_ret
        self.ioctl_ret = ioctl_ret

    def readinto(self, buf):
        if False:
            for i in range(10):
                print('nop')
        print('read', len(buf))
        return self.read_ret

    def ioctl(self, cmd, arg):
        if False:
            return 10
        print('ioctl', cmd)
        return self.ioctl_ret
try:
    import btree, io, errno
    db = btree.open(Device(), pagesize=511)
except OSError as er:
    print('OSError', er.errno == errno.EINVAL)
try:
    import btree, io, errno
    db = btree.open(Device(-1000), pagesize=512)
except OSError as er:
    print(repr(er))
try:
    import btree, io, errno
    db = btree.open(Device(0, -1001), pagesize=512)
except OSError as er:
    print(repr(er))
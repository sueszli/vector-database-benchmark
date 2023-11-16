import sys
try:
    import io
    io.IOBase
    import os
    os.mount
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class UserFile(io.IOBase):
    buffer_size = 16

    def __init__(self, mode, data):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(data, bytes)
        self.is_text = mode.find('b') == -1
        self.data = data
        self.pos = 0

    def read(self):
        if False:
            return 10
        if self.is_text:
            return str(self.data, 'utf8')
        else:
            return self.data

    def readinto(self, buf):
        if False:
            return 10
        assert not self.is_text
        n = 0
        while n < len(buf) and self.pos < len(self.data):
            buf[n] = self.data[self.pos]
            n += 1
            self.pos += 1
        return n

    def ioctl(self, req, arg):
        if False:
            while True:
                i = 10
        print('ioctl', req, arg)
        if req == 4:
            return 0
        if req == 11:
            return UserFile.buffer_size
        return -1

class UserFS:

    def __init__(self, files):
        if False:
            i = 10
            return i + 15
        self.files = files

    def mount(self, readonly, mksfs):
        if False:
            print('Hello World!')
        pass

    def umount(self):
        if False:
            return 10
        pass

    def stat(self, path):
        if False:
            print('Hello World!')
        print('stat', path)
        if path in self.files:
            return (32768, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        raise OSError

    def open(self, path, mode):
        if False:
            return 10
        print('open', path, mode)
        return UserFile(mode, self.files[path])
user_files = {'/data.txt': b'some data in a text file', '/usermod1.py': b"print('in usermod1')\nimport usermod2", '/usermod2.py': b"print('in usermod2')", '/usermod3.py': b'syntax error', '/usermod4.mpy': b'syntax error', '/usermod5.py': b"print('in usermod5')", '/usermod6.py': b"print('in usermod6')"}
os.mount(UserFS(user_files), '/userfs')
f = open('/userfs/data.txt')
print(f.read())
sys.path.append('/userfs')
import usermod1
try:
    import usermod3
except SyntaxError:
    print('SyntaxError in usermod3')
try:
    import usermod4
except ValueError:
    print('ValueError in usermod4')
UserFile.buffer_size = 255
import usermod5
UserFile.buffer_size = 1024
import usermod6
os.umount('/userfs')
sys.path.pop()
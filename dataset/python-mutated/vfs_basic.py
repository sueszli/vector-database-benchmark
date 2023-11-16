try:
    import os
    os.mount
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

class Filesystem:

    def __init__(self, id, fail=0):
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        self.fail = fail

    def mount(self, readonly, mkfs):
        if False:
            return 10
        print(self.id, 'mount', readonly, mkfs)

    def umount(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.id, 'umount')

    def ilistdir(self, dir):
        if False:
            return 10
        print(self.id, 'ilistdir', dir)
        return iter([('a%d' % self.id, 0, 0)])

    def chdir(self, dir):
        if False:
            return 10
        print(self.id, 'chdir', dir)
        if self.fail:
            raise OSError(self.fail)

    def getcwd(self):
        if False:
            i = 10
            return i + 15
        print(self.id, 'getcwd')
        return 'dir%d' % self.id

    def mkdir(self, path):
        if False:
            i = 10
            return i + 15
        print(self.id, 'mkdir', path)

    def remove(self, path):
        if False:
            while True:
                i = 10
        print(self.id, 'remove', path)

    def rename(self, old_path, new_path):
        if False:
            for i in range(10):
                print('nop')
        print(self.id, 'rename', old_path, new_path)

    def rmdir(self, path):
        if False:
            return 10
        print(self.id, 'rmdir', path)

    def stat(self, path):
        if False:
            for i in range(10):
                print('nop')
        print(self.id, 'stat', path)
        return (self.id,)

    def statvfs(self, path):
        if False:
            i = 10
            return i + 15
        print(self.id, 'statvfs', path)
        return (self.id,)

    def open(self, file, mode):
        if False:
            print('Hello World!')
        print(self.id, 'open', file, mode)
try:
    os.umount('/')
except OSError:
    pass
for path in os.listdir('/'):
    os.umount('/' + path)
print(os.stat('/'))
print(os.statvfs('/')[9] >= 32)
print(os.getcwd())
for func in ('chdir', 'listdir', 'mkdir', 'remove', 'rmdir', 'stat'):
    for arg in ('x', '/x'):
        try:
            getattr(os, func)(arg)
        except OSError:
            print(func, arg, 'OSError')
os.mount(Filesystem(1), '/test_mnt')
print(os.listdir())
i = os.ilistdir()
print(next(i))
try:
    next(i)
except StopIteration:
    print('StopIteration')
try:
    next(i)
except StopIteration:
    print('StopIteration')
print(os.listdir('test_mnt'))
print(os.listdir('/test_mnt'))
os.mount(Filesystem(2), '/test_mnt2', readonly=True)
print(os.listdir())
print(os.listdir('/test_mnt2'))
try:
    os.mount(Filesystem(3), '/test_mnt2')
except OSError:
    print('OSError')
try:
    os.mkdir('/test_mnt')
except OSError:
    print('OSError')
try:
    os.rename('/test_mnt/a', '/test_mnt2/b')
except OSError:
    print('OSError')
os.chdir('test_mnt')
print(os.listdir())
print(os.getcwd())
os.mkdir('test_dir')
os.remove('test_file')
os.rename('test_file', 'test_file2')
os.rmdir('test_dir')
print(os.stat('test_file'))
print(os.statvfs('/test_mnt'))
open('test_file')
open('test_file', 'wb')
os.umount('/test_mnt')
os.umount('/test_mnt2')
try:
    os.umount('/test_mnt')
except OSError:
    print('OSError')
os.mount(Filesystem(3), '/')
print(os.stat('/'))
print(os.statvfs('/'))
print(os.listdir())
open('test')
os.mount(Filesystem(4), '/mnt')
print(os.listdir())
print(os.listdir('/mnt'))
os.chdir('/mnt')
print(os.listdir())
os.chdir('/subdir')
print(os.listdir())
os.chdir('/')
os.umount('/')
print(os.listdir('/'))
os.umount('/mnt')
try:
    os.chdir('/foo')
except OSError:
    print('OSError')
print(os.getcwd())
os.mount(Filesystem(5, 1), '/mnt')
try:
    os.chdir('/mnt/subdir')
except OSError:
    print('OSError')
print(os.getcwd())
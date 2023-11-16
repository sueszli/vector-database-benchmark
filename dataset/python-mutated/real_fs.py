import os
import stat
from trashcli import fs
from trashcli.fs import write_file
from trashcli.put.fs.fs import Fs

class RealFs(Fs):

    def atomic_write(self, path, content):
        if False:
            print('Hello World!')
        fs.atomic_write(path, content)

    def chmod(self, path, mode):
        if False:
            print('Hello World!')
        os.chmod(path, mode)

    def isdir(self, path):
        if False:
            return 10
        return os.path.isdir(path)

    def isfile(self, path):
        if False:
            return 10
        return os.path.isfile(path)

    def getsize(self, path):
        if False:
            i = 10
            return i + 15
        return os.path.getsize(path)

    def walk_no_follow(self, path):
        if False:
            while True:
                i = 10
        try:
            import scandir
            walk = scandir.walk
        except ImportError:
            walk = os.walk
        return walk(path, followlinks=False)

    def exists(self, path):
        if False:
            for i in range(10):
                print('nop')
        return os.path.exists(path)

    def makedirs(self, path, mode):
        if False:
            for i in range(10):
                print('nop')
        os.makedirs(path, mode)

    def move(self, path, dest):
        if False:
            for i in range(10):
                print('nop')
        return fs.move(path, dest)

    def remove_file(self, path):
        if False:
            for i in range(10):
                print('nop')
        fs.remove_file(path)

    def islink(self, path):
        if False:
            i = 10
            return i + 15
        return os.path.islink(path)

    def has_sticky_bit(self, path):
        if False:
            print('Hello World!')
        return os.stat(path).st_mode & stat.S_ISVTX == stat.S_ISVTX

    def realpath(self, path):
        if False:
            i = 10
            return i + 15
        return os.path.realpath(path)

    def is_accessible(self, path):
        if False:
            return 10
        return os.access(path, os.F_OK)

    def make_file(self, path, content):
        if False:
            print('Hello World!')
        write_file(path, content)

    def get_mod(self, path):
        if False:
            print('Hello World!')
        return stat.S_IMODE(os.lstat(path).st_mode)

    def listdir(self, path):
        if False:
            print('Hello World!')
        return os.listdir(path)

    def lexists(self, path):
        if False:
            print('Hello World!')
        return os.path.lexists(path)
import os
from typing import Union
from tests.test_put.support.fake_fs.directory import Directory, make_inode_for_dir
from tests.test_put.support.fake_fs.file import File
from tests.test_put.support.fake_fs.inode import SymLink
from tests.test_put.support.format_mode import format_mode
from tests.test_put.support.my_file_not_found_error import MyFileNotFoundError
from trashcli.fs import PathExists
from trashcli.put.fs.fs import Fs

class FakeFs(Fs, PathExists):

    def __init__(self, cwd='/'):
        if False:
            for i in range(10):
                print('nop')
        directory = Directory('/')
        make_inode_for_dir(directory, 493)
        self.root = directory
        self.cwd = cwd

    def touch(self, path):
        if False:
            while True:
                i = 10
        if not self.exists(path):
            self.make_file(path, '')

    def listdir(self, path):
        if False:
            i = 10
            return i + 15
        return self.ls_aa(path)

    def ls_existing(self, paths):
        if False:
            return 10
        return [p for p in paths if self.exists(p)]

    def ls_aa(self, path):
        if False:
            while True:
                i = 10
        all_entries = self.ls_a(path)
        all_entries.remove('.')
        all_entries.remove('..')
        return all_entries

    def ls_a(self, path):
        if False:
            return 10
        dir = self.find_dir_or_file(path)
        return list(dir.entries())

    def mkdir(self, path):
        if False:
            for i in range(10):
                print('nop')
        (dirname, basename) = os.path.split(path)
        dir = self.find_dir_or_file(dirname)
        dir.add_dir(basename, 493, path)

    def find_dir_or_file(self, path):
        if False:
            return 10
        path = os.path.join(self.cwd, path)
        if path == '/':
            return self.root
        cur_dir = self.root
        for component in self.components_for(path):
            try:
                cur_dir = cur_dir.get_file(component)
            except KeyError:
                raise MyFileNotFoundError('no such file or directory: %s\n%s' % (path, '\n'.join(self.list_all())))
        return cur_dir

    def components_for(self, path):
        if False:
            print('Hello World!')
        return path.split('/')[1:]

    def atomic_write(self, path, content):
        if False:
            return 10
        if self.exists(path):
            raise OSError('already exists: %s' % path)
        self.make_file(path, content)

    def read(self, path):
        if False:
            for i in range(10):
                print('nop')
        return self.find_dir_or_file(path).content

    def read_null(self, path):
        if False:
            i = 10
            return i + 15
        try:
            return self.find_dir_or_file(path).content
        except MyFileNotFoundError:
            return None

    def make_file(self, path, content=''):
        if False:
            i = 10
            return i + 15
        path = os.path.join(self.cwd, path)
        (dirname, basename) = os.path.split(path)
        dir = self.find_dir_or_file(dirname)
        dir.add_file(basename, content, path)

    def get_mod(self, path):
        if False:
            return 10
        entry = self._find_entry(path)
        return entry.mode

    def _find_entry(self, path):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.join(self.cwd, path)
        (dirname, basename) = os.path.split(path)
        dir = self.find_dir_or_file(dirname)
        return dir._get_entry(basename)

    def chmod(self, path, mode):
        if False:
            print('Hello World!')
        entry = self._find_entry(path)
        entry.chmod(mode)

    def isdir(self, path):
        if False:
            print('Hello World!')
        try:
            file = self.find_dir_or_file(path)
        except MyFileNotFoundError:
            return False
        return isinstance(file, Directory)

    def exists(self, path):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.find_dir_or_file(path)
            return True
        except MyFileNotFoundError:
            return False

    def remove_file(self, path):
        if False:
            while True:
                i = 10
        (dirname, basename) = os.path.split(path)
        dir = self.find_dir_or_file(dirname)
        dir.remove(basename)

    def makedirs(self, path, mode):
        if False:
            while True:
                i = 10
        cur_dir = self.root
        for component in self.components_for(path):
            try:
                cur_dir = cur_dir.get_file(component)
            except KeyError:
                cur_dir.add_dir(component, mode, path)
                cur_dir = cur_dir.get_file(component)

    def move(self, src, dest):
        if False:
            for i in range(10):
                print('nop')
        (basename, entry) = self._pop_entry_from_dir(src)
        if self.exists(dest) and self.isdir(dest):
            dest_dir = self.find_dir_or_file(dest)
            dest_dir._add_entry(basename, entry)
        else:
            (dest_dirname, dest_basename) = os.path.split(dest)
            dest_dir = self.find_dir_or_file(dest_dirname)
            dest_dir._add_entry(dest_basename, entry)

    def _pop_entry_from_dir(self, path):
        if False:
            i = 10
            return i + 15
        (dirname, basename) = os.path.split(path)
        dir = self.find_dir_or_file(dirname)
        entry = dir._get_entry(basename)
        dir.remove(basename)
        return (basename, entry)

    def islink(self, path):
        if False:
            return 10
        try:
            entry = self._find_entry(path)
        except MyFileNotFoundError:
            return False
        else:
            return isinstance(entry, SymLink)

    def symlink(self, src, dest):
        if False:
            print('Hello World!')
        dest = os.path.join(self.cwd, dest)
        (dirname, basename) = os.path.split(dest)
        if dirname == '':
            raise OSError('only absolute dests are supported, got %s' % dest)
        dir = self.find_dir_or_file(dirname)
        dir.add_link(basename, src)

    def has_sticky_bit(self, path):
        if False:
            i = 10
            return i + 15
        return self._find_entry(path).sticky

    def set_sticky_bit(self, path):
        if False:
            i = 10
            return i + 15
        entry = self._find_entry(path)
        entry.sticky = True

    def realpath(self, path):
        if False:
            return 10
        return os.path.join('/', path)

    def cd(self, path):
        if False:
            return 10
        self.cwd = path

    def isfile(self, path):
        if False:
            print('Hello World!')
        try:
            file = self.find_dir_or_file(path)
        except MyFileNotFoundError:
            return False
        return isinstance(file, File)

    def getsize(self, path):
        if False:
            while True:
                i = 10
        file = self.find_dir_or_file(path)
        return file.getsize()

    def is_accessible(self, path):
        if False:
            print('Hello World!')
        return self.exists(path)

    def get_mod_s(self, path):
        if False:
            return 10
        mode = self.get_mod(path)
        return format_mode(mode)

    def walk_no_follow(self, top):
        if False:
            for i in range(10):
                print('nop')
        names = self.listdir(top)
        (dirs, nondirs) = ([], [])
        for name in names:
            if self.isdir(os.path.join(top, name)):
                dirs.append(name)
            else:
                nondirs.append(name)
        yield (top, dirs, nondirs)
        for name in dirs:
            new_path = os.path.join(top, name)
            if not self.islink(new_path):
                for x in self.walk_no_follow(new_path):
                    yield x

    def lexists(self, path):
        if False:
            print('Hello World!')
        return self.exists(path)

    def list_all(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.walk_no_follow('/')
        for (top, dirs, non_dirs) in result:
            for d in dirs:
                yield os.path.join(top, d)
            for f in non_dirs:
                yield os.path.join(top, f)
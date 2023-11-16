import os
from typing import List
from trashcli.lib.dir_reader import DirReader

class MockDirReader(DirReader):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.root = {}

    def entries_if_dir_exists(self, path):
        if False:
            while True:
                i = 10
        return list(self.pick_dir(path).keys())

    def exists(self, path):
        if False:
            return 10
        raise NotImplementedError()

    def add_file(self, path):
        if False:
            for i in range(10):
                print('nop')
        (dirname, basename) = os.path.split(path)
        dir = self.pick_dir(dirname)
        dir[basename] = ''

    def mkdir(self, path):
        if False:
            i = 10
            return i + 15
        (dirname, basename) = os.path.split(path)
        cwd = self.pick_dir(dirname)
        cwd[basename] = {}

    def pick_dir(self, dir):
        if False:
            while True:
                i = 10
        cwd = self.root
        components = dir.split('/')[1:]
        if components != ['']:
            for p in components:
                if p not in cwd:
                    raise FileNotFoundError('no such file or directory: %s' % dir)
                cwd = cwd[p]
        return cwd
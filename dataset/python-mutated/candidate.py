import os
import posixpath
import re
from typing import NamedTuple
from trashcli.put.core.check_type import CheckType
from trashcli.put.core.path_maker_type import PathMakerType
from trashcli.put.gate import Gate

class Candidate(NamedTuple('Candidate', [('trash_dir_path', str), ('volume', str), ('path_maker_type', PathMakerType), ('check_type', CheckType), ('gate', Gate)])):

    def parent_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return os.path.dirname(self.trash_dir_path)

    def info_dir(self):
        if False:
            while True:
                i = 10
        return os.path.join(self.trash_dir_path, 'info')

    def files_dir(self):
        if False:
            print('Hello World!')
        return os.path.join(self.trash_dir_path, 'files')

    def norm_path(self):
        if False:
            while True:
                i = 10
        return os.path.normpath(self.trash_dir_path)

    def shrink_user(self, environ):
        if False:
            print('Hello World!')
        path = self.norm_path()
        if environ.get('TRASH_PUT_DISABLE_SHRINK', '') == '1':
            return path
        home_dir = environ.get('HOME', '')
        home_dir = posixpath.normpath(home_dir)
        if home_dir != '':
            path = re.sub('^' + re.escape(home_dir + os.path.sep), '~' + os.path.sep, path)
        return path
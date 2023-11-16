import os
from trashcli.put.fs.real_fs import RealFs

class Describer:

    def __init__(self, fs):
        if False:
            while True:
                i = 10
        self.fs = fs

    def describe(self, path):
        if False:
            while True:
                i = 10
        '\n        Return a textual description of the file pointed by this path.\n        Options:\n         - "symbolic link"\n         - "directory"\n         - "\'.\' directory"\n         - "\'..\' directory"\n         - "regular file"\n         - "regular empty file"\n         - "non existent"\n         - "entry"\n        '
        if self.fs.islink(path):
            return 'symbolic link'
        elif self.fs.isdir(path):
            if path == '.':
                return 'directory'
            elif path == '..':
                return 'directory'
            elif os.path.basename(path) == '.':
                return "'.' directory"
            elif os.path.basename(path) == '..':
                return "'..' directory"
            else:
                return 'directory'
        elif self.fs.isfile(path):
            if self.fs.getsize(path) == 0:
                return 'regular empty file'
            else:
                return 'regular file'
        elif not self.fs.exists(path):
            return 'non existent'
        else:
            return 'entry'
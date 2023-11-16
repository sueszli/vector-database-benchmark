from trashcli.put.fs.fs import Fs

class DirMaker:

    def __init__(self, fs):
        if False:
            while True:
                i = 10
        self.fs = fs

    def mkdir_p(self, path, mode):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.fs.makedirs(path, mode)
        except OSError:
            if not self.fs.isdir(path):
                raise
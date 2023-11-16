import os

class ParentRealpathFs:

    def __init__(self, fs):
        if False:
            return 10
        self.fs = fs

    def parent_realpath(self, path):
        if False:
            while True:
                i = 10
        parent = os.path.dirname(path)
        return self.fs.realpath(parent)
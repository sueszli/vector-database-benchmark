import os
import shutil
import tempfile

class MyPath(str):

    def __truediv__(self, other_path):
        if False:
            while True:
                i = 10
        return self.path_join(other_path)

    def __div__(self, other_path):
        if False:
            for i in range(10):
                print('nop')
        return self.path_join(other_path)

    def path_join(self, other_path):
        if False:
            return 10
        return MyPath(os.path.join(self, other_path))

    def clean_up(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self)

    @classmethod
    def make_temp_dir(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls(os.path.realpath(tempfile.mkdtemp(suffix='_trash_cli_test')))
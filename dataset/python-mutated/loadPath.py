import os
import sys
from pathlib import Path

class loadPath:

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        syspath = sys.prefix
        correct_syspath = Path(syspath)
        filepath = correct_syspath / 'path_file.txt'
        if os.path.exists(filepath):
            with open('path_file.txt', 'r') as f:
                self.path = f.readline()
                return self.path
        else:
            return ''

    def storage(self, path):
        if False:
            while True:
                i = 10
        syspath = sys.prefix
        correct_syspath = Path(syspath)
        filepath = correct_syspath / 'path_file.txt'
        with open(filepath, 'w') as f:
            f.write(path)
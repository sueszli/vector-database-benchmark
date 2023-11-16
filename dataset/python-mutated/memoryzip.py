"""
@author: longofo
@file: memoryzip.py
@time: 2020/03/18
"""
import zipfile
from io import BytesIO

class InMemoryZip(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.in_memory_zip = BytesIO()

    def add_file(self, filename_in_zip, file_contents):
        if False:
            i = 10
            return i + 15
        zf = zipfile.ZipFile(self.in_memory_zip, 'a', zipfile.ZIP_DEFLATED)
        zf.writestr(filename_in_zip, file_contents)
        return self

    def read(self):
        if False:
            while True:
                i = 10
        self.in_memory_zip.seek(0)
        return self.in_memory_zip.read()

    def write_to_file(self, filename):
        if False:
            for i in range(10):
                print('nop')
        with open(filename, 'wb') as f:
            f.write(self.read())
"""
Topic: 目录文件列表
Desc : 
"""
import os
import os.path
import glob
from fnmatch import fnmatch

def dir_listfile():
    if False:
        return 10
    names = os.listdir('somedir')
    names = [name for name in os.listdir('somedir') if os.path.isfile(os.path.join('somedir', name))]
    dirnames = [name for name in os.listdir('somedir') if os.path.isdir(os.path.join('somedir', name))]
    pyfiles = [name for name in os.listdir('somedir') if name.endswith('.py')]
    pyfiles = glob.glob('somedir/*.py')
    pyfiles = [name for name in os.listdir('somedir') if fnmatch(name, '*.py')]
    pyfiles = glob.glob('*.py')
    name_sz_date = [(name, os.path.getsize(name), os.path.getmtime(name)) for name in pyfiles]
    for (name, size, mtime) in name_sz_date:
        print(name, size, mtime)
    file_metadata = [(name, os.stat(name)) for name in pyfiles]
    for (name, meta) in file_metadata:
        print(name, meta.st_size, meta.st_mtime)
if __name__ == '__main__':
    dir_listfile()
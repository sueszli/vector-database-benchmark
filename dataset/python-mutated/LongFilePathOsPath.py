import os
from Common.LongFilePathSupport import LongFilePath

def isfile(path):
    if False:
        while True:
            i = 10
    return os.path.isfile(LongFilePath(path))

def isdir(path):
    if False:
        for i in range(10):
            print('nop')
    return os.path.isdir(LongFilePath(path))

def exists(path):
    if False:
        print('Hello World!')
    return os.path.exists(LongFilePath(path))

def getsize(filename):
    if False:
        for i in range(10):
            print('nop')
    return os.path.getsize(LongFilePath(filename))

def getmtime(filename):
    if False:
        i = 10
        return i + 15
    return os.path.getmtime(LongFilePath(filename))

def getatime(filename):
    if False:
        i = 10
        return i + 15
    return os.path.getatime(LongFilePath(filename))

def getctime(filename):
    if False:
        print('Hello World!')
    return os.path.getctime(LongFilePath(filename))
join = os.path.join
splitext = os.path.splitext
splitdrive = os.path.splitdrive
split = os.path.split
abspath = os.path.abspath
basename = os.path.basename
commonprefix = os.path.commonprefix
sep = os.path.sep
normpath = os.path.normpath
normcase = os.path.normcase
dirname = os.path.dirname
islink = os.path.islink
isabs = os.path.isabs
realpath = os.path.realpath
relpath = os.path.relpath
pardir = os.path.pardir
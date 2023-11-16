from __future__ import absolute_import
import os
from . import LongFilePathOsPath
from Common.LongFilePathSupport import LongFilePath
import time
path = LongFilePathOsPath

def access(path, mode):
    if False:
        i = 10
        return i + 15
    return os.access(LongFilePath(path), mode)

def remove(path):
    if False:
        i = 10
        return i + 15
    Timeout = 0.0
    while Timeout < 5.0:
        try:
            return os.remove(LongFilePath(path))
        except:
            time.sleep(0.1)
            Timeout = Timeout + 0.1
    return os.remove(LongFilePath(path))

def removedirs(name):
    if False:
        for i in range(10):
            print('nop')
    return os.removedirs(LongFilePath(name))

def rmdir(path):
    if False:
        return 10
    return os.rmdir(LongFilePath(path))

def mkdir(path):
    if False:
        return 10
    return os.mkdir(LongFilePath(path))

def makedirs(name, mode=511):
    if False:
        print('Hello World!')
    return os.makedirs(LongFilePath(name), mode)

def rename(old, new):
    if False:
        while True:
            i = 10
    return os.rename(LongFilePath(old), LongFilePath(new))

def chdir(path):
    if False:
        print('Hello World!')
    return os.chdir(LongFilePath(path))

def chmod(path, mode):
    if False:
        while True:
            i = 10
    return os.chmod(LongFilePath(path), mode)

def stat(path):
    if False:
        print('Hello World!')
    return os.stat(LongFilePath(path))

def utime(path, times):
    if False:
        return 10
    return os.utime(LongFilePath(path), times)

def listdir(path):
    if False:
        print('Hello World!')
    List = []
    uList = os.listdir(u'%s' % LongFilePath(path))
    for Item in uList:
        List.append(Item)
    return List
if hasattr(os, 'replace'):

    def replace(src, dst):
        if False:
            print('Hello World!')
        return os.replace(LongFilePath(src), LongFilePath(dst))
environ = os.environ
getcwd = os.getcwd
chdir = os.chdir
walk = os.walk
W_OK = os.W_OK
F_OK = os.F_OK
sep = os.sep
linesep = os.linesep
getenv = os.getenv
pathsep = os.pathsep
name = os.name
SEEK_SET = os.SEEK_SET
SEEK_END = os.SEEK_END
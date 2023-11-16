import os
import platform
import shutil
import codecs

def LongFilePath(FileName):
    if False:
        i = 10
        return i + 15
    FileName = os.path.normpath(FileName)
    if platform.system() == 'Windows':
        if FileName.startswith('\\\\?\\'):
            return FileName
        if FileName.startswith('\\\\'):
            return '\\\\?\\UNC\\' + FileName[2:]
        if os.path.isabs(FileName):
            return '\\\\?\\' + FileName
    return FileName

def OpenLongFilePath(FileName, Mode='r', Buffer=-1):
    if False:
        for i in range(10):
            print('nop')
    return open(LongFilePath(FileName), Mode, Buffer)

def CodecOpenLongFilePath(Filename, Mode='rb', Encoding=None, Errors='strict', Buffering=1):
    if False:
        for i in range(10):
            print('nop')
    return codecs.open(LongFilePath(Filename), Mode, Encoding, Errors, Buffering)

def CopyLongFilePath(src, dst):
    if False:
        while True:
            i = 10
    with open(LongFilePath(src), 'rb') as fsrc:
        with open(LongFilePath(dst), 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
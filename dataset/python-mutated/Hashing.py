""" Module for working with hashes in Nuitka.

Offers support for hashing incrementally and files esp. without having
to read their contents.
"""
from binascii import crc32
from nuitka.__past__ import md5, unicode
from .FileOperations import openTextFile

class HashBase(object):
    __slots__ = ('hash',)

    def updateFromValues(self, *values):
        if False:
            for i in range(10):
                print('nop')
        for value in values:
            if type(value) is int:
                value = str(int)
            if type(value) in (str, unicode):
                if str is not bytes:
                    value = value.encode('utf8')
                self.updateFromBytes(value)
            elif type(value) is bytes:
                self.updateFromBytes(value)
            else:
                assert False, type(value)

    def updateFromFile(self, filename):
        if False:
            return 10
        with openTextFile(filename, 'rb') as input_file:
            self.updateFromFileHandle(input_file)

    def updateFromFileHandle(self, file_handle):
        if False:
            while True:
                i = 10
        while 1:
            chunk = file_handle.read(1024 * 64)
            if not chunk:
                break
            self.updateFromBytes(chunk)

class Hash(HashBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.hash = md5()

    def updateFromBytes(self, value):
        if False:
            print('Hello World!')
        self.hash.update(value)

    def asDigest(self):
        if False:
            while True:
                i = 10
        return self.hash.digest()

    def asHexDigest(self):
        if False:
            return 10
        return self.hash.hexdigest()

def getFileContentsHash(filename, as_string=True):
    if False:
        while True:
            i = 10
    result = Hash()
    result.updateFromFile(filename=filename)
    if as_string:
        return result.asHexDigest()
    else:
        return result.asDigest()

def getStringHash(value):
    if False:
        print('Hello World!')
    result = Hash()
    result.updateFromValues(value)
    return result.asHexDigest()

def getHashFromValues(*values):
    if False:
        print('Hello World!')
    result = Hash()
    result.updateFromValues(*values)
    return result.asHexDigest()

class HashCRC32(HashBase):

    def __init__(self):
        if False:
            return 10
        self.hash = 0

    def updateFromBytes(self, value):
        if False:
            while True:
                i = 10
        self.hash = crc32(value, self.hash)

    def asDigest(self):
        if False:
            print('Hello World!')
        return self.hash
import hashlib
import os
import base64

def sha512sum(file, blocksize=65536, format='hexdigest'):
    if False:
        print('Hello World!')
    if type(file) is str:
        file = open(file, 'rb')
    hash = hashlib.sha512()
    for block in iter(lambda : file.read(blocksize), b''):
        hash.update(block)
    if format == 'hexdigest':
        return hash.hexdigest()[0:64]
    else:
        return hash.digest()[0:32]

def sha256sum(file, blocksize=65536):
    if False:
        print('Hello World!')
    if type(file) is str:
        file = open(file, 'rb')
    hash = hashlib.sha256()
    for block in iter(lambda : file.read(blocksize), b''):
        hash.update(block)
    return hash.hexdigest()

def random(length=64, encoding='hex'):
    if False:
        for i in range(10):
            print('nop')
    if encoding == 'base64':
        hash = hashlib.sha512(os.urandom(256)).digest()
        return base64.b64encode(hash).decode('ascii').replace('+', '').replace('/', '').replace('=', '')[0:length]
    else:
        return hashlib.sha512(os.urandom(256)).hexdigest()[0:length]

class Sha512t:

    def __init__(self, data):
        if False:
            return 10
        if data:
            self.sha512 = hashlib.sha512(data)
        else:
            self.sha512 = hashlib.sha512()

    def hexdigest(self):
        if False:
            while True:
                i = 10
        return self.sha512.hexdigest()[0:64]

    def digest(self):
        if False:
            return 10
        return self.sha512.digest()[0:32]

    def update(self, data):
        if False:
            for i in range(10):
                print('nop')
        return self.sha512.update(data)

def sha512t(data=None):
    if False:
        for i in range(10):
            print('nop')
    return Sha512t(data)
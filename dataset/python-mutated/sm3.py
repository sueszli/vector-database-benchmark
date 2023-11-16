from gmssl import sm3, func
from django.contrib.auth.hashers import PBKDF2PasswordHasher

class Hasher:
    name = 'sm3'
    block_size = 64
    digest_size = 32

    def __init__(self, data):
        if False:
            return 10
        self.__data = data

    def hexdigest(self):
        if False:
            for i in range(10):
                print('nop')
        return sm3.sm3_hash(func.bytes_to_list(self.__data))

    def digest(self):
        if False:
            while True:
                i = 10
        return bytes.fromhex(self.hexdigest())

    @staticmethod
    def hash(msg=b''):
        if False:
            for i in range(10):
                print('nop')
        return Hasher(msg)

    def update(self, data):
        if False:
            print('Hello World!')
        self.__data += data

    def copy(self):
        if False:
            print('Hello World!')
        return Hasher(self.__data)

class PBKDF2SM3PasswordHasher(PBKDF2PasswordHasher):
    algorithm = 'pbkdf2_sm3'
    digest = Hasher.hash
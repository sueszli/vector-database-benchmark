import hashlib
import base64

class SimpleHash(object):
    """ Hash methods wrapper meta-class """

    @classmethod
    def base64_encode(cls, data):
        if False:
            return 10
        ' Encode string to base64\n        :param str data: binary string to be encoded\n        :return str: base64-encoded string\n        '
        return base64.encodestring(data)

    @classmethod
    def base64_decode(cls, data):
        if False:
            i = 10
            return i + 15
        ' Decode base64 string\n        :param str data: base64-encoded string to be decoded\n        :return str: binary string\n        '
        return base64.decodestring(data)

    @classmethod
    def hash(cls, data):
        if False:
            while True:
                i = 10
        ' Return sha1 of data (digest)\n        :param str data: string to be hashed\n        :return str: digest sha1 of data\n        '
        sha = hashlib.sha1(data)
        return sha.digest()

    @classmethod
    def hash_hex(cls, data):
        if False:
            i = 10
            return i + 15
        ' Return sha1 of data (hexdigest)\n        :param str data: string to be hashed\n        :return str: hexdigest sha1 of data\n        '
        sha = hashlib.sha1(data)
        return sha.hexdigest()

    @classmethod
    def hash_base64(cls, data):
        if False:
            for i in range(10):
                print('nop')
        ' Return sha1 of data encoded with base64\n        :param str data: data to be hashed and encoded\n        :return str: base64 encoded sha1 of data\n        '
        return cls.base64_encode(cls.hash(data))

    @classmethod
    def hash_file(cls, filename, block_size=2 ** 20):
        if False:
            i = 10
            return i + 15
        'Return sha1 of data from given file\n        :param str filename: name of a file that should be read\n        :param int block_size: *Default: 2**20* data will be read from file in\n        chunks of this size\n        :return bytes: bytes of data from file <filename>\n        '
        with open(filename, 'rb') as f:
            sha = hashlib.sha1()
            while True:
                data = f.read(block_size)
                if not data:
                    break
                sha.update(data)
            return sha.digest()

    @classmethod
    def hash_file_base64(cls, filename, block_size=2 ** 20):
        if False:
            for i in range(10):
                print('nop')
        'Return sha1 of data from given file encoded with base64\n        :param str filename: name of a file that should be read\n        :param int block_size: *Default: 2**20* data will be read from file in\n        chunks of this size\n        :return str: base64 encoded sha1 of data from file <filename>\n        '
        return cls.base64_encode(cls.hash_file(filename, block_size))

    @classmethod
    def hash_object(cls):
        if False:
            for i in range(10):
                print('nop')
        return hashlib.sha1()
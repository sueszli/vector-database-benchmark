import random

class Codec(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__random_length = 6
        self.__tiny_url = 'http://tinyurl.com/'
        self.__alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.__lookup = {}

    def encode(self, longUrl):
        if False:
            i = 10
            return i + 15
        'Encodes a URL to a shortened URL.\n\n        :type longUrl: str\n        :rtype: str\n        '

        def getRand():
            if False:
                i = 10
                return i + 15
            rand = []
            for _ in xrange(self.__random_length):
                rand += self.__alphabet[random.randint(0, len(self.__alphabet) - 1)]
            return ''.join(rand)
        key = getRand()
        while key in self.__lookup:
            key = getRand()
        self.__lookup[key] = longUrl
        return self.__tiny_url + key

    def decode(self, shortUrl):
        if False:
            i = 10
            return i + 15
        'Decodes a shortened URL to its original URL.\n\n        :type shortUrl: str\n        :rtype: str\n        '
        return self.__lookup[shortUrl[len(self.__tiny_url):]]
from hashlib import sha256

class Codec2(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self._cache = {}
        self.url = 'http://tinyurl.com/'

    def encode(self, long_url):
        if False:
            i = 10
            return i + 15
        'Encodes a URL to a shortened URL.\n\n        :type long_url: str\n        :rtype: str\n        '
        key = sha256(long_url.encode()).hexdigest()[:6]
        self._cache[key] = long_url
        return self.url + key

    def decode(self, short_url):
        if False:
            while True:
                i = 10
        'Decodes a shortened URL to its original URL.\n\n        :type short_url: str\n        :rtype: str\n        '
        key = short_url.replace(self.url, '')
        return self._cache[key]
import collections
import itertools

class Encrypter(object):

    def __init__(self, keys, values, dictionary):
        if False:
            i = 10
            return i + 15
        '\n        :type keys: List[str]\n        :type values: List[str]\n        :type dictionary: List[str]\n        '
        self.__lookup = {k: v for (k, v) in itertools.izip(keys, values)}
        self.__cnt = collections.Counter((self.encrypt(x) for x in dictionary))

    def encrypt(self, word1):
        if False:
            i = 10
            return i + 15
        '\n        :type word1: str\n        :rtype: str\n        '
        if any((c not in self.__lookup for c in word1)):
            return ''
        return ''.join((self.__lookup[c] for c in word1))

    def decrypt(self, word2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type word2: str\n        :rtype: int\n        '
        return self.__cnt[word2]
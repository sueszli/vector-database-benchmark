import collections

class AuthenticationManager(object):

    def __init__(self, timeToLive):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type timeToLive: int\n        '
        self.__time = timeToLive
        self.__lookup = collections.OrderedDict()

    def __evict(self, currentTime):
        if False:
            print('Hello World!')
        while self.__lookup and next(self.__lookup.itervalues()) <= currentTime:
            self.__lookup.popitem(last=False)

    def generate(self, tokenId, currentTime):
        if False:
            i = 10
            return i + 15
        '\n        :type tokenId: str\n        :type currentTime: int\n        :rtype: None\n        '
        self.__evict(currentTime)
        self.__lookup[tokenId] = currentTime + self.__time

    def renew(self, tokenId, currentTime):
        if False:
            while True:
                i = 10
        '\n        :type tokenId: str\n        :type currentTime: int\n        :rtype: None\n        '
        self.__evict(currentTime)
        if tokenId not in self.__lookup:
            return
        del self.__lookup[tokenId]
        self.__lookup[tokenId] = currentTime + self.__time

    def countUnexpiredTokens(self, currentTime):
        if False:
            print('Hello World!')
        '\n        :type currentTime: int\n        :rtype: int\n        '
        self.__evict(currentTime)
        return len(self.__lookup)
import random

def rand7():
    if False:
        for i in range(10):
            print('nop')
    return random.randint(1, 7)

class Solution(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__cache = []

    def rand10(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: int\n        '

        def generate(cache):
            if False:
                while True:
                    i = 10
            n = 32
            curr = sum(((rand7() - 1) * 7 ** i for i in xrange(n)))
            rang = 7 ** n
            while curr < rang // 10 * 10:
                cache.append(curr % 10 + 1)
                curr /= 10
                rang /= 10
        while not self.__cache:
            generate(self.__cache)
        return self.__cache.pop()

class Solution2(object):

    def rand10(self):
        if False:
            print('Hello World!')
        '\n        :rtype: int\n        '
        while True:
            x = (rand7() - 1) * 7 + (rand7() - 1)
            if x < 40:
                return x % 10 + 1
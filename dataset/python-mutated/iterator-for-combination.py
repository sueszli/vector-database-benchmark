import itertools

class CombinationIterator(object):

    def __init__(self, characters, combinationLength):
        if False:
            i = 10
            return i + 15
        '\n        :type characters: str\n        :type combinationLength: int\n        '
        self.__it = itertools.combinations(characters, combinationLength)
        self.__curr = None
        self.__last = characters[-combinationLength:]

    def next(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: str\n        '
        self.__curr = ''.join(self.__it.next())
        return self.__curr

    def hasNext(self):
        if False:
            return 10
        '\n        :rtype: bool\n        '
        return self.__curr != self.__last
import functools

class CombinationIterator2(object):

    def __init__(self, characters, combinationLength):
        if False:
            while True:
                i = 10
        '\n        :type characters: str\n        :type combinationLength: int\n        '
        self.__characters = characters
        self.__combinationLength = combinationLength
        self.__it = self.__iterative_backtracking()
        self.__curr = None
        self.__last = characters[-combinationLength:]

    def __iterative_backtracking(self):
        if False:
            i = 10
            return i + 15

        def conquer():
            if False:
                while True:
                    i = 10
            if len(curr) == self.__combinationLength:
                return curr

        def prev_divide(c):
            if False:
                print('Hello World!')
            curr.append(c)

        def divide(i):
            if False:
                for i in range(10):
                    print('nop')
            if len(curr) != self.__combinationLength:
                for j in reversed(xrange(i, len(self.__characters) - (self.__combinationLength - len(curr) - 1))):
                    stk.append(functools.partial(post_divide))
                    stk.append(functools.partial(divide, j + 1))
                    stk.append(functools.partial(prev_divide, self.__characters[j]))
            stk.append(functools.partial(conquer))

        def post_divide():
            if False:
                for i in range(10):
                    print('nop')
            curr.pop()
        curr = []
        stk = [functools.partial(divide, 0)]
        while stk:
            result = stk.pop()()
            if result is not None:
                yield result

    def next(self):
        if False:
            return 10
        '\n        :rtype: str\n        '
        self.__curr = ''.join(next(self.__it))
        return self.__curr

    def hasNext(self):
        if False:
            print('Hello World!')
        '\n        :rtype: bool\n        '
        return self.__curr != self.__last
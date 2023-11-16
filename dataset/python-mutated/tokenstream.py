import re
from ..core.inputscanner import InputScanner
from ..core.token import Token

class TokenStream:

    def __init__(self, parent_token=None):
        if False:
            while True:
                i = 10
        self.__tokens = []
        self.__tokens_length = len(self.__tokens)
        self.__position = 0
        self.__parent_token = parent_token

    def restart(self):
        if False:
            for i in range(10):
                print('nop')
        self.__position = 0

    def isEmpty(self):
        if False:
            return 10
        return self.__tokens_length == 0

    def hasNext(self):
        if False:
            return 10
        return self.__position < self.__tokens_length

    def next(self):
        if False:
            print('Hello World!')
        if self.hasNext():
            val = self.__tokens[self.__position]
            self.__position += 1
            return val
        else:
            raise StopIteration

    def peek(self, index=0):
        if False:
            for i in range(10):
                print('nop')
        val = None
        index += self.__position
        if index >= 0 and index < self.__tokens_length:
            val = self.__tokens[index]
        return val

    def add(self, token):
        if False:
            print('Hello World!')
        if self.__parent_token:
            token.parent = self.__parent_token
        self.__tokens.append(token)
        self.__tokens_length += 1

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.restart()
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        return self.next()
from antlr4.BufferedTokenStream import BufferedTokenStream
from antlr4.Lexer import Lexer
from antlr4.Token import Token

class CommonTokenStream(BufferedTokenStream):
    __slots__ = 'channel'

    def __init__(self, lexer: Lexer, channel: int=Token.DEFAULT_CHANNEL):
        if False:
            print('Hello World!')
        super().__init__(lexer)
        self.channel = channel

    def adjustSeekIndex(self, i: int):
        if False:
            while True:
                i = 10
        return self.nextTokenOnChannel(i, self.channel)

    def LB(self, k: int):
        if False:
            while True:
                i = 10
        if k == 0 or self.index - k < 0:
            return None
        i = self.index
        n = 1
        while n <= k:
            i = self.previousTokenOnChannel(i - 1, self.channel)
            n += 1
        if i < 0:
            return None
        return self.tokens[i]

    def LT(self, k: int):
        if False:
            return 10
        self.lazyInit()
        if k == 0:
            return None
        if k < 0:
            return self.LB(-k)
        i = self.index
        n = 1
        while n < k:
            if self.sync(i + 1):
                i = self.nextTokenOnChannel(i + 1, self.channel)
            n += 1
        return self.tokens[i]

    def getNumberOfOnChannelTokens(self):
        if False:
            print('Hello World!')
        n = 0
        self.fill()
        for i in range(0, len(self.tokens)):
            t = self.tokens[i]
            if t.channel == self.channel:
                n += 1
            if t.type == Token.EOF:
                break
        return n
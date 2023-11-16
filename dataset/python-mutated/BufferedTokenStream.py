from io import StringIO
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException
Lexer = None

class TokenStream(object):
    pass

class BufferedTokenStream(TokenStream):
    __slots__ = ('tokenSource', 'tokens', 'index', 'fetchedEOF')

    def __init__(self, tokenSource: Lexer):
        if False:
            print('Hello World!')
        self.tokenSource = tokenSource
        self.tokens = []
        self.index = -1
        self.fetchedEOF = False

    def mark(self):
        if False:
            print('Hello World!')
        return 0

    def release(self, marker: int):
        if False:
            return 10
        pass

    def reset(self):
        if False:
            print('Hello World!')
        self.seek(0)

    def seek(self, index: int):
        if False:
            while True:
                i = 10
        self.lazyInit()
        self.index = self.adjustSeekIndex(index)

    def get(self, index: int):
        if False:
            return 10
        self.lazyInit()
        return self.tokens[index]

    def consume(self):
        if False:
            while True:
                i = 10
        skipEofCheck = False
        if self.index >= 0:
            if self.fetchedEOF:
                skipEofCheck = self.index < len(self.tokens) - 1
            else:
                skipEofCheck = self.index < len(self.tokens)
        else:
            skipEofCheck = False
        if not skipEofCheck and self.LA(1) == Token.EOF:
            raise IllegalStateException('cannot consume EOF')
        if self.sync(self.index + 1):
            self.index = self.adjustSeekIndex(self.index + 1)

    def sync(self, i: int):
        if False:
            print('Hello World!')
        n = i - len(self.tokens) + 1
        if n > 0:
            fetched = self.fetch(n)
            return fetched >= n
        return True

    def fetch(self, n: int):
        if False:
            while True:
                i = 10
        if self.fetchedEOF:
            return 0
        for i in range(0, n):
            t = self.tokenSource.nextToken()
            t.tokenIndex = len(self.tokens)
            self.tokens.append(t)
            if t.type == Token.EOF:
                self.fetchedEOF = True
                return i + 1
        return n

    def getTokens(self, start: int, stop: int, types: set=None):
        if False:
            for i in range(10):
                print('nop')
        if start < 0 or stop < 0:
            return None
        self.lazyInit()
        subset = []
        if stop >= len(self.tokens):
            stop = len(self.tokens) - 1
        for i in range(start, stop):
            t = self.tokens[i]
            if t.type == Token.EOF:
                break
            if types is None or t.type in types:
                subset.append(t)
        return subset

    def LA(self, i: int):
        if False:
            return 10
        return self.LT(i).type

    def LB(self, k: int):
        if False:
            while True:
                i = 10
        if self.index - k < 0:
            return None
        return self.tokens[self.index - k]

    def LT(self, k: int):
        if False:
            return 10
        self.lazyInit()
        if k == 0:
            return None
        if k < 0:
            return self.LB(-k)
        i = self.index + k - 1
        self.sync(i)
        if i >= len(self.tokens):
            return self.tokens[len(self.tokens) - 1]
        return self.tokens[i]

    def adjustSeekIndex(self, i: int):
        if False:
            i = 10
            return i + 15
        return i

    def lazyInit(self):
        if False:
            while True:
                i = 10
        if self.index == -1:
            self.setup()

    def setup(self):
        if False:
            while True:
                i = 10
        self.sync(0)
        self.index = self.adjustSeekIndex(0)

    def setTokenSource(self, tokenSource: Lexer):
        if False:
            print('Hello World!')
        self.tokenSource = tokenSource
        self.tokens = []
        self.index = -1
        self.fetchedEOF = False

    def nextTokenOnChannel(self, i: int, channel: int):
        if False:
            while True:
                i = 10
        self.sync(i)
        if i >= len(self.tokens):
            return len(self.tokens) - 1
        token = self.tokens[i]
        while token.channel != channel:
            if token.type == Token.EOF:
                return i
            i += 1
            self.sync(i)
            token = self.tokens[i]
        return i

    def previousTokenOnChannel(self, i: int, channel: int):
        if False:
            for i in range(10):
                print('nop')
        while i >= 0 and self.tokens[i].channel != channel:
            i -= 1
        return i

    def getHiddenTokensToRight(self, tokenIndex: int, channel: int=-1):
        if False:
            while True:
                i = 10
        self.lazyInit()
        if tokenIndex < 0 or tokenIndex >= len(self.tokens):
            raise Exception(str(tokenIndex) + ' not in 0..' + str(len(self.tokens) - 1))
        from antlr4.Lexer import Lexer
        nextOnChannel = self.nextTokenOnChannel(tokenIndex + 1, Lexer.DEFAULT_TOKEN_CHANNEL)
        from_ = tokenIndex + 1
        to = len(self.tokens) - 1 if nextOnChannel == -1 else nextOnChannel
        return self.filterForChannel(from_, to, channel)

    def getHiddenTokensToLeft(self, tokenIndex: int, channel: int=-1):
        if False:
            print('Hello World!')
        self.lazyInit()
        if tokenIndex < 0 or tokenIndex >= len(self.tokens):
            raise Exception(str(tokenIndex) + ' not in 0..' + str(len(self.tokens) - 1))
        from antlr4.Lexer import Lexer
        prevOnChannel = self.previousTokenOnChannel(tokenIndex - 1, Lexer.DEFAULT_TOKEN_CHANNEL)
        if prevOnChannel == tokenIndex - 1:
            return None
        from_ = prevOnChannel + 1
        to = tokenIndex - 1
        return self.filterForChannel(from_, to, channel)

    def filterForChannel(self, left: int, right: int, channel: int):
        if False:
            while True:
                i = 10
        hidden = []
        for i in range(left, right + 1):
            t = self.tokens[i]
            if channel == -1:
                from antlr4.Lexer import Lexer
                if t.channel != Lexer.DEFAULT_TOKEN_CHANNEL:
                    hidden.append(t)
            elif t.channel == channel:
                hidden.append(t)
        if len(hidden) == 0:
            return None
        return hidden

    def getSourceName(self):
        if False:
            print('Hello World!')
        return self.tokenSource.getSourceName()

    def getText(self, start: int=None, stop: int=None):
        if False:
            return 10
        self.lazyInit()
        self.fill()
        if isinstance(start, Token):
            start = start.tokenIndex
        elif start is None:
            start = 0
        if isinstance(stop, Token):
            stop = stop.tokenIndex
        elif stop is None or stop >= len(self.tokens):
            stop = len(self.tokens) - 1
        if start < 0 or stop < 0 or stop < start:
            return ''
        with StringIO() as buf:
            for i in range(start, stop + 1):
                t = self.tokens[i]
                if t.type == Token.EOF:
                    break
                buf.write(t.text)
            return buf.getvalue()

    def fill(self):
        if False:
            print('Hello World!')
        self.lazyInit()
        while self.fetch(1000) == 1000:
            pass
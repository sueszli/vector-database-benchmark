from antlr4.CommonTokenFactory import CommonTokenFactory
from antlr4.Lexer import TokenSource
from antlr4.Token import Token

class ListTokenSource(TokenSource):
    __slots__ = ('tokens', 'sourceName', 'pos', 'eofToken', '_factory')

    def __init__(self, tokens: list, sourceName: str=None):
        if False:
            for i in range(10):
                print('nop')
        if tokens is None:
            raise ReferenceError('tokens cannot be null')
        self.tokens = tokens
        self.sourceName = sourceName
        self.pos = 0
        self.eofToken = None
        self._factory = CommonTokenFactory.DEFAULT

    @property
    def column(self):
        if False:
            while True:
                i = 10
        if self.pos < len(self.tokens):
            return self.tokens[self.pos].column
        elif self.eofToken is not None:
            return self.eofToken.column
        elif len(self.tokens) > 0:
            lastToken = self.tokens[len(self.tokens) - 1]
            tokenText = lastToken.text
            if tokenText is not None:
                lastNewLine = tokenText.rfind('\n')
                if lastNewLine >= 0:
                    return len(tokenText) - lastNewLine - 1
            return lastToken.column + lastToken.stop - lastToken.start + 1
        return 0

    def nextToken(self):
        if False:
            return 10
        if self.pos >= len(self.tokens):
            if self.eofToken is None:
                start = -1
                if len(self.tokens) > 0:
                    previousStop = self.tokens[len(self.tokens) - 1].stop
                    if previousStop != -1:
                        start = previousStop + 1
                stop = max(-1, start - 1)
                self.eofToken = self._factory.create((self, self.getInputStream()), Token.EOF, 'EOF', Token.DEFAULT_CHANNEL, start, stop, self.line, self.column)
            return self.eofToken
        t = self.tokens[self.pos]
        if self.pos == len(self.tokens) - 1 and t.type == Token.EOF:
            self.eofToken = t
        self.pos += 1
        return t

    @property
    def line(self):
        if False:
            i = 10
            return i + 15
        if self.pos < len(self.tokens):
            return self.tokens[self.pos].line
        elif self.eofToken is not None:
            return self.eofToken.line
        elif len(self.tokens) > 0:
            lastToken = self.tokens[len(self.tokens) - 1]
            line = lastToken.line
            tokenText = lastToken.text
            if tokenText is not None:
                line += tokenText.count('\n')
            return line
        return 1

    def getInputStream(self):
        if False:
            return 10
        if self.pos < len(self.tokens):
            return self.tokens[self.pos].getInputStream()
        elif self.eofToken is not None:
            return self.eofToken.getInputStream()
        elif len(self.tokens) > 0:
            return self.tokens[len(self.tokens) - 1].getInputStream()
        else:
            return None

    def getSourceName(self):
        if False:
            for i in range(10):
                print('nop')
        if self.sourceName is not None:
            return self.sourceName
        inputStream = self.getInputStream()
        if inputStream is not None:
            return inputStream.getSourceName()
        else:
            return 'List'
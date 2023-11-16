from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO
from antlr4.CommonTokenFactory import CommonTokenFactory
from antlr4.atn.LexerATNSimulator import LexerATNSimulator
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.Token import Token
from antlr4.error.Errors import IllegalStateException, LexerNoViableAltException, RecognitionException

class TokenSource(object):
    pass

class Lexer(Recognizer, TokenSource):
    __slots__ = ('_input', '_output', '_factory', '_tokenFactorySourcePair', '_token', '_tokenStartCharIndex', '_tokenStartLine', '_tokenStartColumn', '_hitEOF', '_channel', '_type', '_modeStack', '_mode', '_text')
    DEFAULT_MODE = 0
    MORE = -2
    SKIP = -3
    DEFAULT_TOKEN_CHANNEL = Token.DEFAULT_CHANNEL
    HIDDEN = Token.HIDDEN_CHANNEL
    MIN_CHAR_VALUE = 0
    MAX_CHAR_VALUE = 1114111

    def __init__(self, input: InputStream, output: TextIO=sys.stdout):
        if False:
            return 10
        super().__init__()
        self._input = input
        self._output = output
        self._factory = CommonTokenFactory.DEFAULT
        self._tokenFactorySourcePair = (self, input)
        self._interp = None
        self._token = None
        self._tokenStartCharIndex = -1
        self._tokenStartLine = -1
        self._tokenStartColumn = -1
        self._hitEOF = False
        self._channel = Token.DEFAULT_CHANNEL
        self._type = Token.INVALID_TYPE
        self._modeStack = []
        self._mode = self.DEFAULT_MODE
        self._text = None

    def reset(self):
        if False:
            i = 10
            return i + 15
        if self._input is not None:
            self._input.seek(0)
        self._token = None
        self._type = Token.INVALID_TYPE
        self._channel = Token.DEFAULT_CHANNEL
        self._tokenStartCharIndex = -1
        self._tokenStartColumn = -1
        self._tokenStartLine = -1
        self._text = None
        self._hitEOF = False
        self._mode = Lexer.DEFAULT_MODE
        self._modeStack = []
        self._interp.reset()

    def nextToken(self):
        if False:
            while True:
                i = 10
        if self._input is None:
            raise IllegalStateException('nextToken requires a non-null input stream.')
        tokenStartMarker = self._input.mark()
        try:
            while True:
                if self._hitEOF:
                    self.emitEOF()
                    return self._token
                self._token = None
                self._channel = Token.DEFAULT_CHANNEL
                self._tokenStartCharIndex = self._input.index
                self._tokenStartColumn = self._interp.column
                self._tokenStartLine = self._interp.line
                self._text = None
                continueOuter = False
                while True:
                    self._type = Token.INVALID_TYPE
                    ttype = self.SKIP
                    try:
                        ttype = self._interp.match(self._input, self._mode)
                    except LexerNoViableAltException as e:
                        self.notifyListeners(e)
                        self.recover(e)
                    if self._input.LA(1) == Token.EOF:
                        self._hitEOF = True
                    if self._type == Token.INVALID_TYPE:
                        self._type = ttype
                    if self._type == self.SKIP:
                        continueOuter = True
                        break
                    if self._type != self.MORE:
                        break
                if continueOuter:
                    continue
                if self._token is None:
                    self.emit()
                return self._token
        finally:
            self._input.release(tokenStartMarker)

    def skip(self):
        if False:
            while True:
                i = 10
        self._type = self.SKIP

    def more(self):
        if False:
            print('Hello World!')
        self._type = self.MORE

    def mode(self, m: int):
        if False:
            return 10
        self._mode = m

    def pushMode(self, m: int):
        if False:
            print('Hello World!')
        if self._interp.debug:
            print('pushMode ' + str(m), file=self._output)
        self._modeStack.append(self._mode)
        self.mode(m)

    def popMode(self):
        if False:
            print('Hello World!')
        if len(self._modeStack) == 0:
            raise Exception('Empty Stack')
        if self._interp.debug:
            print('popMode back to ' + self._modeStack[:-1], file=self._output)
        self.mode(self._modeStack.pop())
        return self._mode

    @property
    def inputStream(self):
        if False:
            while True:
                i = 10
        return self._input

    @inputStream.setter
    def inputStream(self, input: InputStream):
        if False:
            i = 10
            return i + 15
        self._input = None
        self._tokenFactorySourcePair = (self, self._input)
        self.reset()
        self._input = input
        self._tokenFactorySourcePair = (self, self._input)

    @property
    def sourceName(self):
        if False:
            print('Hello World!')
        return self._input.sourceName

    def emitToken(self, token: Token):
        if False:
            while True:
                i = 10
        self._token = token

    def emit(self):
        if False:
            print('Hello World!')
        t = self._factory.create(self._tokenFactorySourcePair, self._type, self._text, self._channel, self._tokenStartCharIndex, self.getCharIndex() - 1, self._tokenStartLine, self._tokenStartColumn)
        self.emitToken(t)
        return t

    def emitEOF(self):
        if False:
            while True:
                i = 10
        cpos = self.column
        lpos = self.line
        eof = self._factory.create(self._tokenFactorySourcePair, Token.EOF, None, Token.DEFAULT_CHANNEL, self._input.index, self._input.index - 1, lpos, cpos)
        self.emitToken(eof)
        return eof

    @property
    def type(self):
        if False:
            while True:
                i = 10
        return self._type

    @type.setter
    def type(self, type: int):
        if False:
            print('Hello World!')
        self._type = type

    @property
    def line(self):
        if False:
            print('Hello World!')
        return self._interp.line

    @line.setter
    def line(self, line: int):
        if False:
            return 10
        self._interp.line = line

    @property
    def column(self):
        if False:
            while True:
                i = 10
        return self._interp.column

    @column.setter
    def column(self, column: int):
        if False:
            while True:
                i = 10
        self._interp.column = column

    def getCharIndex(self):
        if False:
            while True:
                i = 10
        return self._input.index

    @property
    def text(self):
        if False:
            for i in range(10):
                print('nop')
        if self._text is not None:
            return self._text
        else:
            return self._interp.getText(self._input)

    @text.setter
    def text(self, txt: str):
        if False:
            for i in range(10):
                print('nop')
        self._text = txt

    def getAllTokens(self):
        if False:
            return 10
        tokens = []
        t = self.nextToken()
        while t.type != Token.EOF:
            tokens.append(t)
            t = self.nextToken()
        return tokens

    def notifyListeners(self, e: LexerNoViableAltException):
        if False:
            print('Hello World!')
        start = self._tokenStartCharIndex
        stop = self._input.index
        text = self._input.getText(start, stop)
        msg = "token recognition error at: '" + self.getErrorDisplay(text) + "'"
        listener = self.getErrorListenerDispatch()
        listener.syntaxError(self, None, self._tokenStartLine, self._tokenStartColumn, msg, e)

    def getErrorDisplay(self, s: str):
        if False:
            while True:
                i = 10
        with StringIO() as buf:
            for c in s:
                buf.write(self.getErrorDisplayForChar(c))
            return buf.getvalue()

    def getErrorDisplayForChar(self, c: str):
        if False:
            i = 10
            return i + 15
        if ord(c[0]) == Token.EOF:
            return '<EOF>'
        elif c == '\n':
            return '\\n'
        elif c == '\t':
            return '\\t'
        elif c == '\r':
            return '\\r'
        else:
            return c

    def getCharErrorDisplay(self, c: str):
        if False:
            for i in range(10):
                print('nop')
        return "'" + self.getErrorDisplayForChar(c) + "'"

    def recover(self, re: RecognitionException):
        if False:
            print('Hello World!')
        if self._input.LA(1) != Token.EOF:
            if isinstance(re, LexerNoViableAltException):
                self._interp.consume(self._input)
            else:
                self._input.consume()
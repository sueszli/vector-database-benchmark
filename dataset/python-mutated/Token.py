from io import StringIO

class Token(object):
    __slots__ = ('source', 'type', 'channel', 'start', 'stop', 'tokenIndex', 'line', 'column', '_text')
    INVALID_TYPE = 0
    EPSILON = -2
    MIN_USER_TOKEN_TYPE = 1
    EOF = -1
    DEFAULT_CHANNEL = 0
    HIDDEN_CHANNEL = 1

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.source = None
        self.type = None
        self.channel = None
        self.start = None
        self.stop = None
        self.tokenIndex = None
        self.line = None
        self.column = None
        self._text = None

    @property
    def text(self):
        if False:
            while True:
                i = 10
        return self._text

    @text.setter
    def text(self, text: str):
        if False:
            while True:
                i = 10
        self._text = text

    def getTokenSource(self):
        if False:
            return 10
        return self.source[0]

    def getInputStream(self):
        if False:
            while True:
                i = 10
        return self.source[1]

class CommonToken(Token):
    EMPTY_SOURCE = (None, None)

    def __init__(self, source: tuple=EMPTY_SOURCE, type: int=None, channel: int=Token.DEFAULT_CHANNEL, start: int=-1, stop: int=-1):
        if False:
            while True:
                i = 10
        super().__init__()
        self.source = source
        self.type = type
        self.channel = channel
        self.start = start
        self.stop = stop
        self.tokenIndex = -1
        if source[0] is not None:
            self.line = source[0].line
            self.column = source[0].column
        else:
            self.column = -1

    def clone(self):
        if False:
            i = 10
            return i + 15
        t = CommonToken(self.source, self.type, self.channel, self.start, self.stop)
        t.tokenIndex = self.tokenIndex
        t.line = self.line
        t.column = self.column
        t.text = self.text
        return t

    @property
    def text(self):
        if False:
            for i in range(10):
                print('nop')
        if self._text is not None:
            return self._text
        input = self.getInputStream()
        if input is None:
            return None
        n = input.size
        if self.start < n and self.stop < n:
            return input.getText(self.start, self.stop)
        else:
            return '<EOF>'

    @text.setter
    def text(self, text: str):
        if False:
            return 10
        self._text = text

    def __str__(self):
        if False:
            print('Hello World!')
        with StringIO() as buf:
            buf.write('[@')
            buf.write(str(self.tokenIndex))
            buf.write(',')
            buf.write(str(self.start))
            buf.write(':')
            buf.write(str(self.stop))
            buf.write("='")
            txt = self.text
            if txt is not None:
                txt = txt.replace('\n', '\\n')
                txt = txt.replace('\r', '\\r')
                txt = txt.replace('\t', '\\t')
            else:
                txt = '<no text>'
            buf.write(txt)
            buf.write("',<")
            buf.write(str(self.type))
            buf.write('>')
            if self.channel > 0:
                buf.write(',channel=')
                buf.write(str(self.channel))
            buf.write(',')
            buf.write(str(self.line))
            buf.write(':')
            buf.write(str(self.column))
            buf.write(']')
            return buf.getvalue()
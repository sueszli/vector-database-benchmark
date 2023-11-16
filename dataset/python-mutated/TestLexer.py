from __future__ import print_function
from antlr4 import *
from io import StringIO

def serializedATN():
    if False:
        print('Hello World!')
    with StringIO() as buf:
        buf.write(u'\x03а훑舆괭䐗껱趀ꫝ\x02')
        buf.write(u'\x05\x0f\x08\x01\x04\x02\t\x02\x04\x03\t\x03\x04\x04\t\x04\x03\x02\x03\x02\x03\x03\x03\x03\x03\x04')
        buf.write(u'\x03\x04\x02\x02\x05\x03\x03\x05\x04\x07\x05\x03\x02\x02\x0e\x02\x03\x03\x02\x02\x02\x02\x05\x03\x02\x02')
        buf.write(u'\x02\x02\x07\x03\x02\x02\x02\x03\t\x03\x02\x02\x02\x05\x0b\x03\x02\x02\x02\x07\r\x03\x02\x02\x02\t')
        buf.write(u'\n\x07c\x02\x02\n\x04\x03\x02\x02\x02\x0b\x0c\x07d\x02\x02\x0c\x06\x03\x02\x02\x02\r\x0e\x07')
        buf.write(u'e\x02\x02\x0e\x08\x03\x02\x02\x02\x03\x02\x02')
        return buf.getvalue()

class TestLexer(Lexer):
    atn = ATNDeserializer().deserialize(serializedATN())
    decisionsToDFA = [DFA(ds, i) for (i, ds) in enumerate(atn.decisionToState)]
    A = 1
    B = 2
    C = 3
    modeNames = [u'DEFAULT_MODE']
    literalNames = [u'<INVALID>', u"'a'", u"'b'", u"'c'"]
    symbolicNames = [u'<INVALID>', u'A', u'B', u'C']
    ruleNames = [u'A', u'B', u'C']
    grammarFileName = u'T.g4'

    def __init__(self, input=None):
        if False:
            i = 10
            return i + 15
        super(TestLexer, self).__init__(input)
        self.checkVersion('4.9')
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None

def serializedATN2():
    if False:
        for i in range(10):
            print('nop')
    with StringIO() as buf:
        buf.write(u'\x03а훑舆괭䐗껱趀ꫝ\x02')
        buf.write(u'\t(\x08\x01\x04\x02\t\x02\x04\x03\t\x03\x04\x04\t\x04\x04\x05\t\x05\x04\x06\t\x06\x04\x07\t')
        buf.write(u'\x07\x04\x08\t\x08\x03\x02\x06\x02\x13\n\x02\r\x02\x0e\x02\x14\x03\x03\x06\x03\x18\n\x03')
        buf.write(u'\r\x03\x0e\x03\x19\x03\x04\x03\x04\x03\x05\x03\x05\x03\x06\x03\x06\x03\x07\x03\x07\x03\x08\x06\x08')
        buf.write(u'%\n\x08\r\x08\x0e\x08&\x02\x02\t\x03\x03\x05\x04\x07\x05\t\x06\x0b\x07\r\x08\x0f\t\x03')
        buf.write(u'\x02\x02*\x02\x03\x03\x02\x02\x02\x02\x05\x03\x02\x02\x02\x02\x07\x03\x02\x02\x02\x02\t\x03\x02\x02\x02')
        buf.write(u'\x02\x0b\x03\x02\x02\x02\x02\r\x03\x02\x02\x02\x02\x0f\x03\x02\x02\x02\x03\x12\x03\x02\x02\x02\x05')
        buf.write(u'\x17\x03\x02\x02\x02\x07\x1b\x03\x02\x02\x02\t\x1d\x03\x02\x02\x02\x0b\x1f\x03\x02\x02\x02\r')
        buf.write(u'!\x03\x02\x02\x02\x0f$\x03\x02\x02\x02\x11\x13\x04c|\x02\x12\x11\x03\x02\x02\x02\x13\x14')
        buf.write(u'\x03\x02\x02\x02\x14\x12\x03\x02\x02\x02\x14\x15\x03\x02\x02\x02\x15\x04\x03\x02\x02\x02\x16')
        buf.write(u'\x18\x042;\x02\x17\x16\x03\x02\x02\x02\x18\x19\x03\x02\x02\x02\x19\x17\x03\x02\x02\x02')
        buf.write(u'\x19\x1a\x03\x02\x02\x02\x1a\x06\x03\x02\x02\x02\x1b\x1c\x07=\x02\x02\x1c\x08\x03\x02\x02\x02')
        buf.write(u'\x1d\x1e\x07?\x02\x02\x1e\n\x03\x02\x02\x02\x1f \x07-\x02\x02 \x0c\x03\x02\x02\x02!"\x07')
        buf.write(u',\x02\x02"\x0e\x03\x02\x02\x02#%\x07"\x02\x02$#\x03\x02\x02\x02%&\x03\x02\x02\x02&$\x03')
        buf.write(u"\x02\x02\x02&'\x03\x02\x02\x02'\x10\x03\x02\x02\x02\x06\x02\x14\x19&\x02")
        return buf.getvalue()

class TestLexer2(Lexer):
    atn = ATNDeserializer().deserialize(serializedATN2())
    decisionsToDFA = [DFA(ds, i) for (i, ds) in enumerate(atn.decisionToState)]
    ID = 1
    INT = 2
    SEMI = 3
    ASSIGN = 4
    PLUS = 5
    MULT = 6
    WS = 7
    modeNames = [u'DEFAULT_MODE']
    literalNames = [u'<INVALID>', u"';'", u"'='", u"'+'", u"'*'"]
    symbolicNames = [u'<INVALID>', u'ID', u'INT', u'SEMI', u'ASSIGN', u'PLUS', u'MULT', u'WS']
    ruleNames = [u'ID', u'INT', u'SEMI', u'ASSIGN', u'PLUS', u'MULT', u'WS']
    grammarFileName = u'T2.g4'

    def __init__(self, input=None):
        if False:
            for i in range(10):
                print('nop')
        super(TestLexer2, self).__init__(input)
        self.checkVersion('4.9.1')
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None
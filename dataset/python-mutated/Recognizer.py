from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.error.ErrorListener import ProxyErrorListener, ConsoleErrorListener
RecognitionException = None

class Recognizer(object):
    __slots__ = ('_listeners', '_interp', '_stateNumber')
    tokenTypeMapCache = dict()
    ruleIndexMapCache = dict()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._listeners = [ConsoleErrorListener.INSTANCE]
        self._interp = None
        self._stateNumber = -1

    def extractVersion(self, version):
        if False:
            return 10
        pos = version.find('.')
        major = version[0:pos]
        version = version[pos + 1:]
        pos = version.find('.')
        if pos == -1:
            pos = version.find('-')
        if pos == -1:
            pos = len(version)
        minor = version[0:pos]
        return (major, minor)

    def checkVersion(self, toolVersion):
        if False:
            while True:
                i = 10
        runtimeVersion = '4.13.1'
        (rvmajor, rvminor) = self.extractVersion(runtimeVersion)
        (tvmajor, tvminor) = self.extractVersion(toolVersion)
        if rvmajor != tvmajor or rvminor != tvminor:
            print('ANTLR runtime and generated code versions disagree: ' + runtimeVersion + '!=' + toolVersion)

    def addErrorListener(self, listener):
        if False:
            print('Hello World!')
        self._listeners.append(listener)

    def removeErrorListener(self, listener):
        if False:
            while True:
                i = 10
        self._listeners.remove(listener)

    def removeErrorListeners(self):
        if False:
            for i in range(10):
                print('nop')
        self._listeners = []

    def getTokenTypeMap(self):
        if False:
            for i in range(10):
                print('nop')
        tokenNames = self.getTokenNames()
        if tokenNames is None:
            from antlr4.error.Errors import UnsupportedOperationException
            raise UnsupportedOperationException('The current recognizer does not provide a list of token names.')
        result = self.tokenTypeMapCache.get(tokenNames, None)
        if result is None:
            result = zip(tokenNames, range(0, len(tokenNames)))
            result['EOF'] = Token.EOF
            self.tokenTypeMapCache[tokenNames] = result
        return result

    def getRuleIndexMap(self):
        if False:
            for i in range(10):
                print('nop')
        ruleNames = self.getRuleNames()
        if ruleNames is None:
            from antlr4.error.Errors import UnsupportedOperationException
            raise UnsupportedOperationException('The current recognizer does not provide a list of rule names.')
        result = self.ruleIndexMapCache.get(ruleNames, None)
        if result is None:
            result = zip(ruleNames, range(0, len(ruleNames)))
            self.ruleIndexMapCache[ruleNames] = result
        return result

    def getTokenType(self, tokenName: str):
        if False:
            i = 10
            return i + 15
        ttype = self.getTokenTypeMap().get(tokenName, None)
        if ttype is not None:
            return ttype
        else:
            return Token.INVALID_TYPE

    def getErrorHeader(self, e: RecognitionException):
        if False:
            return 10
        line = e.getOffendingToken().line
        column = e.getOffendingToken().column
        return 'line ' + line + ':' + column

    def getTokenErrorDisplay(self, t: Token):
        if False:
            return 10
        if t is None:
            return '<no token>'
        s = t.text
        if s is None:
            if t.type == Token.EOF:
                s = '<EOF>'
            else:
                s = '<' + str(t.type) + '>'
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return "'" + s + "'"

    def getErrorListenerDispatch(self):
        if False:
            i = 10
            return i + 15
        return ProxyErrorListener(self._listeners)

    def sempred(self, localctx: RuleContext, ruleIndex: int, actionIndex: int):
        if False:
            return 10
        return True

    def precpred(self, localctx: RuleContext, precedence: int):
        if False:
            for i in range(10):
                print('nop')
        return True

    @property
    def state(self):
        if False:
            print('Hello World!')
        return self._stateNumber

    @state.setter
    def state(self, atnState: int):
        if False:
            i = 10
            return i + 15
        self._stateNumber = atnState
del RecognitionException
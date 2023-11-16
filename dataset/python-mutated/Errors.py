Token = None
Lexer = None
Parser = None
TokenStream = None
ATNConfigSet = None
ParserRulecontext = None
PredicateTransition = None
BufferedTokenStream = None

class UnsupportedOperationException(Exception):

    def __init__(self, msg: str):
        if False:
            i = 10
            return i + 15
        super().__init__(msg)

class IllegalStateException(Exception):

    def __init__(self, msg: str):
        if False:
            while True:
                i = 10
        super().__init__(msg)

class CancellationException(IllegalStateException):

    def __init__(self, msg: str):
        if False:
            while True:
                i = 10
        super().__init__(msg)
from antlr4.InputStream import InputStream
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Recognizer import Recognizer

class RecognitionException(Exception):

    def __init__(self, message: str=None, recognizer: Recognizer=None, input: InputStream=None, ctx: ParserRulecontext=None):
        if False:
            print('Hello World!')
        super().__init__(message)
        self.message = message
        self.recognizer = recognizer
        self.input = input
        self.ctx = ctx
        self.offendingToken = None
        self.offendingState = -1
        if recognizer is not None:
            self.offendingState = recognizer.state

    def getExpectedTokens(self):
        if False:
            i = 10
            return i + 15
        if self.recognizer is not None:
            return self.recognizer.atn.getExpectedTokens(self.offendingState, self.ctx)
        else:
            return None

class LexerNoViableAltException(RecognitionException):

    def __init__(self, lexer: Lexer, input: InputStream, startIndex: int, deadEndConfigs: ATNConfigSet):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(message=None, recognizer=lexer, input=input, ctx=None)
        self.startIndex = startIndex
        self.deadEndConfigs = deadEndConfigs
        self.message = ''

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        symbol = ''
        if self.startIndex >= 0 and self.startIndex < self.input.size:
            symbol = self.input.getText(self.startIndex, self.startIndex)
        return "LexerNoViableAltException('" + symbol + "')"

class NoViableAltException(RecognitionException):

    def __init__(self, recognizer: Parser, input: TokenStream=None, startToken: Token=None, offendingToken: Token=None, deadEndConfigs: ATNConfigSet=None, ctx: ParserRuleContext=None):
        if False:
            while True:
                i = 10
        if ctx is None:
            ctx = recognizer._ctx
        if offendingToken is None:
            offendingToken = recognizer.getCurrentToken()
        if startToken is None:
            startToken = recognizer.getCurrentToken()
        if input is None:
            input = recognizer.getInputStream()
        super().__init__(recognizer=recognizer, input=input, ctx=ctx)
        self.deadEndConfigs = deadEndConfigs
        self.startToken = startToken
        self.offendingToken = offendingToken

class InputMismatchException(RecognitionException):

    def __init__(self, recognizer: Parser):
        if False:
            while True:
                i = 10
        super().__init__(recognizer=recognizer, input=recognizer.getInputStream(), ctx=recognizer._ctx)
        self.offendingToken = recognizer.getCurrentToken()

class FailedPredicateException(RecognitionException):

    def __init__(self, recognizer: Parser, predicate: str=None, message: str=None):
        if False:
            return 10
        super().__init__(message=self.formatMessage(predicate, message), recognizer=recognizer, input=recognizer.getInputStream(), ctx=recognizer._ctx)
        s = recognizer._interp.atn.states[recognizer.state]
        trans = s.transitions[0]
        from antlr4.atn.Transition import PredicateTransition
        if isinstance(trans, PredicateTransition):
            self.ruleIndex = trans.ruleIndex
            self.predicateIndex = trans.predIndex
        else:
            self.ruleIndex = 0
            self.predicateIndex = 0
        self.predicate = predicate
        self.offendingToken = recognizer.getCurrentToken()

    def formatMessage(self, predicate: str, message: str):
        if False:
            print('Hello World!')
        if message is not None:
            return message
        else:
            return 'failed predicate: {' + predicate + '}?'

class ParseCancellationException(CancellationException):
    pass
del Token
del Lexer
del Parser
del TokenStream
del ATNConfigSet
del ParserRulecontext
del PredicateTransition
del BufferedTokenStream
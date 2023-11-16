import sys
from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.ATNState import ATNState
from antlr4.error.Errors import RecognitionException, NoViableAltException, InputMismatchException, FailedPredicateException, ParseCancellationException
Parser = None

class ErrorStrategy(object):

    def reset(self, recognizer: Parser):
        if False:
            i = 10
            return i + 15
        pass

    def recoverInline(self, recognizer: Parser):
        if False:
            while True:
                i = 10
        pass

    def recover(self, recognizer: Parser, e: RecognitionException):
        if False:
            print('Hello World!')
        pass

    def sync(self, recognizer: Parser):
        if False:
            i = 10
            return i + 15
        pass

    def inErrorRecoveryMode(self, recognizer: Parser):
        if False:
            for i in range(10):
                print('nop')
        pass

    def reportError(self, recognizer: Parser, e: RecognitionException):
        if False:
            return 10
        pass

class DefaultErrorStrategy(ErrorStrategy):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.errorRecoveryMode = False
        self.lastErrorIndex = -1
        self.lastErrorStates = None
        self.nextTokensContext = None
        self.nextTokenState = 0

    def reset(self, recognizer: Parser):
        if False:
            for i in range(10):
                print('nop')
        self.endErrorCondition(recognizer)

    def beginErrorCondition(self, recognizer: Parser):
        if False:
            for i in range(10):
                print('nop')
        self.errorRecoveryMode = True

    def inErrorRecoveryMode(self, recognizer: Parser):
        if False:
            for i in range(10):
                print('nop')
        return self.errorRecoveryMode

    def endErrorCondition(self, recognizer: Parser):
        if False:
            for i in range(10):
                print('nop')
        self.errorRecoveryMode = False
        self.lastErrorStates = None
        self.lastErrorIndex = -1

    def reportMatch(self, recognizer: Parser):
        if False:
            i = 10
            return i + 15
        self.endErrorCondition(recognizer)

    def reportError(self, recognizer: Parser, e: RecognitionException):
        if False:
            print('Hello World!')
        if self.inErrorRecoveryMode(recognizer):
            return
        self.beginErrorCondition(recognizer)
        if isinstance(e, NoViableAltException):
            self.reportNoViableAlternative(recognizer, e)
        elif isinstance(e, InputMismatchException):
            self.reportInputMismatch(recognizer, e)
        elif isinstance(e, FailedPredicateException):
            self.reportFailedPredicate(recognizer, e)
        else:
            print('unknown recognition error type: ' + type(e).__name__)
            recognizer.notifyErrorListeners(e.message, e.offendingToken, e)

    def recover(self, recognizer: Parser, e: RecognitionException):
        if False:
            i = 10
            return i + 15
        if self.lastErrorIndex == recognizer.getInputStream().index and self.lastErrorStates is not None and (recognizer.state in self.lastErrorStates):
            recognizer.consume()
        self.lastErrorIndex = recognizer._input.index
        if self.lastErrorStates is None:
            self.lastErrorStates = []
        self.lastErrorStates.append(recognizer.state)
        followSet = self.getErrorRecoverySet(recognizer)
        self.consumeUntil(recognizer, followSet)

    def sync(self, recognizer: Parser):
        if False:
            while True:
                i = 10
        if self.inErrorRecoveryMode(recognizer):
            return
        s = recognizer._interp.atn.states[recognizer.state]
        la = recognizer.getTokenStream().LA(1)
        nextTokens = recognizer.atn.nextTokens(s)
        if la in nextTokens:
            self.nextTokensContext = None
            self.nextTokenState = ATNState.INVALID_STATE_NUMBER
            return
        elif Token.EPSILON in nextTokens:
            if self.nextTokensContext is None:
                self.nextTokensContext = recognizer._ctx
                self.nextTokensState = recognizer._stateNumber
            return
        if s.stateType in [ATNState.BLOCK_START, ATNState.STAR_BLOCK_START, ATNState.PLUS_BLOCK_START, ATNState.STAR_LOOP_ENTRY]:
            if self.singleTokenDeletion(recognizer) is not None:
                return
            else:
                raise InputMismatchException(recognizer)
        elif s.stateType in [ATNState.PLUS_LOOP_BACK, ATNState.STAR_LOOP_BACK]:
            self.reportUnwantedToken(recognizer)
            expecting = recognizer.getExpectedTokens()
            whatFollowsLoopIterationOrRule = expecting.addSet(self.getErrorRecoverySet(recognizer))
            self.consumeUntil(recognizer, whatFollowsLoopIterationOrRule)
        else:
            pass

    def reportNoViableAlternative(self, recognizer: Parser, e: NoViableAltException):
        if False:
            i = 10
            return i + 15
        tokens = recognizer.getTokenStream()
        if tokens is not None:
            if e.startToken.type == Token.EOF:
                input = '<EOF>'
            else:
                input = tokens.getText(e.startToken, e.offendingToken)
        else:
            input = '<unknown input>'
        msg = 'no viable alternative at input ' + self.escapeWSAndQuote(input)
        recognizer.notifyErrorListeners(msg, e.offendingToken, e)

    def reportInputMismatch(self, recognizer: Parser, e: InputMismatchException):
        if False:
            return 10
        msg = 'mismatched input ' + self.getTokenErrorDisplay(e.offendingToken) + ' expecting ' + e.getExpectedTokens().toString(recognizer.literalNames, recognizer.symbolicNames)
        recognizer.notifyErrorListeners(msg, e.offendingToken, e)

    def reportFailedPredicate(self, recognizer, e):
        if False:
            return 10
        ruleName = recognizer.ruleNames[recognizer._ctx.getRuleIndex()]
        msg = 'rule ' + ruleName + ' ' + e.message
        recognizer.notifyErrorListeners(msg, e.offendingToken, e)

    def reportUnwantedToken(self, recognizer: Parser):
        if False:
            print('Hello World!')
        if self.inErrorRecoveryMode(recognizer):
            return
        self.beginErrorCondition(recognizer)
        t = recognizer.getCurrentToken()
        tokenName = self.getTokenErrorDisplay(t)
        expecting = self.getExpectedTokens(recognizer)
        msg = 'extraneous input ' + tokenName + ' expecting ' + expecting.toString(recognizer.literalNames, recognizer.symbolicNames)
        recognizer.notifyErrorListeners(msg, t, None)

    def reportMissingToken(self, recognizer: Parser):
        if False:
            i = 10
            return i + 15
        if self.inErrorRecoveryMode(recognizer):
            return
        self.beginErrorCondition(recognizer)
        t = recognizer.getCurrentToken()
        expecting = self.getExpectedTokens(recognizer)
        msg = 'missing ' + expecting.toString(recognizer.literalNames, recognizer.symbolicNames) + ' at ' + self.getTokenErrorDisplay(t)
        recognizer.notifyErrorListeners(msg, t, None)

    def recoverInline(self, recognizer: Parser):
        if False:
            for i in range(10):
                print('nop')
        matchedSymbol = self.singleTokenDeletion(recognizer)
        if matchedSymbol is not None:
            recognizer.consume()
            return matchedSymbol
        if self.singleTokenInsertion(recognizer):
            return self.getMissingSymbol(recognizer)
        raise InputMismatchException(recognizer)

    def singleTokenInsertion(self, recognizer: Parser):
        if False:
            while True:
                i = 10
        currentSymbolType = recognizer.getTokenStream().LA(1)
        atn = recognizer._interp.atn
        currentState = atn.states[recognizer.state]
        next = currentState.transitions[0].target
        expectingAtLL2 = atn.nextTokens(next, recognizer._ctx)
        if currentSymbolType in expectingAtLL2:
            self.reportMissingToken(recognizer)
            return True
        else:
            return False

    def singleTokenDeletion(self, recognizer: Parser):
        if False:
            while True:
                i = 10
        nextTokenType = recognizer.getTokenStream().LA(2)
        expecting = self.getExpectedTokens(recognizer)
        if nextTokenType in expecting:
            self.reportUnwantedToken(recognizer)
            recognizer.consume()
            matchedSymbol = recognizer.getCurrentToken()
            self.reportMatch(recognizer)
            return matchedSymbol
        else:
            return None

    def getMissingSymbol(self, recognizer: Parser):
        if False:
            return 10
        currentSymbol = recognizer.getCurrentToken()
        expecting = self.getExpectedTokens(recognizer)
        expectedTokenType = expecting[0]
        if expectedTokenType == Token.EOF:
            tokenText = '<missing EOF>'
        else:
            name = None
            if expectedTokenType < len(recognizer.literalNames):
                name = recognizer.literalNames[expectedTokenType]
            if name is None and expectedTokenType < len(recognizer.symbolicNames):
                name = recognizer.symbolicNames[expectedTokenType]
            tokenText = '<missing ' + str(name) + '>'
        current = currentSymbol
        lookback = recognizer.getTokenStream().LT(-1)
        if current.type == Token.EOF and lookback is not None:
            current = lookback
        return recognizer.getTokenFactory().create(current.source, expectedTokenType, tokenText, Token.DEFAULT_CHANNEL, -1, -1, current.line, current.column)

    def getExpectedTokens(self, recognizer: Parser):
        if False:
            print('Hello World!')
        return recognizer.getExpectedTokens()

    def getTokenErrorDisplay(self, t: Token):
        if False:
            while True:
                i = 10
        if t is None:
            return '<no token>'
        s = t.text
        if s is None:
            if t.type == Token.EOF:
                s = '<EOF>'
            else:
                s = '<' + str(t.type) + '>'
        return self.escapeWSAndQuote(s)

    def escapeWSAndQuote(self, s: str):
        if False:
            for i in range(10):
                print('nop')
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return "'" + s + "'"

    def getErrorRecoverySet(self, recognizer: Parser):
        if False:
            print('Hello World!')
        atn = recognizer._interp.atn
        ctx = recognizer._ctx
        recoverSet = IntervalSet()
        while ctx is not None and ctx.invokingState >= 0:
            invokingState = atn.states[ctx.invokingState]
            rt = invokingState.transitions[0]
            follow = atn.nextTokens(rt.followState)
            recoverSet.addSet(follow)
            ctx = ctx.parentCtx
        recoverSet.removeOne(Token.EPSILON)
        return recoverSet

    def consumeUntil(self, recognizer: Parser, set_: set):
        if False:
            print('Hello World!')
        ttype = recognizer.getTokenStream().LA(1)
        while ttype != Token.EOF and (not ttype in set_):
            recognizer.consume()
            ttype = recognizer.getTokenStream().LA(1)

class BailErrorStrategy(DefaultErrorStrategy):

    def recover(self, recognizer: Parser, e: RecognitionException):
        if False:
            while True:
                i = 10
        context = recognizer._ctx
        while context is not None:
            context.exception = e
            context = context.parentCtx
        raise ParseCancellationException(e)

    def recoverInline(self, recognizer: Parser):
        if False:
            while True:
                i = 10
        self.recover(recognizer, InputMismatchException(recognizer))

    def sync(self, recognizer: Parser):
        if False:
            for i in range(10):
                print('nop')
        pass
del Parser
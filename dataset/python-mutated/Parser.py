import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO
from antlr4.BufferedTokenStream import TokenStream
from antlr4.CommonTokenFactory import TokenFactory
from antlr4.error.ErrorStrategy import DefaultErrorStrategy
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Token import Token
from antlr4.Lexer import Lexer
from antlr4.atn.ATNDeserializer import ATNDeserializer
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
from antlr4.error.Errors import UnsupportedOperationException, RecognitionException
from antlr4.tree.ParseTreePatternMatcher import ParseTreePatternMatcher
from antlr4.tree.Tree import ParseTreeListener, TerminalNode, ErrorNode

class TraceListener(ParseTreeListener):
    __slots__ = '_parser'

    def __init__(self, parser):
        if False:
            print('Hello World!')
        self._parser = parser

    def enterEveryRule(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        print('enter   ' + self._parser.ruleNames[ctx.getRuleIndex()] + ', LT(1)=' + self._parser._input.LT(1).text, file=self._parser._output)

    def visitTerminal(self, node):
        if False:
            return 10
        print('consume ' + str(node.symbol) + ' rule ' + self._parser.ruleNames[self._parser._ctx.getRuleIndex()], file=self._parser._output)

    def visitErrorNode(self, node):
        if False:
            while True:
                i = 10
        pass

    def exitEveryRule(self, ctx):
        if False:
            while True:
                i = 10
        print('exit    ' + self._parser.ruleNames[ctx.getRuleIndex()] + ', LT(1)=' + self._parser._input.LT(1).text, file=self._parser._output)

class Parser(Recognizer):
    __slots__ = ('_input', '_output', '_errHandler', '_precedenceStack', '_ctx', 'buildParseTrees', '_tracer', '_parseListeners', '_syntaxErrors')
    bypassAltsAtnCache = dict()

    def __init__(self, input: TokenStream, output: TextIO=sys.stdout):
        if False:
            while True:
                i = 10
        super().__init__()
        self._input = None
        self._output = output
        self._errHandler = DefaultErrorStrategy()
        self._precedenceStack = list()
        self._precedenceStack.append(0)
        self._ctx = None
        self.buildParseTrees = True
        self._tracer = None
        self._parseListeners = None
        self._syntaxErrors = 0
        self.setInputStream(input)

    def reset(self):
        if False:
            return 10
        if self._input is not None:
            self._input.seek(0)
        self._errHandler.reset(self)
        self._ctx = None
        self._syntaxErrors = 0
        self.setTrace(False)
        self._precedenceStack = list()
        self._precedenceStack.append(0)
        if self._interp is not None:
            self._interp.reset()

    def match(self, ttype: int):
        if False:
            i = 10
            return i + 15
        t = self.getCurrentToken()
        if t.type == ttype:
            self._errHandler.reportMatch(self)
            self.consume()
        else:
            t = self._errHandler.recoverInline(self)
            if self.buildParseTrees and t.tokenIndex == -1:
                self._ctx.addErrorNode(t)
        return t

    def matchWildcard(self):
        if False:
            return 10
        t = self.getCurrentToken()
        if t.type > 0:
            self._errHandler.reportMatch(self)
            self.consume()
        else:
            t = self._errHandler.recoverInline(self)
            if self.buildParseTrees and t.tokenIndex == -1:
                self._ctx.addErrorNode(t)
        return t

    def getParseListeners(self):
        if False:
            i = 10
            return i + 15
        return list() if self._parseListeners is None else self._parseListeners

    def addParseListener(self, listener: ParseTreeListener):
        if False:
            while True:
                i = 10
        if listener is None:
            raise ReferenceError('listener')
        if self._parseListeners is None:
            self._parseListeners = []
        self._parseListeners.append(listener)

    def removeParseListener(self, listener: ParseTreeListener):
        if False:
            while True:
                i = 10
        if self._parseListeners is not None:
            self._parseListeners.remove(listener)
            if len(self._parseListeners) == 0:
                self._parseListeners = None

    def removeParseListeners(self):
        if False:
            print('Hello World!')
        self._parseListeners = None

    def triggerEnterRuleEvent(self):
        if False:
            i = 10
            return i + 15
        if self._parseListeners is not None:
            for listener in self._parseListeners:
                listener.enterEveryRule(self._ctx)
                self._ctx.enterRule(listener)

    def triggerExitRuleEvent(self):
        if False:
            print('Hello World!')
        if self._parseListeners is not None:
            for listener in reversed(self._parseListeners):
                self._ctx.exitRule(listener)
                listener.exitEveryRule(self._ctx)

    def getNumberOfSyntaxErrors(self):
        if False:
            for i in range(10):
                print('nop')
        return self._syntaxErrors

    def getTokenFactory(self):
        if False:
            for i in range(10):
                print('nop')
        return self._input.tokenSource._factory

    def setTokenFactory(self, factory: TokenFactory):
        if False:
            print('Hello World!')
        self._input.tokenSource._factory = factory

    def getATNWithBypassAlts(self):
        if False:
            return 10
        serializedAtn = self.getSerializedATN()
        if serializedAtn is None:
            raise UnsupportedOperationException('The current parser does not support an ATN with bypass alternatives.')
        result = self.bypassAltsAtnCache.get(serializedAtn, None)
        if result is None:
            deserializationOptions = ATNDeserializationOptions()
            deserializationOptions.generateRuleBypassTransitions = True
            result = ATNDeserializer(deserializationOptions).deserialize(serializedAtn)
            self.bypassAltsAtnCache[serializedAtn] = result
        return result

    def compileParseTreePattern(self, pattern: str, patternRuleIndex: int, lexer: Lexer=None):
        if False:
            print('Hello World!')
        if lexer is None:
            if self.getTokenStream() is not None:
                tokenSource = self.getTokenStream().tokenSource
                if isinstance(tokenSource, Lexer):
                    lexer = tokenSource
        if lexer is None:
            raise UnsupportedOperationException("Parser can't discover a lexer to use")
        m = ParseTreePatternMatcher(lexer, self)
        return m.compile(pattern, patternRuleIndex)

    def getInputStream(self):
        if False:
            while True:
                i = 10
        return self.getTokenStream()

    def setInputStream(self, input: InputStream):
        if False:
            i = 10
            return i + 15
        self.setTokenStream(input)

    def getTokenStream(self):
        if False:
            return 10
        return self._input

    def setTokenStream(self, input: TokenStream):
        if False:
            print('Hello World!')
        self._input = None
        self.reset()
        self._input = input

    def getCurrentToken(self):
        if False:
            while True:
                i = 10
        return self._input.LT(1)

    def notifyErrorListeners(self, msg: str, offendingToken: Token=None, e: RecognitionException=None):
        if False:
            while True:
                i = 10
        if offendingToken is None:
            offendingToken = self.getCurrentToken()
        self._syntaxErrors += 1
        line = offendingToken.line
        column = offendingToken.column
        listener = self.getErrorListenerDispatch()
        listener.syntaxError(self, offendingToken, line, column, msg, e)

    def consume(self):
        if False:
            while True:
                i = 10
        o = self.getCurrentToken()
        if o.type != Token.EOF:
            self.getInputStream().consume()
        hasListener = self._parseListeners is not None and len(self._parseListeners) > 0
        if self.buildParseTrees or hasListener:
            if self._errHandler.inErrorRecoveryMode(self):
                node = self._ctx.addErrorNode(o)
            else:
                node = self._ctx.addTokenNode(o)
            if hasListener:
                for listener in self._parseListeners:
                    if isinstance(node, ErrorNode):
                        listener.visitErrorNode(node)
                    elif isinstance(node, TerminalNode):
                        listener.visitTerminal(node)
        return o

    def addContextToParseTree(self):
        if False:
            print('Hello World!')
        if self._ctx.parentCtx is not None:
            self._ctx.parentCtx.addChild(self._ctx)

    def enterRule(self, localctx: ParserRuleContext, state: int, ruleIndex: int):
        if False:
            return 10
        self.state = state
        self._ctx = localctx
        self._ctx.start = self._input.LT(1)
        if self.buildParseTrees:
            self.addContextToParseTree()
        if self._parseListeners is not None:
            self.triggerEnterRuleEvent()

    def exitRule(self):
        if False:
            return 10
        self._ctx.stop = self._input.LT(-1)
        if self._parseListeners is not None:
            self.triggerExitRuleEvent()
        self.state = self._ctx.invokingState
        self._ctx = self._ctx.parentCtx

    def enterOuterAlt(self, localctx: ParserRuleContext, altNum: int):
        if False:
            for i in range(10):
                print('nop')
        localctx.setAltNumber(altNum)
        if self.buildParseTrees and self._ctx != localctx:
            if self._ctx.parentCtx is not None:
                self._ctx.parentCtx.removeLastChild()
                self._ctx.parentCtx.addChild(localctx)
        self._ctx = localctx

    def getPrecedence(self):
        if False:
            i = 10
            return i + 15
        if len(self._precedenceStack) == 0:
            return -1
        else:
            return self._precedenceStack[-1]

    def enterRecursionRule(self, localctx: ParserRuleContext, state: int, ruleIndex: int, precedence: int):
        if False:
            for i in range(10):
                print('nop')
        self.state = state
        self._precedenceStack.append(precedence)
        self._ctx = localctx
        self._ctx.start = self._input.LT(1)
        if self._parseListeners is not None:
            self.triggerEnterRuleEvent()

    def pushNewRecursionContext(self, localctx: ParserRuleContext, state: int, ruleIndex: int):
        if False:
            for i in range(10):
                print('nop')
        previous = self._ctx
        previous.parentCtx = localctx
        previous.invokingState = state
        previous.stop = self._input.LT(-1)
        self._ctx = localctx
        self._ctx.start = previous.start
        if self.buildParseTrees:
            self._ctx.addChild(previous)
        if self._parseListeners is not None:
            self.triggerEnterRuleEvent()

    def unrollRecursionContexts(self, parentCtx: ParserRuleContext):
        if False:
            return 10
        self._precedenceStack.pop()
        self._ctx.stop = self._input.LT(-1)
        retCtx = self._ctx
        if self._parseListeners is not None:
            while self._ctx is not parentCtx:
                self.triggerExitRuleEvent()
                self._ctx = self._ctx.parentCtx
        else:
            self._ctx = parentCtx
        retCtx.parentCtx = parentCtx
        if self.buildParseTrees and parentCtx is not None:
            parentCtx.addChild(retCtx)

    def getInvokingContext(self, ruleIndex: int):
        if False:
            print('Hello World!')
        ctx = self._ctx
        while ctx is not None:
            if ctx.getRuleIndex() == ruleIndex:
                return ctx
            ctx = ctx.parentCtx
        return None

    def precpred(self, localctx: RuleContext, precedence: int):
        if False:
            return 10
        return precedence >= self._precedenceStack[-1]

    def inContext(self, context: str):
        if False:
            for i in range(10):
                print('nop')
        return False

    def isExpectedToken(self, symbol: int):
        if False:
            for i in range(10):
                print('nop')
        atn = self._interp.atn
        ctx = self._ctx
        s = atn.states[self.state]
        following = atn.nextTokens(s)
        if symbol in following:
            return True
        if not Token.EPSILON in following:
            return False
        while ctx is not None and ctx.invokingState >= 0 and (Token.EPSILON in following):
            invokingState = atn.states[ctx.invokingState]
            rt = invokingState.transitions[0]
            following = atn.nextTokens(rt.followState)
            if symbol in following:
                return True
            ctx = ctx.parentCtx
        if Token.EPSILON in following and symbol == Token.EOF:
            return True
        else:
            return False

    def getExpectedTokens(self):
        if False:
            return 10
        return self._interp.atn.getExpectedTokens(self.state, self._ctx)

    def getExpectedTokensWithinCurrentRule(self):
        if False:
            i = 10
            return i + 15
        atn = self._interp.atn
        s = atn.states[self.state]
        return atn.nextTokens(s)

    def getRuleIndex(self, ruleName: str):
        if False:
            return 10
        ruleIndex = self.getRuleIndexMap().get(ruleName, None)
        if ruleIndex is not None:
            return ruleIndex
        else:
            return -1

    def getRuleInvocationStack(self, p: RuleContext=None):
        if False:
            print('Hello World!')
        if p is None:
            p = self._ctx
        stack = list()
        while p is not None:
            ruleIndex = p.getRuleIndex()
            if ruleIndex < 0:
                stack.append('n/a')
            else:
                stack.append(self.ruleNames[ruleIndex])
            p = p.parentCtx
        return stack

    def getDFAStrings(self):
        if False:
            while True:
                i = 10
        return [str(dfa) for dfa in self._interp.decisionToDFA]

    def dumpDFA(self):
        if False:
            return 10
        seenOne = False
        for i in range(0, len(self._interp.decisionToDFA)):
            dfa = self._interp.decisionToDFA[i]
            if len(dfa.states) > 0:
                if seenOne:
                    print(file=self._output)
                print('Decision ' + str(dfa.decision) + ':', file=self._output)
                print(dfa.toString(self.literalNames, self.symbolicNames), end='', file=self._output)
                seenOne = True

    def getSourceName(self):
        if False:
            i = 10
            return i + 15
        return self._input.sourceName

    def setTrace(self, trace: bool):
        if False:
            for i in range(10):
                print('nop')
        if not trace:
            self.removeParseListener(self._tracer)
            self._tracer = None
        else:
            if self._tracer is not None:
                self.removeParseListener(self._tracer)
            self._tracer = TraceListener(self)
            self.addParseListener(self._tracer)
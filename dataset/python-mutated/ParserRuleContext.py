from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.tree.Tree import ParseTreeListener, ParseTree, TerminalNodeImpl, ErrorNodeImpl, TerminalNode, INVALID_INTERVAL
ParserRuleContext = None

class ParserRuleContext(RuleContext):
    __slots__ = ('children', 'start', 'stop', 'exception')

    def __init__(self, parent: ParserRuleContext=None, invokingStateNumber: int=None):
        if False:
            while True:
                i = 10
        super().__init__(parent, invokingStateNumber)
        self.children = None
        self.start = None
        self.stop = None
        self.exception = None

    def copyFrom(self, ctx: ParserRuleContext):
        if False:
            print('Hello World!')
        self.parentCtx = ctx.parentCtx
        self.invokingState = ctx.invokingState
        self.children = None
        self.start = ctx.start
        self.stop = ctx.stop
        if ctx.children is not None:
            self.children = []
            for child in ctx.children:
                if isinstance(child, ErrorNodeImpl):
                    self.children.append(child)
                    child.parentCtx = self

    def enterRule(self, listener: ParseTreeListener):
        if False:
            while True:
                i = 10
        pass

    def exitRule(self, listener: ParseTreeListener):
        if False:
            print('Hello World!')
        pass

    def addChild(self, child: ParseTree):
        if False:
            return 10
        if self.children is None:
            self.children = []
        self.children.append(child)
        return child

    def removeLastChild(self):
        if False:
            while True:
                i = 10
        if self.children is not None:
            del self.children[len(self.children) - 1]

    def addTokenNode(self, token: Token):
        if False:
            for i in range(10):
                print('nop')
        node = TerminalNodeImpl(token)
        self.addChild(node)
        node.parentCtx = self
        return node

    def addErrorNode(self, badToken: Token):
        if False:
            while True:
                i = 10
        node = ErrorNodeImpl(badToken)
        self.addChild(node)
        node.parentCtx = self
        return node

    def getChild(self, i: int, ttype: type=None):
        if False:
            while True:
                i = 10
        if ttype is None:
            return self.children[i] if len(self.children) > i else None
        else:
            for child in self.getChildren():
                if not isinstance(child, ttype):
                    continue
                if i == 0:
                    return child
                i -= 1
            return None

    def getChildren(self, predicate=None):
        if False:
            for i in range(10):
                print('nop')
        if self.children is not None:
            for child in self.children:
                if predicate is not None and (not predicate(child)):
                    continue
                yield child

    def getToken(self, ttype: int, i: int):
        if False:
            print('Hello World!')
        for child in self.getChildren():
            if not isinstance(child, TerminalNode):
                continue
            if child.symbol.type != ttype:
                continue
            if i == 0:
                return child
            i -= 1
        return None

    def getTokens(self, ttype: int):
        if False:
            while True:
                i = 10
        if self.getChildren() is None:
            return []
        tokens = []
        for child in self.getChildren():
            if not isinstance(child, TerminalNode):
                continue
            if child.symbol.type != ttype:
                continue
            tokens.append(child)
        return tokens

    def getTypedRuleContext(self, ctxType: type, i: int):
        if False:
            while True:
                i = 10
        return self.getChild(i, ctxType)

    def getTypedRuleContexts(self, ctxType: type):
        if False:
            return 10
        children = self.getChildren()
        if children is None:
            return []
        contexts = []
        for child in children:
            if not isinstance(child, ctxType):
                continue
            contexts.append(child)
        return contexts

    def getChildCount(self):
        if False:
            return 10
        return len(self.children) if self.children else 0

    def getSourceInterval(self):
        if False:
            return 10
        if self.start is None or self.stop is None:
            return INVALID_INTERVAL
        else:
            return (self.start.tokenIndex, self.stop.tokenIndex)
RuleContext.EMPTY = ParserRuleContext()

class InterpreterRuleContext(ParserRuleContext):

    def __init__(self, parent: ParserRuleContext, invokingStateNumber: int, ruleIndex: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent, invokingStateNumber)
        self.ruleIndex = ruleIndex
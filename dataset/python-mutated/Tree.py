from antlr4.Token import Token
INVALID_INTERVAL = (-1, -2)

class Tree(object):
    pass

class SyntaxTree(Tree):
    pass

class ParseTree(SyntaxTree):
    pass

class RuleNode(ParseTree):
    pass

class TerminalNode(ParseTree):
    pass

class ErrorNode(TerminalNode):
    pass

class ParseTreeVisitor(object):

    def visit(self, tree):
        if False:
            for i in range(10):
                print('nop')
        return tree.accept(self)

    def visitChildren(self, node):
        if False:
            return 10
        result = self.defaultResult()
        n = node.getChildCount()
        for i in range(n):
            if not self.shouldVisitNextChild(node, result):
                return result
            c = node.getChild(i)
            childResult = c.accept(self)
            result = self.aggregateResult(result, childResult)
        return result

    def visitTerminal(self, node):
        if False:
            print('Hello World!')
        return self.defaultResult()

    def visitErrorNode(self, node):
        if False:
            i = 10
            return i + 15
        return self.defaultResult()

    def defaultResult(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def aggregateResult(self, aggregate, nextResult):
        if False:
            for i in range(10):
                print('nop')
        return nextResult

    def shouldVisitNextChild(self, node, currentResult):
        if False:
            for i in range(10):
                print('nop')
        return True
ParserRuleContext = None

class ParseTreeListener(object):

    def visitTerminal(self, node: TerminalNode):
        if False:
            while True:
                i = 10
        pass

    def visitErrorNode(self, node: ErrorNode):
        if False:
            return 10
        pass

    def enterEveryRule(self, ctx: ParserRuleContext):
        if False:
            print('Hello World!')
        pass

    def exitEveryRule(self, ctx: ParserRuleContext):
        if False:
            while True:
                i = 10
        pass
del ParserRuleContext

class TerminalNodeImpl(TerminalNode):
    __slots__ = ('parentCtx', 'symbol')

    def __init__(self, symbol: Token):
        if False:
            return 10
        self.parentCtx = None
        self.symbol = symbol

    def __setattr__(self, key, value):
        if False:
            return 10
        super().__setattr__(key, value)

    def getChild(self, i: int):
        if False:
            while True:
                i = 10
        return None

    def getSymbol(self):
        if False:
            for i in range(10):
                print('nop')
        return self.symbol

    def getParent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parentCtx

    def getPayload(self):
        if False:
            for i in range(10):
                print('nop')
        return self.symbol

    def getSourceInterval(self):
        if False:
            i = 10
            return i + 15
        if self.symbol is None:
            return INVALID_INTERVAL
        tokenIndex = self.symbol.tokenIndex
        return (tokenIndex, tokenIndex)

    def getChildCount(self):
        if False:
            return 10
        return 0

    def accept(self, visitor: ParseTreeVisitor):
        if False:
            i = 10
            return i + 15
        return visitor.visitTerminal(self)

    def getText(self):
        if False:
            for i in range(10):
                print('nop')
        return self.symbol.text

    def __str__(self):
        if False:
            return 10
        if self.symbol.type == Token.EOF:
            return '<EOF>'
        else:
            return self.symbol.text

class ErrorNodeImpl(TerminalNodeImpl, ErrorNode):

    def __init__(self, token: Token):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(token)

    def accept(self, visitor: ParseTreeVisitor):
        if False:
            for i in range(10):
                print('nop')
        return visitor.visitErrorNode(self)

class ParseTreeWalker(object):
    DEFAULT = None

    def walk(self, listener: ParseTreeListener, t: ParseTree):
        if False:
            return 10
        '\n\t    Performs a walk on the given parse tree starting at the root and going down recursively\n\t    with depth-first search. On each node, {@link ParseTreeWalker#enterRule} is called before\n\t    recursively walking down into child nodes, then\n\t    {@link ParseTreeWalker#exitRule} is called after the recursive call to wind up.\n\t    @param listener The listener used by the walker to process grammar rules\n\t    @param t The parse tree to be walked on\n        '
        if isinstance(t, ErrorNode):
            listener.visitErrorNode(t)
            return
        elif isinstance(t, TerminalNode):
            listener.visitTerminal(t)
            return
        self.enterRule(listener, t)
        for child in t.getChildren():
            self.walk(listener, child)
        self.exitRule(listener, t)

    def enterRule(self, listener: ParseTreeListener, r: RuleNode):
        if False:
            while True:
                i = 10
        '\n\t    Enters a grammar rule by first triggering the generic event {@link ParseTreeListener#enterEveryRule}\n\t    then by triggering the event specific to the given parse tree node\n\t    @param listener The listener responding to the trigger events\n\t    @param r The grammar rule containing the rule context\n        '
        ctx = r.getRuleContext()
        listener.enterEveryRule(ctx)
        ctx.enterRule(listener)

    def exitRule(self, listener: ParseTreeListener, r: RuleNode):
        if False:
            return 10
        '\n\t    Exits a grammar rule by first triggering the event specific to the given parse tree node\n\t    then by triggering the generic event {@link ParseTreeListener#exitEveryRule}\n\t    @param listener The listener responding to the trigger events\n\t    @param r The grammar rule containing the rule context\n        '
        ctx = r.getRuleContext()
        ctx.exitRule(listener)
        listener.exitEveryRule(ctx)
ParseTreeWalker.DEFAULT = ParseTreeWalker()
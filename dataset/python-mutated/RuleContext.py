from io import StringIO
from antlr4.tree.Tree import RuleNode, INVALID_INTERVAL, ParseTreeVisitor
from antlr4.tree.Trees import Trees
RuleContext = None
Parser = None

class RuleContext(RuleNode):
    __slots__ = ('parentCtx', 'invokingState')
    EMPTY = None

    def __init__(self, parent: RuleContext=None, invokingState: int=-1):
        if False:
            while True:
                i = 10
        super().__init__()
        self.parentCtx = parent
        self.invokingState = invokingState

    def depth(self):
        if False:
            for i in range(10):
                print('nop')
        n = 0
        p = self
        while p is not None:
            p = p.parentCtx
            n += 1
        return n

    def isEmpty(self):
        if False:
            print('Hello World!')
        return self.invokingState == -1

    def getSourceInterval(self):
        if False:
            print('Hello World!')
        return INVALID_INTERVAL

    def getRuleContext(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def getPayload(self):
        if False:
            print('Hello World!')
        return self

    def getText(self):
        if False:
            while True:
                i = 10
        if self.getChildCount() == 0:
            return ''
        with StringIO() as builder:
            for child in self.getChildren():
                builder.write(child.getText())
            return builder.getvalue()

    def getRuleIndex(self):
        if False:
            return 10
        return -1

    def getAltNumber(self):
        if False:
            while True:
                i = 10
        return 0

    def setAltNumber(self, altNumber: int):
        if False:
            print('Hello World!')
        pass

    def getChild(self, i: int):
        if False:
            return 10
        return None

    def getChildCount(self):
        if False:
            while True:
                i = 10
        return 0

    def getChildren(self):
        if False:
            for i in range(10):
                print('nop')
        for c in []:
            yield c

    def accept(self, visitor: ParseTreeVisitor):
        if False:
            print('Hello World!')
        return visitor.visitChildren(self)

    def toStringTree(self, ruleNames: list=None, recog: Parser=None):
        if False:
            return 10
        return Trees.toStringTree(self, ruleNames=ruleNames, recog=recog)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.toString(None, None)

    def toString(self, ruleNames: list, stop: RuleContext) -> str:
        if False:
            print('Hello World!')
        with StringIO() as buf:
            p = self
            buf.write('[')
            while p is not None and p is not stop:
                if ruleNames is None:
                    if not p.isEmpty():
                        buf.write(str(p.invokingState))
                else:
                    ri = p.getRuleIndex()
                    ruleName = ruleNames[ri] if ri >= 0 and ri < len(ruleNames) else str(ri)
                    buf.write(ruleName)
                if p.parentCtx is not None and (ruleNames is not None or not p.parentCtx.isEmpty()):
                    buf.write(' ')
                p = p.parentCtx
            buf.write(']')
            return buf.getvalue()
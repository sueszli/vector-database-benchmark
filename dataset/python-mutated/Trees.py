from io import StringIO
from antlr4.Token import Token
from antlr4.Utils import escapeWhitespace
from antlr4.tree.Tree import RuleNode, ErrorNode, TerminalNode, Tree, ParseTree
Parser = None

class Trees(object):

    @classmethod
    def toStringTree(cls, t: Tree, ruleNames: list=None, recog: Parser=None):
        if False:
            i = 10
            return i + 15
        if recog is not None:
            ruleNames = recog.ruleNames
        s = escapeWhitespace(cls.getNodeText(t, ruleNames), False)
        if t.getChildCount() == 0:
            return s
        with StringIO() as buf:
            buf.write('(')
            buf.write(s)
            buf.write(' ')
            for i in range(0, t.getChildCount()):
                if i > 0:
                    buf.write(' ')
                buf.write(cls.toStringTree(t.getChild(i), ruleNames))
            buf.write(')')
            return buf.getvalue()

    @classmethod
    def getNodeText(cls, t: Tree, ruleNames: list=None, recog: Parser=None):
        if False:
            i = 10
            return i + 15
        if recog is not None:
            ruleNames = recog.ruleNames
        if ruleNames is not None:
            if isinstance(t, RuleNode):
                if t.getAltNumber() != 0:
                    return ruleNames[t.getRuleIndex()] + ':' + str(t.getAltNumber())
                return ruleNames[t.getRuleIndex()]
            elif isinstance(t, ErrorNode):
                return str(t)
            elif isinstance(t, TerminalNode):
                if t.symbol is not None:
                    return t.symbol.text
        payload = t.getPayload()
        if isinstance(payload, Token):
            return payload.text
        return str(t.getPayload())

    @classmethod
    def getChildren(cls, t: Tree):
        if False:
            return 10
        return [t.getChild(i) for i in range(0, t.getChildCount())]

    @classmethod
    def getAncestors(cls, t: Tree):
        if False:
            i = 10
            return i + 15
        ancestors = []
        t = t.getParent()
        while t is not None:
            ancestors.insert(0, t)
            t = t.getParent()
        return ancestors

    @classmethod
    def findAllTokenNodes(cls, t: ParseTree, ttype: int):
        if False:
            return 10
        return cls.findAllNodes(t, ttype, True)

    @classmethod
    def findAllRuleNodes(cls, t: ParseTree, ruleIndex: int):
        if False:
            while True:
                i = 10
        return cls.findAllNodes(t, ruleIndex, False)

    @classmethod
    def findAllNodes(cls, t: ParseTree, index: int, findTokens: bool):
        if False:
            print('Hello World!')
        nodes = []
        cls._findAllNodes(t, index, findTokens, nodes)
        return nodes

    @classmethod
    def _findAllNodes(cls, t: ParseTree, index: int, findTokens: bool, nodes: list):
        if False:
            i = 10
            return i + 15
        from antlr4.ParserRuleContext import ParserRuleContext
        if findTokens and isinstance(t, TerminalNode):
            if t.symbol.type == index:
                nodes.append(t)
        elif not findTokens and isinstance(t, ParserRuleContext):
            if t.ruleIndex == index:
                nodes.append(t)
        for i in range(0, t.getChildCount()):
            cls._findAllNodes(t.getChild(i), index, findTokens, nodes)

    @classmethod
    def descendants(cls, t: ParseTree):
        if False:
            while True:
                i = 10
        nodes = [t]
        for i in range(0, t.getChildCount()):
            nodes.extend(cls.descendants(t.getChild(i)))
        return nodes
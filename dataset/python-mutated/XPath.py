from antlr4 import CommonTokenStream, DFA, PredictionContextCache, Lexer, LexerATNSimulator, ParserRuleContext, TerminalNode
from antlr4.InputStream import InputStream
from antlr4.Parser import Parser
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNDeserializer import ATNDeserializer
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import LexerNoViableAltException
from antlr4.tree.Tree import ParseTree
from antlr4.tree.Trees import Trees
from io import StringIO
from antlr4.xpath.XPathLexer import XPathLexer

class XPath(object):
    WILDCARD = '*'
    NOT = '!'

    def __init__(self, parser: Parser, path: str):
        if False:
            return 10
        self.parser = parser
        self.path = path
        self.elements = self.split(path)

    def split(self, path: str):
        if False:
            print('Hello World!')
        input = InputStream(path)
        lexer = XPathLexer(input)

        def recover(self, e):
            if False:
                while True:
                    i = 10
            raise e
        lexer.recover = recover
        lexer.removeErrorListeners()
        lexer.addErrorListener(ErrorListener())
        tokenStream = CommonTokenStream(lexer)
        try:
            tokenStream.fill()
        except LexerNoViableAltException as e:
            pos = lexer.column
            msg = "Invalid tokens or characters at index %d in path '%s'" % (pos, path)
            raise Exception(msg, e)
        tokens = iter(tokenStream.tokens)
        elements = list()
        for el in tokens:
            invert = False
            anywhere = False
            if el.type in [XPathLexer.ROOT, XPathLexer.ANYWHERE]:
                anywhere = el.type == XPathLexer.ANYWHERE
                next_el = next(tokens, None)
                if not next_el:
                    raise Exception('Missing element after %s' % el.getText())
                else:
                    el = next_el
            if el.type == XPathLexer.BANG:
                invert = True
                next_el = next(tokens, None)
                if not next_el:
                    raise Exception('Missing element after %s' % el.getText())
                else:
                    el = next_el
            if el.type in [XPathLexer.TOKEN_REF, XPathLexer.RULE_REF, XPathLexer.WILDCARD, XPathLexer.STRING]:
                element = self.getXPathElement(el, anywhere)
                element.invert = invert
                elements.append(element)
            elif el.type == Token.EOF:
                break
            else:
                raise Exception('Unknown path element %s' % lexer.symbolicNames[el.type])
        return elements

    def getXPathElement(self, wordToken: Token, anywhere: bool):
        if False:
            return 10
        if wordToken.type == Token.EOF:
            raise Exception('Missing path element at end of path')
        word = wordToken.text
        if wordToken.type == XPathLexer.WILDCARD:
            return XPathWildcardAnywhereElement() if anywhere else XPathWildcardElement()
        elif wordToken.type in [XPathLexer.TOKEN_REF, XPathLexer.STRING]:
            tsource = self.parser.getTokenStream().tokenSource
            ttype = Token.INVALID_TYPE
            if wordToken.type == XPathLexer.TOKEN_REF:
                if word in tsource.ruleNames:
                    ttype = tsource.ruleNames.index(word) + 1
            elif word in tsource.literalNames:
                ttype = tsource.literalNames.index(word)
            if ttype == Token.INVALID_TYPE:
                raise Exception("%s at index %d isn't a valid token name" % (word, wordToken.tokenIndex))
            return XPathTokenAnywhereElement(word, ttype) if anywhere else XPathTokenElement(word, ttype)
        else:
            ruleIndex = self.parser.ruleNames.index(word) if word in self.parser.ruleNames else -1
            if ruleIndex == -1:
                raise Exception("%s at index %d isn't a valid rule name" % (word, wordToken.tokenIndex))
            return XPathRuleAnywhereElement(word, ruleIndex) if anywhere else XPathRuleElement(word, ruleIndex)

    @staticmethod
    def findAll(tree: ParseTree, xpath: str, parser: Parser):
        if False:
            return 10
        p = XPath(parser, xpath)
        return p.evaluate(tree)

    def evaluate(self, t: ParseTree):
        if False:
            for i in range(10):
                print('nop')
        dummyRoot = ParserRuleContext()
        dummyRoot.children = [t]
        work = [dummyRoot]
        for element in self.elements:
            work_next = list()
            for node in work:
                if not isinstance(node, TerminalNode) and node.children:
                    matching = element.evaluate(node)
                    matching = filter(lambda m: m not in work_next, matching)
                    work_next.extend(matching)
            work = work_next
        return work

class XPathElement(object):

    def __init__(self, nodeName: str):
        if False:
            for i in range(10):
                print('nop')
        self.nodeName = nodeName
        self.invert = False

    def __str__(self):
        if False:
            while True:
                i = 10
        return type(self).__name__ + '[' + ('!' if self.invert else '') + self.nodeName + ']'

class XPathRuleAnywhereElement(XPathElement):

    def __init__(self, ruleName: str, ruleIndex: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(ruleName)
        self.ruleIndex = ruleIndex

    def evaluate(self, t: ParseTree):
        if False:
            while True:
                i = 10
        return filter(lambda c: isinstance(c, ParserRuleContext) and self.invert ^ (c.getRuleIndex() == self.ruleIndex), Trees.descendants(t))

class XPathRuleElement(XPathElement):

    def __init__(self, ruleName: str, ruleIndex: int):
        if False:
            while True:
                i = 10
        super().__init__(ruleName)
        self.ruleIndex = ruleIndex

    def evaluate(self, t: ParseTree):
        if False:
            while True:
                i = 10
        return filter(lambda c: isinstance(c, ParserRuleContext) and self.invert ^ (c.getRuleIndex() == self.ruleIndex), Trees.getChildren(t))

class XPathTokenAnywhereElement(XPathElement):

    def __init__(self, ruleName: str, tokenType: int):
        if False:
            print('Hello World!')
        super().__init__(ruleName)
        self.tokenType = tokenType

    def evaluate(self, t: ParseTree):
        if False:
            return 10
        return filter(lambda c: isinstance(c, TerminalNode) and self.invert ^ (c.symbol.type == self.tokenType), Trees.descendants(t))

class XPathTokenElement(XPathElement):

    def __init__(self, ruleName: str, tokenType: int):
        if False:
            while True:
                i = 10
        super().__init__(ruleName)
        self.tokenType = tokenType

    def evaluate(self, t: ParseTree):
        if False:
            return 10
        return filter(lambda c: isinstance(c, TerminalNode) and self.invert ^ (c.symbol.type == self.tokenType), Trees.getChildren(t))

class XPathWildcardAnywhereElement(XPathElement):

    def __init__(self):
        if False:
            return 10
        super().__init__(XPath.WILDCARD)

    def evaluate(self, t: ParseTree):
        if False:
            i = 10
            return i + 15
        if self.invert:
            return list()
        else:
            return Trees.descendants(t)

class XPathWildcardElement(XPathElement):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(XPath.WILDCARD)

    def evaluate(self, t: ParseTree):
        if False:
            print('Hello World!')
        if self.invert:
            return list()
        else:
            return Trees.getChildren(t)
from antlr4.tree.ParseTreePatternMatcher import ParseTreePatternMatcher
from antlr4.tree.Tree import ParseTree
from antlr4.xpath.XPathLexer import XPathLexer

class ParseTreePattern(object):
    __slots__ = ('matcher', 'patternRuleIndex', 'pattern', 'patternTree')

    def __init__(self, matcher: ParseTreePatternMatcher, pattern: str, patternRuleIndex: int, patternTree: ParseTree):
        if False:
            for i in range(10):
                print('nop')
        self.matcher = matcher
        self.patternRuleIndex = patternRuleIndex
        self.pattern = pattern
        self.patternTree = patternTree

    def match(self, tree: ParseTree):
        if False:
            while True:
                i = 10
        return self.matcher.match(tree, self)

    def matches(self, tree: ParseTree):
        if False:
            while True:
                i = 10
        return self.matcher.match(tree, self).succeeded()

    def findAll(self, tree: ParseTree, xpath: str):
        if False:
            for i in range(10):
                print('nop')
        subtrees = XPath.findAll(tree, xpath, self.matcher.parser)
        matches = list()
        for t in subtrees:
            match = self.match(t)
            if match.succeeded():
                matches.append(match)
        return matches
from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Lexer import Lexer
from antlr4.ListTokenSource import ListTokenSource
from antlr4.Token import Token
from antlr4.error.ErrorStrategy import BailErrorStrategy
from antlr4.error.Errors import RecognitionException, ParseCancellationException
from antlr4.tree.Chunk import TagChunk, TextChunk
from antlr4.tree.RuleTagToken import RuleTagToken
from antlr4.tree.TokenTagToken import TokenTagToken
from antlr4.tree.Tree import ParseTree, TerminalNode, RuleNode
Parser = None
ParseTreePattern = None

class CannotInvokeStartRule(Exception):

    def __init__(self, e: Exception):
        if False:
            while True:
                i = 10
        super().__init__(e)

class StartRuleDoesNotConsumeFullPattern(Exception):
    pass

class ParseTreePatternMatcher(object):
    __slots__ = ('lexer', 'parser', 'start', 'stop', 'escape')

    def __init__(self, lexer: Lexer, parser: Parser):
        if False:
            i = 10
            return i + 15
        self.lexer = lexer
        self.parser = parser
        self.start = '<'
        self.stop = '>'
        self.escape = '\\'

    def setDelimiters(self, start: str, stop: str, escapeLeft: str):
        if False:
            i = 10
            return i + 15
        if start is None or len(start) == 0:
            raise Exception('start cannot be null or empty')
        if stop is None or len(stop) == 0:
            raise Exception('stop cannot be null or empty')
        self.start = start
        self.stop = stop
        self.escape = escapeLeft

    def matchesRuleIndex(self, tree: ParseTree, pattern: str, patternRuleIndex: int):
        if False:
            while True:
                i = 10
        p = self.compileTreePattern(pattern, patternRuleIndex)
        return self.matches(tree, p)

    def matchesPattern(self, tree: ParseTree, pattern: ParseTreePattern):
        if False:
            for i in range(10):
                print('nop')
        mismatchedNode = self.matchImpl(tree, pattern.patternTree, dict())
        return mismatchedNode is None

    def matchRuleIndex(self, tree: ParseTree, pattern: str, patternRuleIndex: int):
        if False:
            return 10
        p = self.compileTreePattern(pattern, patternRuleIndex)
        return self.matchPattern(tree, p)

    def matchPattern(self, tree: ParseTree, pattern: ParseTreePattern):
        if False:
            for i in range(10):
                print('nop')
        labels = dict()
        mismatchedNode = self.matchImpl(tree, pattern.patternTree, labels)
        from antlr4.tree.ParseTreeMatch import ParseTreeMatch
        return ParseTreeMatch(tree, pattern, labels, mismatchedNode)

    def compileTreePattern(self, pattern: str, patternRuleIndex: int):
        if False:
            print('Hello World!')
        tokenList = self.tokenize(pattern)
        tokenSrc = ListTokenSource(tokenList)
        tokens = CommonTokenStream(tokenSrc)
        from antlr4.ParserInterpreter import ParserInterpreter
        parserInterp = ParserInterpreter(self.parser.grammarFileName, self.parser.tokenNames, self.parser.ruleNames, self.parser.getATNWithBypassAlts(), tokens)
        tree = None
        try:
            parserInterp.setErrorHandler(BailErrorStrategy())
            tree = parserInterp.parse(patternRuleIndex)
        except ParseCancellationException as e:
            raise e.cause
        except RecognitionException as e:
            raise e
        except Exception as e:
            raise CannotInvokeStartRule(e)
        if tokens.LA(1) != Token.EOF:
            raise StartRuleDoesNotConsumeFullPattern()
        from antlr4.tree.ParseTreePattern import ParseTreePattern
        return ParseTreePattern(self, pattern, patternRuleIndex, tree)

    def matchImpl(self, tree: ParseTree, patternTree: ParseTree, labels: dict):
        if False:
            while True:
                i = 10
        if tree is None:
            raise Exception('tree cannot be null')
        if patternTree is None:
            raise Exception('patternTree cannot be null')
        if isinstance(tree, TerminalNode) and isinstance(patternTree, TerminalNode):
            mismatchedNode = None
            if tree.symbol.type == patternTree.symbol.type:
                if isinstance(patternTree.symbol, TokenTagToken):
                    tokenTagToken = patternTree.symbol
                    self.map(labels, tokenTagToken.tokenName, tree)
                    if tokenTagToken.label is not None:
                        self.map(labels, tokenTagToken.label, tree)
                elif tree.getText() == patternTree.getText():
                    pass
                elif mismatchedNode is None:
                    mismatchedNode = tree
            elif mismatchedNode is None:
                mismatchedNode = tree
            return mismatchedNode
        if isinstance(tree, ParserRuleContext) and isinstance(patternTree, ParserRuleContext):
            mismatchedNode = None
            ruleTagToken = self.getRuleTagToken(patternTree)
            if ruleTagToken is not None:
                m = None
                if tree.ruleContext.ruleIndex == patternTree.ruleContext.ruleIndex:
                    self.map(labels, ruleTagToken.ruleName, tree)
                    if ruleTagToken.label is not None:
                        self.map(labels, ruleTagToken.label, tree)
                elif mismatchedNode is None:
                    mismatchedNode = tree
                return mismatchedNode
            if tree.getChildCount() != patternTree.getChildCount():
                if mismatchedNode is None:
                    mismatchedNode = tree
                return mismatchedNode
            n = tree.getChildCount()
            for i in range(0, n):
                childMatch = self.matchImpl(tree.getChild(i), patternTree.getChild(i), labels)
                if childMatch is not None:
                    return childMatch
            return mismatchedNode
        return tree

    def map(self, labels, label, tree):
        if False:
            print('Hello World!')
        v = labels.get(label, None)
        if v is None:
            v = list()
            labels[label] = v
        v.append(tree)

    def getRuleTagToken(self, tree: ParseTree):
        if False:
            while True:
                i = 10
        if isinstance(tree, RuleNode):
            if tree.getChildCount() == 1 and isinstance(tree.getChild(0), TerminalNode):
                c = tree.getChild(0)
                if isinstance(c.symbol, RuleTagToken):
                    return c.symbol
        return None

    def tokenize(self, pattern: str):
        if False:
            return 10
        chunks = self.split(pattern)
        tokens = list()
        for chunk in chunks:
            if isinstance(chunk, TagChunk):
                if chunk.tag[0].isupper():
                    ttype = self.parser.getTokenType(chunk.tag)
                    if ttype == Token.INVALID_TYPE:
                        raise Exception('Unknown token ' + str(chunk.tag) + ' in pattern: ' + pattern)
                    tokens.append(TokenTagToken(chunk.tag, ttype, chunk.label))
                elif chunk.tag[0].islower():
                    ruleIndex = self.parser.getRuleIndex(chunk.tag)
                    if ruleIndex == -1:
                        raise Exception('Unknown rule ' + str(chunk.tag) + ' in pattern: ' + pattern)
                    ruleImaginaryTokenType = self.parser.getATNWithBypassAlts().ruleToTokenType[ruleIndex]
                    tokens.append(RuleTagToken(chunk.tag, ruleImaginaryTokenType, chunk.label))
                else:
                    raise Exception('invalid tag: ' + str(chunk.tag) + ' in pattern: ' + pattern)
            else:
                self.lexer.setInputStream(InputStream(chunk.text))
                t = self.lexer.nextToken()
                while t.type != Token.EOF:
                    tokens.append(t)
                    t = self.lexer.nextToken()
        return tokens

    def split(self, pattern: str):
        if False:
            i = 10
            return i + 15
        p = 0
        n = len(pattern)
        chunks = list()
        starts = list()
        stops = list()
        while p < n:
            if p == pattern.find(self.escape + self.start, p):
                p += len(self.escape) + len(self.start)
            elif p == pattern.find(self.escape + self.stop, p):
                p += len(self.escape) + len(self.stop)
            elif p == pattern.find(self.start, p):
                starts.append(p)
                p += len(self.start)
            elif p == pattern.find(self.stop, p):
                stops.append(p)
                p += len(self.stop)
            else:
                p += 1
        nt = len(starts)
        if nt > len(stops):
            raise Exception('unterminated tag in pattern: ' + pattern)
        if nt < len(stops):
            raise Exception('missing start tag in pattern: ' + pattern)
        for i in range(0, nt):
            if starts[i] >= stops[i]:
                raise Exception('tag delimiters out of order in pattern: ' + pattern)
        if nt == 0:
            chunks.append(TextChunk(pattern))
        if nt > 0 and starts[0] > 0:
            text = pattern[0:starts[0]]
            chunks.add(TextChunk(text))
        for i in range(0, nt):
            tag = pattern[starts[i] + len(self.start):stops[i]]
            ruleOrToken = tag
            label = None
            colon = tag.find(':')
            if colon >= 0:
                label = tag[0:colon]
                ruleOrToken = tag[colon + 1:len(tag)]
            chunks.append(TagChunk(label, ruleOrToken))
            if i + 1 < len(starts):
                text = pattern[stops[i] + len(self.stop):starts[i + 1]]
                chunks.append(TextChunk(text))
        if nt > 0:
            afterLastTag = stops[nt - 1] + len(self.stop)
            if afterLastTag < n:
                text = pattern[afterLastTag:n]
                chunks.append(TextChunk(text))
        for i in range(0, len(chunks)):
            c = chunks[i]
            if isinstance(c, TextChunk):
                unescaped = c.text.replace(self.escape, '')
                if len(unescaped) < len(c.text):
                    chunks[i] = TextChunk(unescaped)
        return chunks
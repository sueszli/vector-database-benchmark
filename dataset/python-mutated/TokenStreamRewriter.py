from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream

class TokenStreamRewriter(object):
    __slots__ = ('tokens', 'programs', 'lastRewriteTokenIndexes')
    DEFAULT_PROGRAM_NAME = 'default'
    PROGRAM_INIT_SIZE = 100
    MIN_TOKEN_INDEX = 0

    def __init__(self, tokens):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type  tokens: antlr4.BufferedTokenStream.BufferedTokenStream\n        :param tokens:\n        :return:\n        '
        super(TokenStreamRewriter, self).__init__()
        self.tokens = tokens
        self.programs = {self.DEFAULT_PROGRAM_NAME: []}
        self.lastRewriteTokenIndexes = {}

    def getTokenStream(self):
        if False:
            i = 10
            return i + 15
        return self.tokens

    def rollback(self, instruction_index, program_name):
        if False:
            while True:
                i = 10
        ins = self.programs.get(program_name, None)
        if ins:
            self.programs[program_name] = ins[self.MIN_TOKEN_INDEX:instruction_index]

    def deleteProgram(self, program_name=DEFAULT_PROGRAM_NAME):
        if False:
            i = 10
            return i + 15
        self.rollback(self.MIN_TOKEN_INDEX, program_name)

    def insertAfterToken(self, token, text, program_name=DEFAULT_PROGRAM_NAME):
        if False:
            i = 10
            return i + 15
        self.insertAfter(token.tokenIndex, text, program_name)

    def insertAfter(self, index, text, program_name=DEFAULT_PROGRAM_NAME):
        if False:
            print('Hello World!')
        op = self.InsertAfterOp(self.tokens, index + 1, text)
        rewrites = self.getProgram(program_name)
        op.instructionIndex = len(rewrites)
        rewrites.append(op)

    def insertBeforeIndex(self, index, text):
        if False:
            while True:
                i = 10
        self.insertBefore(self.DEFAULT_PROGRAM_NAME, index, text)

    def insertBeforeToken(self, token, text, program_name=DEFAULT_PROGRAM_NAME):
        if False:
            while True:
                i = 10
        self.insertBefore(program_name, token.tokenIndex, text)

    def insertBefore(self, program_name, index, text):
        if False:
            i = 10
            return i + 15
        op = self.InsertBeforeOp(self.tokens, index, text)
        rewrites = self.getProgram(program_name)
        op.instructionIndex = len(rewrites)
        rewrites.append(op)

    def replaceIndex(self, index, text):
        if False:
            for i in range(10):
                print('nop')
        self.replace(self.DEFAULT_PROGRAM_NAME, index, index, text)

    def replaceRange(self, from_idx, to_idx, text):
        if False:
            while True:
                i = 10
        self.replace(self.DEFAULT_PROGRAM_NAME, from_idx, to_idx, text)

    def replaceSingleToken(self, token, text):
        if False:
            return 10
        self.replace(self.DEFAULT_PROGRAM_NAME, token.tokenIndex, token.tokenIndex, text)

    def replaceRangeTokens(self, from_token, to_token, text, program_name=DEFAULT_PROGRAM_NAME):
        if False:
            while True:
                i = 10
        self.replace(program_name, from_token.tokenIndex, to_token.tokenIndex, text)

    def replace(self, program_name, from_idx, to_idx, text):
        if False:
            while True:
                i = 10
        if any((from_idx > to_idx, from_idx < 0, to_idx < 0, to_idx >= len(self.tokens.tokens))):
            raise ValueError('replace: range invalid: {}..{}(size={})'.format(from_idx, to_idx, len(self.tokens.tokens)))
        op = self.ReplaceOp(from_idx, to_idx, self.tokens, text)
        rewrites = self.getProgram(program_name)
        op.instructionIndex = len(rewrites)
        rewrites.append(op)

    def deleteToken(self, token):
        if False:
            print('Hello World!')
        self.delete(self.DEFAULT_PROGRAM_NAME, token, token)

    def deleteIndex(self, index):
        if False:
            while True:
                i = 10
        self.delete(self.DEFAULT_PROGRAM_NAME, index, index)

    def delete(self, program_name, from_idx, to_idx):
        if False:
            return 10
        if isinstance(from_idx, Token):
            self.replace(program_name, from_idx.tokenIndex, to_idx.tokenIndex, '')
        else:
            self.replace(program_name, from_idx, to_idx, '')

    def lastRewriteTokenIndex(self, program_name=DEFAULT_PROGRAM_NAME):
        if False:
            print('Hello World!')
        return self.lastRewriteTokenIndexes.get(program_name, -1)

    def setLastRewriteTokenIndex(self, program_name, i):
        if False:
            return 10
        self.lastRewriteTokenIndexes[program_name] = i

    def getProgram(self, program_name):
        if False:
            while True:
                i = 10
        return self.programs.setdefault(program_name, [])

    def getDefaultText(self):
        if False:
            while True:
                i = 10
        return self.getText(self.DEFAULT_PROGRAM_NAME, 0, len(self.tokens.tokens) - 1)

    def getText(self, program_name, start: int, stop: int):
        if False:
            print('Hello World!')
        '\n        :return: the text in tokens[start, stop](closed interval)\n        '
        rewrites = self.programs.get(program_name)
        if stop > len(self.tokens.tokens) - 1:
            stop = len(self.tokens.tokens) - 1
        if start < 0:
            start = 0
        if not rewrites:
            return self.tokens.getText(start, stop)
        buf = StringIO()
        indexToOp = self._reduceToSingleOperationPerIndex(rewrites)
        i = start
        while all((i <= stop, i < len(self.tokens.tokens))):
            op = indexToOp.pop(i, None)
            token = self.tokens.get(i)
            if op is None:
                if token.type != Token.EOF:
                    buf.write(token.text)
                i += 1
            else:
                i = op.execute(buf)
        if stop == len(self.tokens.tokens) - 1:
            for op in indexToOp.values():
                if op.index >= len(self.tokens.tokens) - 1:
                    buf.write(op.text)
        return buf.getvalue()

    def _reduceToSingleOperationPerIndex(self, rewrites):
        if False:
            print('Hello World!')
        for (i, rop) in enumerate(rewrites):
            if any((rop is None, not isinstance(rop, TokenStreamRewriter.ReplaceOp))):
                continue
            inserts = [op for op in rewrites[:i] if isinstance(op, TokenStreamRewriter.InsertBeforeOp)]
            for iop in inserts:
                if iop.index == rop.index:
                    rewrites[iop.instructionIndex] = None
                    rop.text = '{}{}'.format(iop.text, rop.text)
                elif all((iop.index > rop.index, iop.index <= rop.last_index)):
                    rewrites[iop.instructionIndex] = None
            prevReplaces = [op for op in rewrites[:i] if isinstance(op, TokenStreamRewriter.ReplaceOp)]
            for prevRop in prevReplaces:
                if all((prevRop.index >= rop.index, prevRop.last_index <= rop.last_index)):
                    rewrites[prevRop.instructionIndex] = None
                    continue
                isDisjoint = any((prevRop.last_index < rop.index, prevRop.index > rop.last_index))
                if all((prevRop.text is None, rop.text is None, not isDisjoint)):
                    rewrites[prevRop.instructionIndex] = None
                    rop.index = min(prevRop.index, rop.index)
                    rop.last_index = min(prevRop.last_index, rop.last_index)
                    print('New rop {}'.format(rop))
                elif not isDisjoint:
                    raise ValueError('replace op boundaries of {} overlap with previous {}'.format(rop, prevRop))
        for (i, iop) in enumerate(rewrites):
            if any((iop is None, not isinstance(iop, TokenStreamRewriter.InsertBeforeOp))):
                continue
            prevInserts = [op for op in rewrites[:i] if isinstance(op, TokenStreamRewriter.InsertBeforeOp)]
            for (prev_index, prevIop) in enumerate(prevInserts):
                if prevIop.index == iop.index and type(prevIop) is TokenStreamRewriter.InsertBeforeOp:
                    iop.text += prevIop.text
                    rewrites[prev_index] = None
                elif prevIop.index == iop.index and type(prevIop) is TokenStreamRewriter.InsertAfterOp:
                    iop.text = prevIop.text + iop.text
                    rewrites[prev_index] = None
            prevReplaces = [op for op in rewrites[:i] if isinstance(op, TokenStreamRewriter.ReplaceOp)]
            for rop in prevReplaces:
                if iop.index == rop.index:
                    rop.text = iop.text + rop.text
                    rewrites[i] = None
                    continue
                if all((iop.index >= rop.index, iop.index <= rop.last_index)):
                    raise ValueError('insert op {} within boundaries of previous {}'.format(iop, rop))
        reduced = {}
        for (i, op) in enumerate(rewrites):
            if op is None:
                continue
            if reduced.get(op.index):
                raise ValueError('should be only one op per index')
            reduced[op.index] = op
        return reduced

    class RewriteOperation(object):
        __slots__ = ('tokens', 'index', 'text', 'instructionIndex')

        def __init__(self, tokens, index, text=''):
            if False:
                for i in range(10):
                    print('nop')
            '\n            :type tokens: CommonTokenStream\n            :param tokens:\n            :param index:\n            :param text:\n            :return:\n            '
            self.tokens = tokens
            self.index = index
            self.text = text
            self.instructionIndex = 0

        def execute(self, buf):
            if False:
                print('Hello World!')
            '\n            :type buf: StringIO.StringIO\n            :param buf:\n            :return:\n            '
            return self.index

        def __str__(self):
            if False:
                i = 10
                return i + 15
            return '<{}@{}:"{}">'.format(self.__class__.__name__, self.tokens.get(self.index), self.text)

    class InsertBeforeOp(RewriteOperation):

        def __init__(self, tokens, index, text=''):
            if False:
                for i in range(10):
                    print('nop')
            super(TokenStreamRewriter.InsertBeforeOp, self).__init__(tokens, index, text)

        def execute(self, buf):
            if False:
                i = 10
                return i + 15
            buf.write(self.text)
            if self.tokens.get(self.index).type != Token.EOF:
                buf.write(self.tokens.get(self.index).text)
            return self.index + 1

    class InsertAfterOp(InsertBeforeOp):
        pass

    class ReplaceOp(RewriteOperation):
        __slots__ = 'last_index'

        def __init__(self, from_idx, to_idx, tokens, text):
            if False:
                return 10
            super(TokenStreamRewriter.ReplaceOp, self).__init__(tokens, from_idx, text)
            self.last_index = to_idx

        def execute(self, buf):
            if False:
                return 10
            if self.text:
                buf.write(self.text)
            return self.last_index + 1

        def __str__(self):
            if False:
                return 10
            if self.text:
                return '<ReplaceOp@{}..{}:"{}">'.format(self.tokens.get(self.index), self.tokens.get(self.last_index), self.text)
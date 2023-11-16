from antlr4.InputStream import InputStream
from antlr4.atn.LexerAction import LexerAction, LexerIndexedCustomAction
Lexer = None
LexerActionExecutor = None

class LexerActionExecutor(object):
    __slots__ = ('lexerActions', 'hashCode')

    def __init__(self, lexerActions: list=list()):
        if False:
            for i in range(10):
                print('nop')
        self.lexerActions = lexerActions
        self.hashCode = hash(''.join([str(la) for la in lexerActions]))

    @staticmethod
    def append(lexerActionExecutor: LexerActionExecutor, lexerAction: LexerAction):
        if False:
            for i in range(10):
                print('nop')
        if lexerActionExecutor is None:
            return LexerActionExecutor([lexerAction])
        lexerActions = lexerActionExecutor.lexerActions + [lexerAction]
        return LexerActionExecutor(lexerActions)

    def fixOffsetBeforeMatch(self, offset: int):
        if False:
            i = 10
            return i + 15
        updatedLexerActions = None
        for i in range(0, len(self.lexerActions)):
            if self.lexerActions[i].isPositionDependent and (not isinstance(self.lexerActions[i], LexerIndexedCustomAction)):
                if updatedLexerActions is None:
                    updatedLexerActions = [la for la in self.lexerActions]
                updatedLexerActions[i] = LexerIndexedCustomAction(offset, self.lexerActions[i])
        if updatedLexerActions is None:
            return self
        else:
            return LexerActionExecutor(updatedLexerActions)

    def execute(self, lexer: Lexer, input: InputStream, startIndex: int):
        if False:
            i = 10
            return i + 15
        requiresSeek = False
        stopIndex = input.index
        try:
            for lexerAction in self.lexerActions:
                if isinstance(lexerAction, LexerIndexedCustomAction):
                    offset = lexerAction.offset
                    input.seek(startIndex + offset)
                    lexerAction = lexerAction.action
                    requiresSeek = startIndex + offset != stopIndex
                elif lexerAction.isPositionDependent:
                    input.seek(stopIndex)
                    requiresSeek = False
                lexerAction.execute(lexer)
        finally:
            if requiresSeek:
                input.seek(stopIndex)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return self.hashCode

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if self is other:
            return True
        elif not isinstance(other, LexerActionExecutor):
            return False
        else:
            return self.hashCode == other.hashCode and self.lexerActions == other.lexerActions
del Lexer
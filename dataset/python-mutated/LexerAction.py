from enum import IntEnum
Lexer = None

class LexerActionType(IntEnum):
    CHANNEL = 0
    CUSTOM = 1
    MODE = 2
    MORE = 3
    POP_MODE = 4
    PUSH_MODE = 5
    SKIP = 6
    TYPE = 7

class LexerAction(object):
    __slots__ = ('actionType', 'isPositionDependent')

    def __init__(self, action: LexerActionType):
        if False:
            for i in range(10):
                print('nop')
        self.actionType = action
        self.isPositionDependent = False

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.actionType)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self is other

class LexerSkipAction(LexerAction):
    INSTANCE = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(LexerActionType.SKIP)

    def execute(self, lexer: Lexer):
        if False:
            return 10
        lexer.skip()

    def __str__(self):
        if False:
            return 10
        return 'skip'
LexerSkipAction.INSTANCE = LexerSkipAction()

class LexerTypeAction(LexerAction):
    __slots__ = 'type'

    def __init__(self, type: int):
        if False:
            return 10
        super().__init__(LexerActionType.TYPE)
        self.type = type

    def execute(self, lexer: Lexer):
        if False:
            for i in range(10):
                print('nop')
        lexer.type = self.type

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((self.actionType, self.type))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self is other:
            return True
        elif not isinstance(other, LexerTypeAction):
            return False
        else:
            return self.type == other.type

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'type(' + str(self.type) + ')'

class LexerPushModeAction(LexerAction):
    __slots__ = 'mode'

    def __init__(self, mode: int):
        if False:
            while True:
                i = 10
        super().__init__(LexerActionType.PUSH_MODE)
        self.mode = mode

    def execute(self, lexer: Lexer):
        if False:
            i = 10
            return i + 15
        lexer.pushMode(self.mode)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.actionType, self.mode))

    def __eq__(self, other):
        if False:
            return 10
        if self is other:
            return True
        elif not isinstance(other, LexerPushModeAction):
            return False
        else:
            return self.mode == other.mode

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'pushMode(' + str(self.mode) + ')'

class LexerPopModeAction(LexerAction):
    INSTANCE = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(LexerActionType.POP_MODE)

    def execute(self, lexer: Lexer):
        if False:
            print('Hello World!')
        lexer.popMode()

    def __str__(self):
        if False:
            print('Hello World!')
        return 'popMode'
LexerPopModeAction.INSTANCE = LexerPopModeAction()

class LexerMoreAction(LexerAction):
    INSTANCE = None

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(LexerActionType.MORE)

    def execute(self, lexer: Lexer):
        if False:
            while True:
                i = 10
        lexer.more()

    def __str__(self):
        if False:
            return 10
        return 'more'
LexerMoreAction.INSTANCE = LexerMoreAction()

class LexerModeAction(LexerAction):
    __slots__ = 'mode'

    def __init__(self, mode: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(LexerActionType.MODE)
        self.mode = mode

    def execute(self, lexer: Lexer):
        if False:
            i = 10
            return i + 15
        lexer.mode(self.mode)

    def __hash__(self):
        if False:
            return 10
        return hash((self.actionType, self.mode))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self is other:
            return True
        elif not isinstance(other, LexerModeAction):
            return False
        else:
            return self.mode == other.mode

    def __str__(self):
        if False:
            print('Hello World!')
        return 'mode(' + str(self.mode) + ')'

class LexerCustomAction(LexerAction):
    __slots__ = ('ruleIndex', 'actionIndex')

    def __init__(self, ruleIndex: int, actionIndex: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(LexerActionType.CUSTOM)
        self.ruleIndex = ruleIndex
        self.actionIndex = actionIndex
        self.isPositionDependent = True

    def execute(self, lexer: Lexer):
        if False:
            return 10
        lexer.action(None, self.ruleIndex, self.actionIndex)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((self.actionType, self.ruleIndex, self.actionIndex))

    def __eq__(self, other):
        if False:
            return 10
        if self is other:
            return True
        elif not isinstance(other, LexerCustomAction):
            return False
        else:
            return self.ruleIndex == other.ruleIndex and self.actionIndex == other.actionIndex

class LexerChannelAction(LexerAction):
    __slots__ = 'channel'

    def __init__(self, channel: int):
        if False:
            while True:
                i = 10
        super().__init__(LexerActionType.CHANNEL)
        self.channel = channel

    def execute(self, lexer: Lexer):
        if False:
            return 10
        lexer._channel = self.channel

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.actionType, self.channel))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self is other:
            return True
        elif not isinstance(other, LexerChannelAction):
            return False
        else:
            return self.channel == other.channel

    def __str__(self):
        if False:
            return 10
        return 'channel(' + str(self.channel) + ')'

class LexerIndexedCustomAction(LexerAction):
    __slots__ = ('offset', 'action')

    def __init__(self, offset: int, action: LexerAction):
        if False:
            return 10
        super().__init__(action.actionType)
        self.offset = offset
        self.action = action
        self.isPositionDependent = True

    def execute(self, lexer: Lexer):
        if False:
            i = 10
            return i + 15
        self.action.execute(lexer)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.actionType, self.offset, self.action))

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if self is other:
            return True
        elif not isinstance(other, LexerIndexedCustomAction):
            return False
        else:
            return self.offset == other.offset and self.action == other.action
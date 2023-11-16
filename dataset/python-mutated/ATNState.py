from antlr4.atn.Transition import Transition
INITIAL_NUM_TRANSITIONS = 4

class ATNState(object):
    __slots__ = ('atn', 'stateNumber', 'stateType', 'ruleIndex', 'epsilonOnlyTransitions', 'transitions', 'nextTokenWithinRule')
    INVALID_TYPE = 0
    BASIC = 1
    RULE_START = 2
    BLOCK_START = 3
    PLUS_BLOCK_START = 4
    STAR_BLOCK_START = 5
    TOKEN_START = 6
    RULE_STOP = 7
    BLOCK_END = 8
    STAR_LOOP_BACK = 9
    STAR_LOOP_ENTRY = 10
    PLUS_LOOP_BACK = 11
    LOOP_END = 12
    serializationNames = ['INVALID', 'BASIC', 'RULE_START', 'BLOCK_START', 'PLUS_BLOCK_START', 'STAR_BLOCK_START', 'TOKEN_START', 'RULE_STOP', 'BLOCK_END', 'STAR_LOOP_BACK', 'STAR_LOOP_ENTRY', 'PLUS_LOOP_BACK', 'LOOP_END']
    INVALID_STATE_NUMBER = -1

    def __init__(self):
        if False:
            while True:
                i = 10
        self.atn = None
        self.stateNumber = ATNState.INVALID_STATE_NUMBER
        self.stateType = None
        self.ruleIndex = 0
        self.epsilonOnlyTransitions = False
        self.transitions = []
        self.nextTokenWithinRule = None

    def __hash__(self):
        if False:
            while True:
                i = 10
        return self.stateNumber

    def __eq__(self, other):
        if False:
            return 10
        return isinstance(other, ATNState) and self.stateNumber == other.stateNumber

    def onlyHasEpsilonTransitions(self):
        if False:
            while True:
                i = 10
        return self.epsilonOnlyTransitions

    def isNonGreedyExitState(self):
        if False:
            return 10
        return False

    def __str__(self):
        if False:
            return 10
        return str(self.stateNumber)

    def addTransition(self, trans: Transition, index: int=-1):
        if False:
            i = 10
            return i + 15
        if len(self.transitions) == 0:
            self.epsilonOnlyTransitions = trans.isEpsilon
        elif self.epsilonOnlyTransitions != trans.isEpsilon:
            self.epsilonOnlyTransitions = False
        if index == -1:
            self.transitions.append(trans)
        else:
            self.transitions.insert(index, trans)

class BasicState(ATNState):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.stateType = self.BASIC

class DecisionState(ATNState):
    __slots__ = ('decision', 'nonGreedy')

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.decision = -1
        self.nonGreedy = False

class BlockStartState(DecisionState):
    __slots__ = 'endState'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.endState = None

class BasicBlockStartState(BlockStartState):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.stateType = self.BLOCK_START

class BlockEndState(ATNState):
    __slots__ = 'startState'

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.stateType = self.BLOCK_END
        self.startState = None

class RuleStopState(ATNState):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.stateType = self.RULE_STOP

class RuleStartState(ATNState):
    __slots__ = ('stopState', 'isPrecedenceRule')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.stateType = self.RULE_START
        self.stopState = None
        self.isPrecedenceRule = False

class PlusLoopbackState(DecisionState):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.stateType = self.PLUS_LOOP_BACK

class PlusBlockStartState(BlockStartState):
    __slots__ = 'loopBackState'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.stateType = self.PLUS_BLOCK_START
        self.loopBackState = None

class StarBlockStartState(BlockStartState):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.stateType = self.STAR_BLOCK_START

class StarLoopbackState(ATNState):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.stateType = self.STAR_LOOP_BACK

class StarLoopEntryState(DecisionState):
    __slots__ = ('loopBackState', 'isPrecedenceDecision')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.stateType = self.STAR_LOOP_ENTRY
        self.loopBackState = None
        self.isPrecedenceDecision = None

class LoopEndState(ATNState):
    __slots__ = 'loopBackState'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.stateType = self.LOOP_END
        self.loopBackState = None

class TokensStartState(DecisionState):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.stateType = self.TOKEN_START
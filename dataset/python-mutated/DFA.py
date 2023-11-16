from antlr4.atn.ATNState import StarLoopEntryState
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import DecisionState
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import IllegalStateException

class DFA(object):
    __slots__ = ('atnStartState', 'decision', '_states', 's0', 'precedenceDfa')

    def __init__(self, atnStartState: DecisionState, decision: int=0):
        if False:
            return 10
        self.atnStartState = atnStartState
        self.decision = decision
        self._states = dict()
        self.s0 = None
        self.precedenceDfa = False
        if isinstance(atnStartState, StarLoopEntryState):
            if atnStartState.isPrecedenceDecision:
                self.precedenceDfa = True
                precedenceState = DFAState(configs=ATNConfigSet())
                precedenceState.edges = []
                precedenceState.isAcceptState = False
                precedenceState.requiresFullContext = False
                self.s0 = precedenceState

    def getPrecedenceStartState(self, precedence: int):
        if False:
            return 10
        if not self.precedenceDfa:
            raise IllegalStateException('Only precedence DFAs may contain a precedence start state.')
        if precedence < 0 or precedence >= len(self.s0.edges):
            return None
        return self.s0.edges[precedence]

    def setPrecedenceStartState(self, precedence: int, startState: DFAState):
        if False:
            print('Hello World!')
        if not self.precedenceDfa:
            raise IllegalStateException('Only precedence DFAs may contain a precedence start state.')
        if precedence < 0:
            return
        if precedence >= len(self.s0.edges):
            ext = [None] * (precedence + 1 - len(self.s0.edges))
            self.s0.edges.extend(ext)
        self.s0.edges[precedence] = startState

    def setPrecedenceDfa(self, precedenceDfa: bool):
        if False:
            return 10
        if self.precedenceDfa != precedenceDfa:
            self._states = dict()
            if precedenceDfa:
                precedenceState = DFAState(configs=ATNConfigSet())
                precedenceState.edges = []
                precedenceState.isAcceptState = False
                precedenceState.requiresFullContext = False
                self.s0 = precedenceState
            else:
                self.s0 = None
            self.precedenceDfa = precedenceDfa

    @property
    def states(self):
        if False:
            print('Hello World!')
        return self._states

    def sortedStates(self):
        if False:
            print('Hello World!')
        return sorted(self._states.keys(), key=lambda state: state.stateNumber)

    def __str__(self):
        if False:
            print('Hello World!')
        return self.toString(None)

    def toString(self, literalNames: list=None, symbolicNames: list=None):
        if False:
            i = 10
            return i + 15
        if self.s0 is None:
            return ''
        from antlr4.dfa.DFASerializer import DFASerializer
        serializer = DFASerializer(self, literalNames, symbolicNames)
        return str(serializer)

    def toLexerString(self):
        if False:
            i = 10
            return i + 15
        if self.s0 is None:
            return ''
        from antlr4.dfa.DFASerializer import LexerDFASerializer
        serializer = LexerDFASerializer(self)
        return str(serializer)
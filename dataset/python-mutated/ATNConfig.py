from io import StringIO
from antlr4.PredictionContext import PredictionContext
from antlr4.atn.ATNState import ATNState, DecisionState
from antlr4.atn.LexerActionExecutor import LexerActionExecutor
from antlr4.atn.SemanticContext import SemanticContext
ATNConfig = None

class ATNConfig(object):
    __slots__ = ('state', 'alt', 'context', 'semanticContext', 'reachesIntoOuterContext', 'precedenceFilterSuppressed')

    def __init__(self, state: ATNState=None, alt: int=None, context: PredictionContext=None, semantic: SemanticContext=None, config: ATNConfig=None):
        if False:
            return 10
        if config is not None:
            if state is None:
                state = config.state
            if alt is None:
                alt = config.alt
            if context is None:
                context = config.context
            if semantic is None:
                semantic = config.semanticContext
        if semantic is None:
            semantic = SemanticContext.NONE
        self.state = state
        self.alt = alt
        self.context = context
        self.semanticContext = semantic
        self.reachesIntoOuterContext = 0 if config is None else config.reachesIntoOuterContext
        self.precedenceFilterSuppressed = False if config is None else config.precedenceFilterSuppressed

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self is other:
            return True
        elif not isinstance(other, ATNConfig):
            return False
        else:
            return self.state.stateNumber == other.state.stateNumber and self.alt == other.alt and (self.context is other.context or self.context == other.context) and (self.semanticContext == other.semanticContext) and (self.precedenceFilterSuppressed == other.precedenceFilterSuppressed)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.state.stateNumber, self.alt, self.context, self.semanticContext))

    def hashCodeForConfigSet(self):
        if False:
            i = 10
            return i + 15
        return hash((self.state.stateNumber, self.alt, hash(self.semanticContext)))

    def equalsForConfigSet(self, other):
        if False:
            while True:
                i = 10
        if self is other:
            return True
        elif not isinstance(other, ATNConfig):
            return False
        else:
            return self.state.stateNumber == other.state.stateNumber and self.alt == other.alt and (self.semanticContext == other.semanticContext)

    def __str__(self):
        if False:
            while True:
                i = 10
        with StringIO() as buf:
            buf.write('(')
            buf.write(str(self.state))
            buf.write(',')
            buf.write(str(self.alt))
            if self.context is not None:
                buf.write(',[')
                buf.write(str(self.context))
                buf.write(']')
            if self.semanticContext is not None and self.semanticContext is not SemanticContext.NONE:
                buf.write(',')
                buf.write(str(self.semanticContext))
            if self.reachesIntoOuterContext > 0:
                buf.write(',up=')
                buf.write(str(self.reachesIntoOuterContext))
            buf.write(')')
            return buf.getvalue()
LexerATNConfig = None

class LexerATNConfig(ATNConfig):
    __slots__ = ('lexerActionExecutor', 'passedThroughNonGreedyDecision')

    def __init__(self, state: ATNState, alt: int=None, context: PredictionContext=None, semantic: SemanticContext=SemanticContext.NONE, lexerActionExecutor: LexerActionExecutor=None, config: LexerATNConfig=None):
        if False:
            print('Hello World!')
        super().__init__(state=state, alt=alt, context=context, semantic=semantic, config=config)
        if config is not None:
            if lexerActionExecutor is None:
                lexerActionExecutor = config.lexerActionExecutor
        self.lexerActionExecutor = lexerActionExecutor
        self.passedThroughNonGreedyDecision = False if config is None else self.checkNonGreedyDecision(config, state)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((self.state.stateNumber, self.alt, self.context, self.semanticContext, self.passedThroughNonGreedyDecision, self.lexerActionExecutor))

    def __eq__(self, other):
        if False:
            return 10
        if self is other:
            return True
        elif not isinstance(other, LexerATNConfig):
            return False
        if self.passedThroughNonGreedyDecision != other.passedThroughNonGreedyDecision:
            return False
        if not self.lexerActionExecutor == other.lexerActionExecutor:
            return False
        return super().__eq__(other)

    def hashCodeForConfigSet(self):
        if False:
            return 10
        return hash(self)

    def equalsForConfigSet(self, other):
        if False:
            while True:
                i = 10
        return self == other

    def checkNonGreedyDecision(self, source: LexerATNConfig, target: ATNState):
        if False:
            print('Hello World!')
        return source.passedThroughNonGreedyDecision or (isinstance(target, DecisionState) and target.nonGreedy)
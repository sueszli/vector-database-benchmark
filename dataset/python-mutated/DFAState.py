from io import StringIO
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.SemanticContext import SemanticContext

class PredPrediction(object):
    __slots__ = ('alt', 'pred')

    def __init__(self, pred: SemanticContext, alt: int):
        if False:
            i = 10
            return i + 15
        self.alt = alt
        self.pred = pred

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '(' + str(self.pred) + ', ' + str(self.alt) + ')'

class DFAState(object):
    __slots__ = ('stateNumber', 'configs', 'edges', 'isAcceptState', 'prediction', 'lexerActionExecutor', 'requiresFullContext', 'predicates')

    def __init__(self, stateNumber: int=-1, configs: ATNConfigSet=ATNConfigSet()):
        if False:
            return 10
        self.stateNumber = stateNumber
        self.configs = configs
        self.edges = None
        self.isAcceptState = False
        self.prediction = 0
        self.lexerActionExecutor = None
        self.requiresFullContext = False
        self.predicates = None

    def getAltSet(self):
        if False:
            return 10
        if self.configs is not None:
            return set((cfg.alt for cfg in self.configs)) or None
        return None

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.configs)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self is other:
            return True
        elif not isinstance(other, DFAState):
            return False
        else:
            return self.configs == other.configs

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        with StringIO() as buf:
            buf.write(str(self.stateNumber))
            buf.write(':')
            buf.write(str(self.configs))
            if self.isAcceptState:
                buf.write('=>')
                if self.predicates is not None:
                    buf.write(str(self.predicates))
                else:
                    buf.write(str(self.prediction))
            return buf.getvalue()
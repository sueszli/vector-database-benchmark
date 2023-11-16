from antlr4.PredictionContext import PredictionContextCache, PredictionContext, getCachedPredictionContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.dfa.DFAState import DFAState

class ATNSimulator(object):
    __slots__ = ('atn', 'sharedContextCache', '__dict__')
    ERROR = DFAState(configs=ATNConfigSet())
    ERROR.stateNumber = 2147483647

    def __init__(self, atn: ATN, sharedContextCache: PredictionContextCache):
        if False:
            i = 10
            return i + 15
        self.atn = atn
        self.sharedContextCache = sharedContextCache

    def getCachedContext(self, context: PredictionContext):
        if False:
            while True:
                i = 10
        if self.sharedContextCache is None:
            return context
        visited = dict()
        return getCachedPredictionContext(context, self.sharedContextCache, visited)
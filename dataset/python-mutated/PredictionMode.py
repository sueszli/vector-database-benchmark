from enum import Enum
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import RuleStopState
from antlr4.atn.SemanticContext import SemanticContext
PredictionMode = None

class PredictionMode(Enum):
    SLL = 0
    LL = 1
    LL_EXACT_AMBIG_DETECTION = 2

    @classmethod
    def hasSLLConflictTerminatingPrediction(cls, mode: PredictionMode, configs: ATNConfigSet):
        if False:
            while True:
                i = 10
        if cls.allConfigsInRuleStopStates(configs):
            return True
        if mode == PredictionMode.SLL:
            if configs.hasSemanticContext:
                dup = ATNConfigSet()
                for c in configs:
                    c = ATNConfig(config=c, semantic=SemanticContext.NONE)
                    dup.add(c)
                configs = dup
        altsets = cls.getConflictingAltSubsets(configs)
        return cls.hasConflictingAltSet(altsets) and (not cls.hasStateAssociatedWithOneAlt(configs))

    @classmethod
    def hasConfigInRuleStopState(cls, configs: ATNConfigSet):
        if False:
            while True:
                i = 10
        return any((isinstance(cfg.state, RuleStopState) for cfg in configs))

    @classmethod
    def allConfigsInRuleStopStates(cls, configs: ATNConfigSet):
        if False:
            for i in range(10):
                print('nop')
        return all((isinstance(cfg.state, RuleStopState) for cfg in configs))

    @classmethod
    def resolvesToJustOneViableAlt(cls, altsets: list):
        if False:
            return 10
        return cls.getSingleViableAlt(altsets)

    @classmethod
    def allSubsetsConflict(cls, altsets: list):
        if False:
            for i in range(10):
                print('nop')
        return not cls.hasNonConflictingAltSet(altsets)

    @classmethod
    def hasNonConflictingAltSet(cls, altsets: list):
        if False:
            print('Hello World!')
        return any((len(alts) == 1 for alts in altsets))

    @classmethod
    def hasConflictingAltSet(cls, altsets: list):
        if False:
            for i in range(10):
                print('nop')
        return any((len(alts) > 1 for alts in altsets))

    @classmethod
    def allSubsetsEqual(cls, altsets: list):
        if False:
            i = 10
            return i + 15
        if not altsets:
            return True
        first = next(iter(altsets))
        return all((alts == first for alts in iter(altsets)))

    @classmethod
    def getUniqueAlt(cls, altsets: list):
        if False:
            for i in range(10):
                print('nop')
        all = cls.getAlts(altsets)
        if len(all) == 1:
            return next(iter(all))
        return ATN.INVALID_ALT_NUMBER

    @classmethod
    def getAlts(cls, altsets: list):
        if False:
            return 10
        return set.union(*altsets)

    @classmethod
    def getConflictingAltSubsets(cls, configs: ATNConfigSet):
        if False:
            i = 10
            return i + 15
        configToAlts = dict()
        for c in configs:
            h = hash((c.state.stateNumber, c.context))
            alts = configToAlts.get(h, None)
            if alts is None:
                alts = set()
                configToAlts[h] = alts
            alts.add(c.alt)
        return configToAlts.values()

    @classmethod
    def getStateToAltMap(cls, configs: ATNConfigSet):
        if False:
            for i in range(10):
                print('nop')
        m = dict()
        for c in configs:
            alts = m.get(c.state, None)
            if alts is None:
                alts = set()
                m[c.state] = alts
            alts.add(c.alt)
        return m

    @classmethod
    def hasStateAssociatedWithOneAlt(cls, configs: ATNConfigSet):
        if False:
            i = 10
            return i + 15
        return any((len(alts) == 1 for alts in cls.getStateToAltMap(configs).values()))

    @classmethod
    def getSingleViableAlt(cls, altsets: list):
        if False:
            while True:
                i = 10
        viableAlts = set()
        for alts in altsets:
            minAlt = min(alts)
            viableAlts.add(minAlt)
            if len(viableAlts) > 1:
                return ATN.INVALID_ALT_NUMBER
        return min(viableAlts)
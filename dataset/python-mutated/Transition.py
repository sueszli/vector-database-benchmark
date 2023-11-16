from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
ATNState = None
RuleStartState = None

class Transition(object):
    __slots__ = ('target', 'isEpsilon', 'label')
    EPSILON = 1
    RANGE = 2
    RULE = 3
    PREDICATE = 4
    ATOM = 5
    ACTION = 6
    SET = 7
    NOT_SET = 8
    WILDCARD = 9
    PRECEDENCE = 10
    serializationNames = ['INVALID', 'EPSILON', 'RANGE', 'RULE', 'PREDICATE', 'ATOM', 'ACTION', 'SET', 'NOT_SET', 'WILDCARD', 'PRECEDENCE']
    serializationTypes = dict()

    def __init__(self, target: ATNState):
        if False:
            while True:
                i = 10
        if target is None:
            raise Exception('target cannot be null.')
        self.target = target
        self.isEpsilon = False
        self.label = None

class AtomTransition(Transition):
    __slots__ = ('label_', 'serializationType')

    def __init__(self, target: ATNState, label: int):
        if False:
            print('Hello World!')
        super().__init__(target)
        self.label_ = label
        self.label = self.makeLabel()
        self.serializationType = self.ATOM

    def makeLabel(self):
        if False:
            print('Hello World!')
        s = IntervalSet()
        s.addOne(self.label_)
        return s

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            print('Hello World!')
        return self.label_ == symbol

    def __str__(self):
        if False:
            while True:
                i = 10
        return str(self.label_)

class RuleTransition(Transition):
    __slots__ = ('ruleIndex', 'precedence', 'followState', 'serializationType')

    def __init__(self, ruleStart: RuleStartState, ruleIndex: int, precedence: int, followState: ATNState):
        if False:
            while True:
                i = 10
        super().__init__(ruleStart)
        self.ruleIndex = ruleIndex
        self.precedence = precedence
        self.followState = followState
        self.serializationType = self.RULE
        self.isEpsilon = True

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            while True:
                i = 10
        return False

class EpsilonTransition(Transition):
    __slots__ = ('serializationType', 'outermostPrecedenceReturn')

    def __init__(self, target, outermostPrecedenceReturn=-1):
        if False:
            while True:
                i = 10
        super(EpsilonTransition, self).__init__(target)
        self.serializationType = self.EPSILON
        self.isEpsilon = True
        self.outermostPrecedenceReturn = outermostPrecedenceReturn

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            for i in range(10):
                print('nop')
        return False

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'epsilon'

class RangeTransition(Transition):
    __slots__ = ('serializationType', 'start', 'stop')

    def __init__(self, target: ATNState, start: int, stop: int):
        if False:
            i = 10
            return i + 15
        super().__init__(target)
        self.serializationType = self.RANGE
        self.start = start
        self.stop = stop
        self.label = self.makeLabel()

    def makeLabel(self):
        if False:
            i = 10
            return i + 15
        s = IntervalSet()
        s.addRange(range(self.start, self.stop + 1))
        return s

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            while True:
                i = 10
        return symbol >= self.start and symbol <= self.stop

    def __str__(self):
        if False:
            while True:
                i = 10
        return "'" + chr(self.start) + "'..'" + chr(self.stop) + "'"

class AbstractPredicateTransition(Transition):

    def __init__(self, target: ATNState):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(target)

class PredicateTransition(AbstractPredicateTransition):
    __slots__ = ('serializationType', 'ruleIndex', 'predIndex', 'isCtxDependent')

    def __init__(self, target: ATNState, ruleIndex: int, predIndex: int, isCtxDependent: bool):
        if False:
            return 10
        super().__init__(target)
        self.serializationType = self.PREDICATE
        self.ruleIndex = ruleIndex
        self.predIndex = predIndex
        self.isCtxDependent = isCtxDependent
        self.isEpsilon = True

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            i = 10
            return i + 15
        return False

    def getPredicate(self):
        if False:
            i = 10
            return i + 15
        return Predicate(self.ruleIndex, self.predIndex, self.isCtxDependent)

    def __str__(self):
        if False:
            print('Hello World!')
        return 'pred_' + str(self.ruleIndex) + ':' + str(self.predIndex)

class ActionTransition(Transition):
    __slots__ = ('serializationType', 'ruleIndex', 'actionIndex', 'isCtxDependent')

    def __init__(self, target: ATNState, ruleIndex: int, actionIndex: int=-1, isCtxDependent: bool=False):
        if False:
            return 10
        super().__init__(target)
        self.serializationType = self.ACTION
        self.ruleIndex = ruleIndex
        self.actionIndex = actionIndex
        self.isCtxDependent = isCtxDependent
        self.isEpsilon = True

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            i = 10
            return i + 15
        return False

    def __str__(self):
        if False:
            print('Hello World!')
        return 'action_' + self.ruleIndex + ':' + self.actionIndex

class SetTransition(Transition):
    __slots__ = 'serializationType'

    def __init__(self, target: ATNState, set: IntervalSet):
        if False:
            return 10
        super().__init__(target)
        self.serializationType = self.SET
        if set is not None:
            self.label = set
        else:
            self.label = IntervalSet()
            self.label.addRange(range(Token.INVALID_TYPE, Token.INVALID_TYPE + 1))

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            print('Hello World!')
        return symbol in self.label

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.label)

class NotSetTransition(SetTransition):

    def __init__(self, target: ATNState, set: IntervalSet):
        if False:
            while True:
                i = 10
        super().__init__(target, set)
        self.serializationType = self.NOT_SET

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            while True:
                i = 10
        return symbol >= minVocabSymbol and symbol <= maxVocabSymbol and (not super(type(self), self).matches(symbol, minVocabSymbol, maxVocabSymbol))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '~' + super(type(self), self).__str__()

class WildcardTransition(Transition):
    __slots__ = 'serializationType'

    def __init__(self, target: ATNState):
        if False:
            return 10
        super().__init__(target)
        self.serializationType = self.WILDCARD

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            i = 10
            return i + 15
        return symbol >= minVocabSymbol and symbol <= maxVocabSymbol

    def __str__(self):
        if False:
            print('Hello World!')
        return '.'

class PrecedencePredicateTransition(AbstractPredicateTransition):
    __slots__ = ('serializationType', 'precedence')

    def __init__(self, target: ATNState, precedence: int):
        if False:
            print('Hello World!')
        super().__init__(target)
        self.serializationType = self.PRECEDENCE
        self.precedence = precedence
        self.isEpsilon = True

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        if False:
            while True:
                i = 10
        return False

    def getPredicate(self):
        if False:
            while True:
                i = 10
        return PrecedencePredicate(self.precedence)

    def __str__(self):
        if False:
            return 10
        return self.precedence + ' >= _p'
Transition.serializationTypes = {EpsilonTransition: Transition.EPSILON, RangeTransition: Transition.RANGE, RuleTransition: Transition.RULE, PredicateTransition: Transition.PREDICATE, AtomTransition: Transition.ATOM, ActionTransition: Transition.ACTION, SetTransition: Transition.SET, NotSetTransition: Transition.NOT_SET, WildcardTransition: Transition.WILDCARD, PrecedencePredicateTransition: Transition.PRECEDENCE}
del ATNState
del RuleStartState
from antlr4.atn.ATNState import *
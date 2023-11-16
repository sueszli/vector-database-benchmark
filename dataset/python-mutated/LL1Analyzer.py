from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.PredictionContext import PredictionContext, SingletonPredictionContext, PredictionContextFromRuleContext
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNState import ATNState, RuleStopState
from antlr4.atn.Transition import WildcardTransition, NotSetTransition, AbstractPredicateTransition, RuleTransition

class LL1Analyzer(object):
    __slots__ = 'atn'
    HIT_PRED = Token.INVALID_TYPE

    def __init__(self, atn: ATN):
        if False:
            print('Hello World!')
        self.atn = atn

    def getDecisionLookahead(self, s: ATNState):
        if False:
            return 10
        if s is None:
            return None
        count = len(s.transitions)
        look = [] * count
        for alt in range(0, count):
            look[alt] = set()
            lookBusy = set()
            seeThruPreds = False
            self._LOOK(s.transition(alt).target, None, PredictionContext.EMPTY, look[alt], lookBusy, set(), seeThruPreds, False)
            if len(look[alt]) == 0 or self.HIT_PRED in look[alt]:
                look[alt] = None
        return look

    def LOOK(self, s: ATNState, stopState: ATNState=None, ctx: RuleContext=None):
        if False:
            for i in range(10):
                print('nop')
        r = IntervalSet()
        seeThruPreds = True
        lookContext = PredictionContextFromRuleContext(s.atn, ctx) if ctx is not None else None
        self._LOOK(s, stopState, lookContext, r, set(), set(), seeThruPreds, True)
        return r

    def _LOOK(self, s: ATNState, stopState: ATNState, ctx: PredictionContext, look: IntervalSet, lookBusy: set, calledRuleStack: set, seeThruPreds: bool, addEOF: bool):
        if False:
            while True:
                i = 10
        c = ATNConfig(s, 0, ctx)
        if c in lookBusy:
            return
        lookBusy.add(c)
        if s == stopState:
            if ctx is None:
                look.addOne(Token.EPSILON)
                return
            elif ctx.isEmpty() and addEOF:
                look.addOne(Token.EOF)
                return
        if isinstance(s, RuleStopState):
            if ctx is None:
                look.addOne(Token.EPSILON)
                return
            elif ctx.isEmpty() and addEOF:
                look.addOne(Token.EOF)
                return
            if ctx != PredictionContext.EMPTY:
                removed = s.ruleIndex in calledRuleStack
                try:
                    calledRuleStack.discard(s.ruleIndex)
                    for i in range(0, len(ctx)):
                        returnState = self.atn.states[ctx.getReturnState(i)]
                        self._LOOK(returnState, stopState, ctx.getParent(i), look, lookBusy, calledRuleStack, seeThruPreds, addEOF)
                finally:
                    if removed:
                        calledRuleStack.add(s.ruleIndex)
                return
        for t in s.transitions:
            if type(t) == RuleTransition:
                if t.target.ruleIndex in calledRuleStack:
                    continue
                newContext = SingletonPredictionContext.create(ctx, t.followState.stateNumber)
                try:
                    calledRuleStack.add(t.target.ruleIndex)
                    self._LOOK(t.target, stopState, newContext, look, lookBusy, calledRuleStack, seeThruPreds, addEOF)
                finally:
                    calledRuleStack.remove(t.target.ruleIndex)
            elif isinstance(t, AbstractPredicateTransition):
                if seeThruPreds:
                    self._LOOK(t.target, stopState, ctx, look, lookBusy, calledRuleStack, seeThruPreds, addEOF)
                else:
                    look.addOne(self.HIT_PRED)
            elif t.isEpsilon:
                self._LOOK(t.target, stopState, ctx, look, lookBusy, calledRuleStack, seeThruPreds, addEOF)
            elif type(t) == WildcardTransition:
                look.addRange(range(Token.MIN_USER_TOKEN_TYPE, self.atn.maxTokenType + 1))
            else:
                set_ = t.label
                if set_ is not None:
                    if isinstance(t, NotSetTransition):
                        set_ = set_.complement(Token.MIN_USER_TOKEN_TYPE, self.atn.maxTokenType)
                    look.addSet(set_)
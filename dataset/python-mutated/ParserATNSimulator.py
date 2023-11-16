import sys
from antlr4 import DFA
from antlr4.BufferedTokenStream import TokenStream
from antlr4.Parser import Parser
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.PredictionContext import PredictionContextCache, PredictionContext, SingletonPredictionContext, PredictionContextFromRuleContext
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.Utils import str_list
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNSimulator import ATNSimulator
from antlr4.atn.ATNState import DecisionState, RuleStopState, ATNState
from antlr4.atn.PredictionMode import PredictionMode
from antlr4.atn.SemanticContext import SemanticContext, andContext, orContext
from antlr4.atn.Transition import Transition, RuleTransition, ActionTransition, PrecedencePredicateTransition, PredicateTransition, AtomTransition, SetTransition, NotSetTransition
from antlr4.dfa.DFAState import DFAState, PredPrediction
from antlr4.error.Errors import NoViableAltException

class ParserATNSimulator(ATNSimulator):
    __slots__ = ('parser', 'decisionToDFA', 'predictionMode', '_input', '_startIndex', '_outerContext', '_dfa', 'mergeCache')
    debug = False
    trace_atn_sim = False
    dfa_debug = False
    retry_debug = False

    def __init__(self, parser: Parser, atn: ATN, decisionToDFA: list, sharedContextCache: PredictionContextCache):
        if False:
            i = 10
            return i + 15
        super().__init__(atn, sharedContextCache)
        self.parser = parser
        self.decisionToDFA = decisionToDFA
        self.predictionMode = PredictionMode.LL
        self._input = None
        self._startIndex = 0
        self._outerContext = None
        self._dfa = None
        self.mergeCache = None

    def reset(self):
        if False:
            while True:
                i = 10
        pass

    def adaptivePredict(self, input: TokenStream, decision: int, outerContext: ParserRuleContext):
        if False:
            for i in range(10):
                print('nop')
        if ParserATNSimulator.debug or ParserATNSimulator.trace_atn_sim:
            print('adaptivePredict decision ' + str(decision) + ' exec LA(1)==' + self.getLookaheadName(input) + ' line ' + str(input.LT(1).line) + ':' + str(input.LT(1).column))
        self._input = input
        self._startIndex = input.index
        self._outerContext = outerContext
        dfa = self.decisionToDFA[decision]
        self._dfa = dfa
        m = input.mark()
        index = input.index
        try:
            if dfa.precedenceDfa:
                s0 = dfa.getPrecedenceStartState(self.parser.getPrecedence())
            else:
                s0 = dfa.s0
            if s0 is None:
                if outerContext is None:
                    outerContext = ParserRuleContext.EMPTY
                if ParserATNSimulator.debug:
                    print('predictATN decision ' + str(dfa.decision) + ' exec LA(1)==' + self.getLookaheadName(input) + ', outerContext=' + str(outerContext))
                fullCtx = False
                s0_closure = self.computeStartState(dfa.atnStartState, ParserRuleContext.EMPTY, fullCtx)
                if dfa.precedenceDfa:
                    dfa.s0.configs = s0_closure
                    s0_closure = self.applyPrecedenceFilter(s0_closure)
                    s0 = self.addDFAState(dfa, DFAState(configs=s0_closure))
                    dfa.setPrecedenceStartState(self.parser.getPrecedence(), s0)
                else:
                    s0 = self.addDFAState(dfa, DFAState(configs=s0_closure))
                    dfa.s0 = s0
            alt = self.execATN(dfa, s0, input, index, outerContext)
            if ParserATNSimulator.debug:
                print('DFA after predictATN: ' + dfa.toString(self.parser.literalNames))
            return alt
        finally:
            self._dfa = None
            self.mergeCache = None
            input.seek(index)
            input.release(m)

    def execATN(self, dfa: DFA, s0: DFAState, input: TokenStream, startIndex: int, outerContext: ParserRuleContext):
        if False:
            i = 10
            return i + 15
        if ParserATNSimulator.debug or ParserATNSimulator.trace_atn_sim:
            print('execATN decision ' + str(dfa.decision) + ', DFA state ' + str(s0) + ', LA(1)==' + self.getLookaheadName(input) + ' line ' + str(input.LT(1).line) + ':' + str(input.LT(1).column))
        previousD = s0
        t = input.LA(1)
        while True:
            D = self.getExistingTargetState(previousD, t)
            if D is None:
                D = self.computeTargetState(dfa, previousD, t)
            if D is self.ERROR:
                e = self.noViableAlt(input, outerContext, previousD.configs, startIndex)
                input.seek(startIndex)
                alt = self.getSynValidOrSemInvalidAltThatFinishedDecisionEntryRule(previousD.configs, outerContext)
                if alt != ATN.INVALID_ALT_NUMBER:
                    return alt
                raise e
            if D.requiresFullContext and self.predictionMode != PredictionMode.SLL:
                conflictingAlts = D.configs.conflictingAlts
                if D.predicates is not None:
                    if ParserATNSimulator.debug:
                        print('DFA state has preds in DFA sim LL failover')
                    conflictIndex = input.index
                    if conflictIndex != startIndex:
                        input.seek(startIndex)
                    conflictingAlts = self.evalSemanticContext(D.predicates, outerContext, True)
                    if len(conflictingAlts) == 1:
                        if ParserATNSimulator.debug:
                            print('Full LL avoided')
                        return min(conflictingAlts)
                    if conflictIndex != startIndex:
                        input.seek(conflictIndex)
                if ParserATNSimulator.dfa_debug:
                    print('ctx sensitive state ' + str(outerContext) + ' in ' + str(D))
                fullCtx = True
                s0_closure = self.computeStartState(dfa.atnStartState, outerContext, fullCtx)
                self.reportAttemptingFullContext(dfa, conflictingAlts, D.configs, startIndex, input.index)
                alt = self.execATNWithFullContext(dfa, D, s0_closure, input, startIndex, outerContext)
                return alt
            if D.isAcceptState:
                if D.predicates is None:
                    return D.prediction
                stopIndex = input.index
                input.seek(startIndex)
                alts = self.evalSemanticContext(D.predicates, outerContext, True)
                if len(alts) == 0:
                    raise self.noViableAlt(input, outerContext, D.configs, startIndex)
                elif len(alts) == 1:
                    return min(alts)
                else:
                    self.reportAmbiguity(dfa, D, startIndex, stopIndex, False, alts, D.configs)
                    return min(alts)
            previousD = D
            if t != Token.EOF:
                input.consume()
                t = input.LA(1)

    def getExistingTargetState(self, previousD: DFAState, t: int):
        if False:
            while True:
                i = 10
        edges = previousD.edges
        if edges is None or t + 1 < 0 or t + 1 >= len(edges):
            return None
        else:
            return edges[t + 1]

    def computeTargetState(self, dfa: DFA, previousD: DFAState, t: int):
        if False:
            for i in range(10):
                print('nop')
        reach = self.computeReachSet(previousD.configs, t, False)
        if reach is None:
            self.addDFAEdge(dfa, previousD, t, self.ERROR)
            return self.ERROR
        D = DFAState(configs=reach)
        predictedAlt = self.getUniqueAlt(reach)
        if ParserATNSimulator.debug:
            altSubSets = PredictionMode.getConflictingAltSubsets(reach)
            print('SLL altSubSets=' + str(altSubSets) + ', configs=' + str(reach) + ', predict=' + str(predictedAlt) + ', allSubsetsConflict=' + str(PredictionMode.allSubsetsConflict(altSubSets)) + ', conflictingAlts=' + str(self.getConflictingAlts(reach)))
        if predictedAlt != ATN.INVALID_ALT_NUMBER:
            D.isAcceptState = True
            D.configs.uniqueAlt = predictedAlt
            D.prediction = predictedAlt
        elif PredictionMode.hasSLLConflictTerminatingPrediction(self.predictionMode, reach):
            D.configs.conflictingAlts = self.getConflictingAlts(reach)
            D.requiresFullContext = True
            D.isAcceptState = True
            D.prediction = min(D.configs.conflictingAlts)
        if D.isAcceptState and D.configs.hasSemanticContext:
            self.predicateDFAState(D, self.atn.getDecisionState(dfa.decision))
            if D.predicates is not None:
                D.prediction = ATN.INVALID_ALT_NUMBER
        D = self.addDFAEdge(dfa, previousD, t, D)
        return D

    def predicateDFAState(self, dfaState: DFAState, decisionState: DecisionState):
        if False:
            return 10
        nalts = len(decisionState.transitions)
        altsToCollectPredsFrom = self.getConflictingAltsOrUniqueAlt(dfaState.configs)
        altToPred = self.getPredsForAmbigAlts(altsToCollectPredsFrom, dfaState.configs, nalts)
        if altToPred is not None:
            dfaState.predicates = self.getPredicatePredictions(altsToCollectPredsFrom, altToPred)
            dfaState.prediction = ATN.INVALID_ALT_NUMBER
        else:
            dfaState.prediction = min(altsToCollectPredsFrom)

    def execATNWithFullContext(self, dfa: DFA, D: DFAState, s0: ATNConfigSet, input: TokenStream, startIndex: int, outerContext: ParserRuleContext):
        if False:
            print('Hello World!')
        if ParserATNSimulator.debug or ParserATNSimulator.trace_atn_sim:
            print('execATNWithFullContext', str(s0))
        fullCtx = True
        foundExactAmbig = False
        reach = None
        previous = s0
        input.seek(startIndex)
        t = input.LA(1)
        predictedAlt = -1
        while True:
            reach = self.computeReachSet(previous, t, fullCtx)
            if reach is None:
                e = self.noViableAlt(input, outerContext, previous, startIndex)
                input.seek(startIndex)
                alt = self.getSynValidOrSemInvalidAltThatFinishedDecisionEntryRule(previous, outerContext)
                if alt != ATN.INVALID_ALT_NUMBER:
                    return alt
                else:
                    raise e
            altSubSets = PredictionMode.getConflictingAltSubsets(reach)
            if ParserATNSimulator.debug:
                print('LL altSubSets=' + str(altSubSets) + ', predict=' + str(PredictionMode.getUniqueAlt(altSubSets)) + ', resolvesToJustOneViableAlt=' + str(PredictionMode.resolvesToJustOneViableAlt(altSubSets)))
            reach.uniqueAlt = self.getUniqueAlt(reach)
            if reach.uniqueAlt != ATN.INVALID_ALT_NUMBER:
                predictedAlt = reach.uniqueAlt
                break
            elif self.predictionMode is not PredictionMode.LL_EXACT_AMBIG_DETECTION:
                predictedAlt = PredictionMode.resolvesToJustOneViableAlt(altSubSets)
                if predictedAlt != ATN.INVALID_ALT_NUMBER:
                    break
            elif PredictionMode.allSubsetsConflict(altSubSets) and PredictionMode.allSubsetsEqual(altSubSets):
                foundExactAmbig = True
                predictedAlt = PredictionMode.getSingleViableAlt(altSubSets)
                break
            previous = reach
            if t != Token.EOF:
                input.consume()
                t = input.LA(1)
        if reach.uniqueAlt != ATN.INVALID_ALT_NUMBER:
            self.reportContextSensitivity(dfa, predictedAlt, reach, startIndex, input.index)
            return predictedAlt
        self.reportAmbiguity(dfa, D, startIndex, input.index, foundExactAmbig, None, reach)
        return predictedAlt

    def computeReachSet(self, closure: ATNConfigSet, t: int, fullCtx: bool):
        if False:
            i = 10
            return i + 15
        if ParserATNSimulator.debug:
            print('in computeReachSet, starting closure: ' + str(closure))
        if self.mergeCache is None:
            self.mergeCache = dict()
        intermediate = ATNConfigSet(fullCtx)
        skippedStopStates = None
        for c in closure:
            if ParserATNSimulator.debug:
                print('testing ' + self.getTokenName(t) + ' at ' + str(c))
            if isinstance(c.state, RuleStopState):
                if fullCtx or t == Token.EOF:
                    if skippedStopStates is None:
                        skippedStopStates = list()
                    skippedStopStates.append(c)
                continue
            for trans in c.state.transitions:
                target = self.getReachableTarget(trans, t)
                if target is not None:
                    intermediate.add(ATNConfig(state=target, config=c), self.mergeCache)
        reach = None
        if skippedStopStates is None and t != Token.EOF:
            if len(intermediate) == 1:
                reach = intermediate
            elif self.getUniqueAlt(intermediate) != ATN.INVALID_ALT_NUMBER:
                reach = intermediate
        if reach is None:
            reach = ATNConfigSet(fullCtx)
            closureBusy = set()
            treatEofAsEpsilon = t == Token.EOF
            for c in intermediate:
                self.closure(c, reach, closureBusy, False, fullCtx, treatEofAsEpsilon)
        if t == Token.EOF:
            reach = self.removeAllConfigsNotInRuleStopState(reach, reach is intermediate)
        if skippedStopStates is not None and (not fullCtx or not PredictionMode.hasConfigInRuleStopState(reach)):
            for c in skippedStopStates:
                reach.add(c, self.mergeCache)
        if ParserATNSimulator.trace_atn_sim:
            print('computeReachSet', str(closure), '->', reach)
        if len(reach) == 0:
            return None
        else:
            return reach

    def removeAllConfigsNotInRuleStopState(self, configs: ATNConfigSet, lookToEndOfRule: bool):
        if False:
            return 10
        if PredictionMode.allConfigsInRuleStopStates(configs):
            return configs
        result = ATNConfigSet(configs.fullCtx)
        for config in configs:
            if isinstance(config.state, RuleStopState):
                result.add(config, self.mergeCache)
                continue
            if lookToEndOfRule and config.state.epsilonOnlyTransitions:
                nextTokens = self.atn.nextTokens(config.state)
                if Token.EPSILON in nextTokens:
                    endOfRuleState = self.atn.ruleToStopState[config.state.ruleIndex]
                    result.add(ATNConfig(state=endOfRuleState, config=config), self.mergeCache)
        return result

    def computeStartState(self, p: ATNState, ctx: RuleContext, fullCtx: bool):
        if False:
            i = 10
            return i + 15
        initialContext = PredictionContextFromRuleContext(self.atn, ctx)
        configs = ATNConfigSet(fullCtx)
        if ParserATNSimulator.trace_atn_sim:
            print('computeStartState from ATN state ' + str(p) + ' initialContext=' + str(initialContext))
        for i in range(0, len(p.transitions)):
            target = p.transitions[i].target
            c = ATNConfig(target, i + 1, initialContext)
            closureBusy = set()
            self.closure(c, configs, closureBusy, True, fullCtx, False)
        return configs

    def applyPrecedenceFilter(self, configs: ATNConfigSet):
        if False:
            for i in range(10):
                print('nop')
        statesFromAlt1 = dict()
        configSet = ATNConfigSet(configs.fullCtx)
        for config in configs:
            if config.alt != 1:
                continue
            updatedContext = config.semanticContext.evalPrecedence(self.parser, self._outerContext)
            if updatedContext is None:
                continue
            statesFromAlt1[config.state.stateNumber] = config.context
            if updatedContext is not config.semanticContext:
                configSet.add(ATNConfig(config=config, semantic=updatedContext), self.mergeCache)
            else:
                configSet.add(config, self.mergeCache)
        for config in configs:
            if config.alt == 1:
                continue
            if not config.precedenceFilterSuppressed:
                context = statesFromAlt1.get(config.state.stateNumber, None)
                if context == config.context:
                    continue
            configSet.add(config, self.mergeCache)
        return configSet

    def getReachableTarget(self, trans: Transition, ttype: int):
        if False:
            while True:
                i = 10
        if trans.matches(ttype, 0, self.atn.maxTokenType):
            return trans.target
        else:
            return None

    def getPredsForAmbigAlts(self, ambigAlts: set, configs: ATNConfigSet, nalts: int):
        if False:
            while True:
                i = 10
        altToPred = [None] * (nalts + 1)
        for c in configs:
            if c.alt in ambigAlts:
                altToPred[c.alt] = orContext(altToPred[c.alt], c.semanticContext)
        nPredAlts = 0
        for i in range(1, nalts + 1):
            if altToPred[i] is None:
                altToPred[i] = SemanticContext.NONE
            elif altToPred[i] is not SemanticContext.NONE:
                nPredAlts += 1
        if nPredAlts == 0:
            altToPred = None
        if ParserATNSimulator.debug:
            print('getPredsForAmbigAlts result ' + str_list(altToPred))
        return altToPred

    def getPredicatePredictions(self, ambigAlts: set, altToPred: list):
        if False:
            i = 10
            return i + 15
        pairs = []
        containsPredicate = False
        for i in range(1, len(altToPred)):
            pred = altToPred[i]
            if ambigAlts is not None and i in ambigAlts:
                pairs.append(PredPrediction(pred, i))
            if pred is not SemanticContext.NONE:
                containsPredicate = True
        if not containsPredicate:
            return None
        return pairs

    def getSynValidOrSemInvalidAltThatFinishedDecisionEntryRule(self, configs: ATNConfigSet, outerContext: ParserRuleContext):
        if False:
            for i in range(10):
                print('nop')
        (semValidConfigs, semInvalidConfigs) = self.splitAccordingToSemanticValidity(configs, outerContext)
        alt = self.getAltThatFinishedDecisionEntryRule(semValidConfigs)
        if alt != ATN.INVALID_ALT_NUMBER:
            return alt
        if len(semInvalidConfigs) > 0:
            alt = self.getAltThatFinishedDecisionEntryRule(semInvalidConfigs)
            if alt != ATN.INVALID_ALT_NUMBER:
                return alt
        return ATN.INVALID_ALT_NUMBER

    def getAltThatFinishedDecisionEntryRule(self, configs: ATNConfigSet):
        if False:
            while True:
                i = 10
        alts = set()
        for c in configs:
            if c.reachesIntoOuterContext > 0 or (isinstance(c.state, RuleStopState) and c.context.hasEmptyPath()):
                alts.add(c.alt)
        if len(alts) == 0:
            return ATN.INVALID_ALT_NUMBER
        else:
            return min(alts)

    def splitAccordingToSemanticValidity(self, configs: ATNConfigSet, outerContext: ParserRuleContext):
        if False:
            for i in range(10):
                print('nop')
        succeeded = ATNConfigSet(configs.fullCtx)
        failed = ATNConfigSet(configs.fullCtx)
        for c in configs:
            if c.semanticContext is not SemanticContext.NONE:
                predicateEvaluationResult = c.semanticContext.eval(self.parser, outerContext)
                if predicateEvaluationResult:
                    succeeded.add(c)
                else:
                    failed.add(c)
            else:
                succeeded.add(c)
        return (succeeded, failed)

    def evalSemanticContext(self, predPredictions: list, outerContext: ParserRuleContext, complete: bool):
        if False:
            while True:
                i = 10
        predictions = set()
        for pair in predPredictions:
            if pair.pred is SemanticContext.NONE:
                predictions.add(pair.alt)
                if not complete:
                    break
                continue
            predicateEvaluationResult = pair.pred.eval(self.parser, outerContext)
            if ParserATNSimulator.debug or ParserATNSimulator.dfa_debug:
                print('eval pred ' + str(pair) + '=' + str(predicateEvaluationResult))
            if predicateEvaluationResult:
                if ParserATNSimulator.debug or ParserATNSimulator.dfa_debug:
                    print('PREDICT ' + str(pair.alt))
                predictions.add(pair.alt)
                if not complete:
                    break
        return predictions

    def closure(self, config: ATNConfig, configs: ATNConfigSet, closureBusy: set, collectPredicates: bool, fullCtx: bool, treatEofAsEpsilon: bool):
        if False:
            while True:
                i = 10
        initialDepth = 0
        self.closureCheckingStopState(config, configs, closureBusy, collectPredicates, fullCtx, initialDepth, treatEofAsEpsilon)

    def closureCheckingStopState(self, config: ATNConfig, configs: ATNConfigSet, closureBusy: set, collectPredicates: bool, fullCtx: bool, depth: int, treatEofAsEpsilon: bool):
        if False:
            i = 10
            return i + 15
        if ParserATNSimulator.trace_atn_sim:
            print('closure(' + str(config) + ')')
        if isinstance(config.state, RuleStopState):
            if not config.context.isEmpty():
                for i in range(0, len(config.context)):
                    state = config.context.getReturnState(i)
                    if state is PredictionContext.EMPTY_RETURN_STATE:
                        if fullCtx:
                            configs.add(ATNConfig(state=config.state, context=PredictionContext.EMPTY, config=config), self.mergeCache)
                            continue
                        else:
                            if ParserATNSimulator.debug:
                                print('FALLING off rule ' + self.getRuleName(config.state.ruleIndex))
                            self.closure_(config, configs, closureBusy, collectPredicates, fullCtx, depth, treatEofAsEpsilon)
                        continue
                    returnState = self.atn.states[state]
                    newContext = config.context.getParent(i)
                    c = ATNConfig(state=returnState, alt=config.alt, context=newContext, semantic=config.semanticContext)
                    c.reachesIntoOuterContext = config.reachesIntoOuterContext
                    self.closureCheckingStopState(c, configs, closureBusy, collectPredicates, fullCtx, depth - 1, treatEofAsEpsilon)
                return
            elif fullCtx:
                configs.add(config, self.mergeCache)
                return
            elif ParserATNSimulator.debug:
                print('FALLING off rule ' + self.getRuleName(config.state.ruleIndex))
        self.closure_(config, configs, closureBusy, collectPredicates, fullCtx, depth, treatEofAsEpsilon)

    def closure_(self, config: ATNConfig, configs: ATNConfigSet, closureBusy: set, collectPredicates: bool, fullCtx: bool, depth: int, treatEofAsEpsilon: bool):
        if False:
            for i in range(10):
                print('nop')
        p = config.state
        if not p.epsilonOnlyTransitions:
            configs.add(config, self.mergeCache)
        first = True
        for t in p.transitions:
            if first:
                first = False
                if self.canDropLoopEntryEdgeInLeftRecursiveRule(config):
                    continue
            continueCollecting = collectPredicates and (not isinstance(t, ActionTransition))
            c = self.getEpsilonTarget(config, t, continueCollecting, depth == 0, fullCtx, treatEofAsEpsilon)
            if c is not None:
                newDepth = depth
                if isinstance(config.state, RuleStopState):
                    if self._dfa is not None and self._dfa.precedenceDfa:
                        if t.outermostPrecedenceReturn == self._dfa.atnStartState.ruleIndex:
                            c.precedenceFilterSuppressed = True
                    c.reachesIntoOuterContext += 1
                    if c in closureBusy:
                        continue
                    closureBusy.add(c)
                    configs.dipsIntoOuterContext = True
                    newDepth -= 1
                    if ParserATNSimulator.debug:
                        print('dips into outer ctx: ' + str(c))
                else:
                    if not t.isEpsilon:
                        if c in closureBusy:
                            continue
                        closureBusy.add(c)
                    if isinstance(t, RuleTransition):
                        if newDepth >= 0:
                            newDepth += 1
                self.closureCheckingStopState(c, configs, closureBusy, continueCollecting, fullCtx, newDepth, treatEofAsEpsilon)

    def canDropLoopEntryEdgeInLeftRecursiveRule(self, config):
        if False:
            return 10
        p = config.state
        if p.stateType != ATNState.STAR_LOOP_ENTRY or not p.isPrecedenceDecision or config.context.isEmpty() or config.context.hasEmptyPath():
            return False
        numCtxs = len(config.context)
        for i in range(0, numCtxs):
            returnState = self.atn.states[config.context.getReturnState(i)]
            if returnState.ruleIndex != p.ruleIndex:
                return False
        decisionStartState = p.transitions[0].target
        blockEndStateNum = decisionStartState.endState.stateNumber
        blockEndState = self.atn.states[blockEndStateNum]
        for i in range(0, numCtxs):
            returnStateNumber = config.context.getReturnState(i)
            returnState = self.atn.states[returnStateNumber]
            if len(returnState.transitions) != 1 or not returnState.transitions[0].isEpsilon:
                return False
            returnStateTarget = returnState.transitions[0].target
            if returnState.stateType == ATNState.BLOCK_END and returnStateTarget is p:
                continue
            if returnState is blockEndState:
                continue
            if returnStateTarget is blockEndState:
                continue
            if returnStateTarget.stateType == ATNState.BLOCK_END and len(returnStateTarget.transitions) == 1 and returnStateTarget.transitions[0].isEpsilon and (returnStateTarget.transitions[0].target is p):
                continue
            return False
        return True

    def getRuleName(self, index: int):
        if False:
            print('Hello World!')
        if self.parser is not None and index >= 0:
            return self.parser.ruleNames[index]
        else:
            return '<rule ' + str(index) + '>'
    epsilonTargetMethods = dict()
    epsilonTargetMethods[Transition.RULE] = lambda sim, config, t, collectPredicates, inContext, fullCtx, treatEofAsEpsilon: sim.ruleTransition(config, t)
    epsilonTargetMethods[Transition.PRECEDENCE] = lambda sim, config, t, collectPredicates, inContext, fullCtx, treatEofAsEpsilon: sim.precedenceTransition(config, t, collectPredicates, inContext, fullCtx)
    epsilonTargetMethods[Transition.PREDICATE] = lambda sim, config, t, collectPredicates, inContext, fullCtx, treatEofAsEpsilon: sim.predTransition(config, t, collectPredicates, inContext, fullCtx)
    epsilonTargetMethods[Transition.ACTION] = lambda sim, config, t, collectPredicates, inContext, fullCtx, treatEofAsEpsilon: sim.actionTransition(config, t)
    epsilonTargetMethods[Transition.EPSILON] = lambda sim, config, t, collectPredicates, inContext, fullCtx, treatEofAsEpsilon: ATNConfig(state=t.target, config=config)
    epsilonTargetMethods[Transition.ATOM] = lambda sim, config, t, collectPredicates, inContext, fullCtx, treatEofAsEpsilon: ATNConfig(state=t.target, config=config) if treatEofAsEpsilon and t.matches(Token.EOF, 0, 1) else None
    epsilonTargetMethods[Transition.RANGE] = lambda sim, config, t, collectPredicates, inContext, fullCtx, treatEofAsEpsilon: ATNConfig(state=t.target, config=config) if treatEofAsEpsilon and t.matches(Token.EOF, 0, 1) else None
    epsilonTargetMethods[Transition.SET] = lambda sim, config, t, collectPredicates, inContext, fullCtx, treatEofAsEpsilon: ATNConfig(state=t.target, config=config) if treatEofAsEpsilon and t.matches(Token.EOF, 0, 1) else None

    def getEpsilonTarget(self, config: ATNConfig, t: Transition, collectPredicates: bool, inContext: bool, fullCtx: bool, treatEofAsEpsilon: bool):
        if False:
            for i in range(10):
                print('nop')
        m = self.epsilonTargetMethods.get(t.serializationType, None)
        if m is None:
            return None
        else:
            return m(self, config, t, collectPredicates, inContext, fullCtx, treatEofAsEpsilon)

    def actionTransition(self, config: ATNConfig, t: ActionTransition):
        if False:
            print('Hello World!')
        if ParserATNSimulator.debug:
            print('ACTION edge ' + str(t.ruleIndex) + ':' + str(t.actionIndex))
        return ATNConfig(state=t.target, config=config)

    def precedenceTransition(self, config: ATNConfig, pt: PrecedencePredicateTransition, collectPredicates: bool, inContext: bool, fullCtx: bool):
        if False:
            print('Hello World!')
        if ParserATNSimulator.debug:
            print('PRED (collectPredicates=' + str(collectPredicates) + ') ' + str(pt.precedence) + '>=_p, ctx dependent=true')
            if self.parser is not None:
                print('context surrounding pred is ' + str(self.parser.getRuleInvocationStack()))
        c = None
        if collectPredicates and inContext:
            if fullCtx:
                currentPosition = self._input.index
                self._input.seek(self._startIndex)
                predSucceeds = pt.getPredicate().eval(self.parser, self._outerContext)
                self._input.seek(currentPosition)
                if predSucceeds:
                    c = ATNConfig(state=pt.target, config=config)
            else:
                newSemCtx = andContext(config.semanticContext, pt.getPredicate())
                c = ATNConfig(state=pt.target, semantic=newSemCtx, config=config)
        else:
            c = ATNConfig(state=pt.target, config=config)
        if ParserATNSimulator.debug:
            print('config from pred transition=' + str(c))
        return c

    def predTransition(self, config: ATNConfig, pt: PredicateTransition, collectPredicates: bool, inContext: bool, fullCtx: bool):
        if False:
            print('Hello World!')
        if ParserATNSimulator.debug:
            print('PRED (collectPredicates=' + str(collectPredicates) + ') ' + str(pt.ruleIndex) + ':' + str(pt.predIndex) + ', ctx dependent=' + str(pt.isCtxDependent))
            if self.parser is not None:
                print('context surrounding pred is ' + str(self.parser.getRuleInvocationStack()))
        c = None
        if collectPredicates and (not pt.isCtxDependent or (pt.isCtxDependent and inContext)):
            if fullCtx:
                currentPosition = self._input.index
                self._input.seek(self._startIndex)
                predSucceeds = pt.getPredicate().eval(self.parser, self._outerContext)
                self._input.seek(currentPosition)
                if predSucceeds:
                    c = ATNConfig(state=pt.target, config=config)
            else:
                newSemCtx = andContext(config.semanticContext, pt.getPredicate())
                c = ATNConfig(state=pt.target, semantic=newSemCtx, config=config)
        else:
            c = ATNConfig(state=pt.target, config=config)
        if ParserATNSimulator.debug:
            print('config from pred transition=' + str(c))
        return c

    def ruleTransition(self, config: ATNConfig, t: RuleTransition):
        if False:
            for i in range(10):
                print('nop')
        if ParserATNSimulator.debug:
            print('CALL rule ' + self.getRuleName(t.target.ruleIndex) + ', ctx=' + str(config.context))
        returnState = t.followState
        newContext = SingletonPredictionContext.create(config.context, returnState.stateNumber)
        return ATNConfig(state=t.target, context=newContext, config=config)

    def getConflictingAlts(self, configs: ATNConfigSet):
        if False:
            return 10
        altsets = PredictionMode.getConflictingAltSubsets(configs)
        return PredictionMode.getAlts(altsets)

    def getConflictingAltsOrUniqueAlt(self, configs: ATNConfigSet):
        if False:
            print('Hello World!')
        conflictingAlts = None
        if configs.uniqueAlt != ATN.INVALID_ALT_NUMBER:
            conflictingAlts = set()
            conflictingAlts.add(configs.uniqueAlt)
        else:
            conflictingAlts = configs.conflictingAlts
        return conflictingAlts

    def getTokenName(self, t: int):
        if False:
            for i in range(10):
                print('nop')
        if t == Token.EOF:
            return 'EOF'
        if self.parser is not None and self.parser.literalNames is not None and (t < len(self.parser.literalNames)):
            return self.parser.literalNames[t] + '<' + str(t) + '>'
        if self.parser is not None and self.parser.symbolicNames is not None and (t < len(self.parser.symbolicNames)):
            return self.parser.symbolicNames[t] + '<' + str(t) + '>'
        else:
            return str(t)

    def getLookaheadName(self, input: TokenStream):
        if False:
            i = 10
            return i + 15
        return self.getTokenName(input.LA(1))

    def dumpDeadEndConfigs(self, nvae: NoViableAltException):
        if False:
            while True:
                i = 10
        print('dead end configs: ')
        for c in nvae.getDeadEndConfigs():
            trans = 'no edges'
            if len(c.state.transitions) > 0:
                t = c.state.transitions[0]
                if isinstance(t, AtomTransition):
                    trans = 'Atom ' + self.getTokenName(t.label)
                elif isinstance(t, SetTransition):
                    neg = isinstance(t, NotSetTransition)
                    trans = ('~' if neg else '') + 'Set ' + str(t.set)
            print(c.toString(self.parser, True) + ':' + trans, file=sys.stderr)

    def noViableAlt(self, input: TokenStream, outerContext: ParserRuleContext, configs: ATNConfigSet, startIndex: int):
        if False:
            while True:
                i = 10
        return NoViableAltException(self.parser, input, input.get(startIndex), input.LT(1), configs, outerContext)

    def getUniqueAlt(self, configs: ATNConfigSet):
        if False:
            i = 10
            return i + 15
        alt = ATN.INVALID_ALT_NUMBER
        for c in configs:
            if alt == ATN.INVALID_ALT_NUMBER:
                alt = c.alt
            elif c.alt != alt:
                return ATN.INVALID_ALT_NUMBER
        return alt

    def addDFAEdge(self, dfa: DFA, from_: DFAState, t: int, to: DFAState):
        if False:
            i = 10
            return i + 15
        if ParserATNSimulator.debug:
            print('EDGE ' + str(from_) + ' -> ' + str(to) + ' upon ' + self.getTokenName(t))
        if to is None:
            return None
        to = self.addDFAState(dfa, to)
        if from_ is None or t < -1 or t > self.atn.maxTokenType:
            return to
        if from_.edges is None:
            from_.edges = [None] * (self.atn.maxTokenType + 2)
        from_.edges[t + 1] = to
        if ParserATNSimulator.debug:
            names = None if self.parser is None else self.parser.literalNames
            print('DFA=\n' + dfa.toString(names))
        return to

    def addDFAState(self, dfa: DFA, D: DFAState):
        if False:
            while True:
                i = 10
        if D is self.ERROR:
            return D
        existing = dfa.states.get(D, None)
        if existing is not None:
            if ParserATNSimulator.trace_atn_sim:
                print('addDFAState', str(D), 'exists')
            return existing
        D.stateNumber = len(dfa.states)
        if not D.configs.readonly:
            D.configs.optimizeConfigs(self)
            D.configs.setReadonly(True)
        if ParserATNSimulator.trace_atn_sim:
            print('addDFAState new', str(D))
        dfa.states[D] = D
        return D

    def reportAttemptingFullContext(self, dfa: DFA, conflictingAlts: set, configs: ATNConfigSet, startIndex: int, stopIndex: int):
        if False:
            i = 10
            return i + 15
        if ParserATNSimulator.debug or ParserATNSimulator.retry_debug:
            print('reportAttemptingFullContext decision=' + str(dfa.decision) + ':' + str(configs) + ', input=' + self.parser.getTokenStream().getText(startIndex, stopIndex))
        if self.parser is not None:
            self.parser.getErrorListenerDispatch().reportAttemptingFullContext(self.parser, dfa, startIndex, stopIndex, conflictingAlts, configs)

    def reportContextSensitivity(self, dfa: DFA, prediction: int, configs: ATNConfigSet, startIndex: int, stopIndex: int):
        if False:
            while True:
                i = 10
        if ParserATNSimulator.debug or ParserATNSimulator.retry_debug:
            print('reportContextSensitivity decision=' + str(dfa.decision) + ':' + str(configs) + ', input=' + self.parser.getTokenStream().getText(startIndex, stopIndex))
        if self.parser is not None:
            self.parser.getErrorListenerDispatch().reportContextSensitivity(self.parser, dfa, startIndex, stopIndex, prediction, configs)

    def reportAmbiguity(self, dfa: DFA, D: DFAState, startIndex: int, stopIndex: int, exact: bool, ambigAlts: set, configs: ATNConfigSet):
        if False:
            i = 10
            return i + 15
        if ParserATNSimulator.debug or ParserATNSimulator.retry_debug:
            print('reportAmbiguity ' + str(ambigAlts) + ':' + str(configs) + ', input=' + self.parser.getTokenStream().getText(startIndex, stopIndex))
        if self.parser is not None:
            self.parser.getErrorListenerDispatch().reportAmbiguity(self.parser, dfa, startIndex, stopIndex, exact, ambigAlts, configs)
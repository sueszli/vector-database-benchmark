from antlr4.PredictionContext import PredictionContextCache, SingletonPredictionContext, PredictionContext
from antlr4.InputStream import InputStream
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import LexerATNConfig
from antlr4.atn.ATNSimulator import ATNSimulator
from antlr4.atn.ATNConfigSet import ATNConfigSet, OrderedATNConfigSet
from antlr4.atn.ATNState import RuleStopState, ATNState
from antlr4.atn.LexerActionExecutor import LexerActionExecutor
from antlr4.atn.Transition import Transition
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import LexerNoViableAltException, UnsupportedOperationException

class SimState(object):
    __slots__ = ('index', 'line', 'column', 'dfaState')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.reset()

    def reset(self):
        if False:
            return 10
        self.index = -1
        self.line = 0
        self.column = -1
        self.dfaState = None
Lexer = None
LexerATNSimulator = None

class LexerATNSimulator(ATNSimulator):
    __slots__ = ('decisionToDFA', 'recog', 'startIndex', 'line', 'column', 'mode', 'DEFAULT_MODE', 'MAX_CHAR_VALUE', 'prevAccept')
    debug = False
    dfa_debug = False
    MIN_DFA_EDGE = 0
    MAX_DFA_EDGE = 127
    ERROR = None

    def __init__(self, recog: Lexer, atn: ATN, decisionToDFA: list, sharedContextCache: PredictionContextCache):
        if False:
            return 10
        super().__init__(atn, sharedContextCache)
        self.decisionToDFA = decisionToDFA
        self.recog = recog
        self.startIndex = -1
        self.line = 1
        self.column = 0
        from antlr4.Lexer import Lexer
        self.mode = Lexer.DEFAULT_MODE
        self.DEFAULT_MODE = Lexer.DEFAULT_MODE
        self.MAX_CHAR_VALUE = Lexer.MAX_CHAR_VALUE
        self.prevAccept = SimState()

    def copyState(self, simulator: LexerATNSimulator):
        if False:
            for i in range(10):
                print('nop')
        self.column = simulator.column
        self.line = simulator.line
        self.mode = simulator.mode
        self.startIndex = simulator.startIndex

    def match(self, input: InputStream, mode: int):
        if False:
            return 10
        self.mode = mode
        mark = input.mark()
        try:
            self.startIndex = input.index
            self.prevAccept.reset()
            dfa = self.decisionToDFA[mode]
            if dfa.s0 is None:
                return self.matchATN(input)
            else:
                return self.execATN(input, dfa.s0)
        finally:
            input.release(mark)

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.prevAccept.reset()
        self.startIndex = -1
        self.line = 1
        self.column = 0
        self.mode = self.DEFAULT_MODE

    def matchATN(self, input: InputStream):
        if False:
            while True:
                i = 10
        startState = self.atn.modeToStartState[self.mode]
        if LexerATNSimulator.debug:
            print('matchATN mode ' + str(self.mode) + ' start: ' + str(startState))
        old_mode = self.mode
        s0_closure = self.computeStartState(input, startState)
        suppressEdge = s0_closure.hasSemanticContext
        s0_closure.hasSemanticContext = False
        next = self.addDFAState(s0_closure)
        if not suppressEdge:
            self.decisionToDFA[self.mode].s0 = next
        predict = self.execATN(input, next)
        if LexerATNSimulator.debug:
            print('DFA after matchATN: ' + str(self.decisionToDFA[old_mode].toLexerString()))
        return predict

    def execATN(self, input: InputStream, ds0: DFAState):
        if False:
            return 10
        if LexerATNSimulator.debug:
            print('start state closure=' + str(ds0.configs))
        if ds0.isAcceptState:
            self.captureSimState(self.prevAccept, input, ds0)
        t = input.LA(1)
        s = ds0
        while True:
            if LexerATNSimulator.debug:
                print('execATN loop starting closure:', str(s.configs))
            target = self.getExistingTargetState(s, t)
            if target is None:
                target = self.computeTargetState(input, s, t)
            if target == self.ERROR:
                break
            if t != Token.EOF:
                self.consume(input)
            if target.isAcceptState:
                self.captureSimState(self.prevAccept, input, target)
                if t == Token.EOF:
                    break
            t = input.LA(1)
            s = target
        return self.failOrAccept(self.prevAccept, input, s.configs, t)

    def getExistingTargetState(self, s: DFAState, t: int):
        if False:
            while True:
                i = 10
        if s.edges is None or t < self.MIN_DFA_EDGE or t > self.MAX_DFA_EDGE:
            return None
        target = s.edges[t - self.MIN_DFA_EDGE]
        if LexerATNSimulator.debug and target is not None:
            print('reuse state', str(s.stateNumber), 'edge to', str(target.stateNumber))
        return target

    def computeTargetState(self, input: InputStream, s: DFAState, t: int):
        if False:
            i = 10
            return i + 15
        reach = OrderedATNConfigSet()
        self.getReachableConfigSet(input, s.configs, reach, t)
        if len(reach) == 0:
            if not reach.hasSemanticContext:
                self.addDFAEdge(s, t, self.ERROR)
            return self.ERROR
        return self.addDFAEdge(s, t, cfgs=reach)

    def failOrAccept(self, prevAccept: SimState, input: InputStream, reach: ATNConfigSet, t: int):
        if False:
            print('Hello World!')
        if self.prevAccept.dfaState is not None:
            lexerActionExecutor = prevAccept.dfaState.lexerActionExecutor
            self.accept(input, lexerActionExecutor, self.startIndex, prevAccept.index, prevAccept.line, prevAccept.column)
            return prevAccept.dfaState.prediction
        else:
            if t == Token.EOF and input.index == self.startIndex:
                return Token.EOF
            raise LexerNoViableAltException(self.recog, input, self.startIndex, reach)

    def getReachableConfigSet(self, input: InputStream, closure: ATNConfigSet, reach: ATNConfigSet, t: int):
        if False:
            i = 10
            return i + 15
        skipAlt = ATN.INVALID_ALT_NUMBER
        for cfg in closure:
            currentAltReachedAcceptState = cfg.alt == skipAlt
            if currentAltReachedAcceptState and cfg.passedThroughNonGreedyDecision:
                continue
            if LexerATNSimulator.debug:
                print('testing', self.getTokenName(t), 'at', str(cfg))
            for trans in cfg.state.transitions:
                target = self.getReachableTarget(trans, t)
                if target is not None:
                    lexerActionExecutor = cfg.lexerActionExecutor
                    if lexerActionExecutor is not None:
                        lexerActionExecutor = lexerActionExecutor.fixOffsetBeforeMatch(input.index - self.startIndex)
                    treatEofAsEpsilon = t == Token.EOF
                    config = LexerATNConfig(state=target, lexerActionExecutor=lexerActionExecutor, config=cfg)
                    if self.closure(input, config, reach, currentAltReachedAcceptState, True, treatEofAsEpsilon):
                        skipAlt = cfg.alt

    def accept(self, input: InputStream, lexerActionExecutor: LexerActionExecutor, startIndex: int, index: int, line: int, charPos: int):
        if False:
            print('Hello World!')
        if LexerATNSimulator.debug:
            print('ACTION', lexerActionExecutor)
        input.seek(index)
        self.line = line
        self.column = charPos
        if lexerActionExecutor is not None and self.recog is not None:
            lexerActionExecutor.execute(self.recog, input, startIndex)

    def getReachableTarget(self, trans: Transition, t: int):
        if False:
            return 10
        if trans.matches(t, 0, self.MAX_CHAR_VALUE):
            return trans.target
        else:
            return None

    def computeStartState(self, input: InputStream, p: ATNState):
        if False:
            print('Hello World!')
        initialContext = PredictionContext.EMPTY
        configs = OrderedATNConfigSet()
        for i in range(0, len(p.transitions)):
            target = p.transitions[i].target
            c = LexerATNConfig(state=target, alt=i + 1, context=initialContext)
            self.closure(input, c, configs, False, False, False)
        return configs

    def closure(self, input: InputStream, config: LexerATNConfig, configs: ATNConfigSet, currentAltReachedAcceptState: bool, speculative: bool, treatEofAsEpsilon: bool):
        if False:
            for i in range(10):
                print('nop')
        if LexerATNSimulator.debug:
            print('closure(' + str(config) + ')')
        if isinstance(config.state, RuleStopState):
            if LexerATNSimulator.debug:
                if self.recog is not None:
                    print('closure at', self.recog.symbolicNames[config.state.ruleIndex], 'rule stop', str(config))
                else:
                    print('closure at rule stop', str(config))
            if config.context is None or config.context.hasEmptyPath():
                if config.context is None or config.context.isEmpty():
                    configs.add(config)
                    return True
                else:
                    configs.add(LexerATNConfig(state=config.state, config=config, context=PredictionContext.EMPTY))
                    currentAltReachedAcceptState = True
            if config.context is not None and (not config.context.isEmpty()):
                for i in range(0, len(config.context)):
                    if config.context.getReturnState(i) != PredictionContext.EMPTY_RETURN_STATE:
                        newContext = config.context.getParent(i)
                        returnState = self.atn.states[config.context.getReturnState(i)]
                        c = LexerATNConfig(state=returnState, config=config, context=newContext)
                        currentAltReachedAcceptState = self.closure(input, c, configs, currentAltReachedAcceptState, speculative, treatEofAsEpsilon)
            return currentAltReachedAcceptState
        if not config.state.epsilonOnlyTransitions:
            if not currentAltReachedAcceptState or not config.passedThroughNonGreedyDecision:
                configs.add(config)
        for t in config.state.transitions:
            c = self.getEpsilonTarget(input, config, t, configs, speculative, treatEofAsEpsilon)
            if c is not None:
                currentAltReachedAcceptState = self.closure(input, c, configs, currentAltReachedAcceptState, speculative, treatEofAsEpsilon)
        return currentAltReachedAcceptState

    def getEpsilonTarget(self, input: InputStream, config: LexerATNConfig, t: Transition, configs: ATNConfigSet, speculative: bool, treatEofAsEpsilon: bool):
        if False:
            while True:
                i = 10
        c = None
        if t.serializationType == Transition.RULE:
            newContext = SingletonPredictionContext.create(config.context, t.followState.stateNumber)
            c = LexerATNConfig(state=t.target, config=config, context=newContext)
        elif t.serializationType == Transition.PRECEDENCE:
            raise UnsupportedOperationException('Precedence predicates are not supported in lexers.')
        elif t.serializationType == Transition.PREDICATE:
            if LexerATNSimulator.debug:
                print('EVAL rule ' + str(t.ruleIndex) + ':' + str(t.predIndex))
            configs.hasSemanticContext = True
            if self.evaluatePredicate(input, t.ruleIndex, t.predIndex, speculative):
                c = LexerATNConfig(state=t.target, config=config)
        elif t.serializationType == Transition.ACTION:
            if config.context is None or config.context.hasEmptyPath():
                lexerActionExecutor = LexerActionExecutor.append(config.lexerActionExecutor, self.atn.lexerActions[t.actionIndex])
                c = LexerATNConfig(state=t.target, config=config, lexerActionExecutor=lexerActionExecutor)
            else:
                c = LexerATNConfig(state=t.target, config=config)
        elif t.serializationType == Transition.EPSILON:
            c = LexerATNConfig(state=t.target, config=config)
        elif t.serializationType in [Transition.ATOM, Transition.RANGE, Transition.SET]:
            if treatEofAsEpsilon:
                if t.matches(Token.EOF, 0, self.MAX_CHAR_VALUE):
                    c = LexerATNConfig(state=t.target, config=config)
        return c

    def evaluatePredicate(self, input: InputStream, ruleIndex: int, predIndex: int, speculative: bool):
        if False:
            print('Hello World!')
        if self.recog is None:
            return True
        if not speculative:
            return self.recog.sempred(None, ruleIndex, predIndex)
        savedcolumn = self.column
        savedLine = self.line
        index = input.index
        marker = input.mark()
        try:
            self.consume(input)
            return self.recog.sempred(None, ruleIndex, predIndex)
        finally:
            self.column = savedcolumn
            self.line = savedLine
            input.seek(index)
            input.release(marker)

    def captureSimState(self, settings: SimState, input: InputStream, dfaState: DFAState):
        if False:
            while True:
                i = 10
        settings.index = input.index
        settings.line = self.line
        settings.column = self.column
        settings.dfaState = dfaState

    def addDFAEdge(self, from_: DFAState, tk: int, to: DFAState=None, cfgs: ATNConfigSet=None) -> DFAState:
        if False:
            for i in range(10):
                print('nop')
        if to is None and cfgs is not None:
            suppressEdge = cfgs.hasSemanticContext
            cfgs.hasSemanticContext = False
            to = self.addDFAState(cfgs)
            if suppressEdge:
                return to
        if tk < self.MIN_DFA_EDGE or tk > self.MAX_DFA_EDGE:
            return to
        if LexerATNSimulator.debug:
            print('EDGE ' + str(from_) + ' -> ' + str(to) + ' upon ' + chr(tk))
        if from_.edges is None:
            from_.edges = [None] * (self.MAX_DFA_EDGE - self.MIN_DFA_EDGE + 1)
        from_.edges[tk - self.MIN_DFA_EDGE] = to
        return to

    def addDFAState(self, configs: ATNConfigSet) -> DFAState:
        if False:
            while True:
                i = 10
        proposed = DFAState(configs=configs)
        firstConfigWithRuleStopState = next((cfg for cfg in configs if isinstance(cfg.state, RuleStopState)), None)
        if firstConfigWithRuleStopState is not None:
            proposed.isAcceptState = True
            proposed.lexerActionExecutor = firstConfigWithRuleStopState.lexerActionExecutor
            proposed.prediction = self.atn.ruleToTokenType[firstConfigWithRuleStopState.state.ruleIndex]
        dfa = self.decisionToDFA[self.mode]
        existing = dfa.states.get(proposed, None)
        if existing is not None:
            return existing
        newState = proposed
        newState.stateNumber = len(dfa.states)
        configs.setReadonly(True)
        newState.configs = configs
        dfa.states[newState] = newState
        return newState

    def getDFA(self, mode: int):
        if False:
            print('Hello World!')
        return self.decisionToDFA[mode]

    def getText(self, input: InputStream):
        if False:
            return 10
        return input.getText(self.startIndex, input.index - 1)

    def consume(self, input: InputStream):
        if False:
            print('Hello World!')
        curChar = input.LA(1)
        if curChar == ord('\n'):
            self.line += 1
            self.column = 0
        else:
            self.column += 1
        input.consume()

    def getTokenName(self, t: int):
        if False:
            for i in range(10):
                print('nop')
        if t == -1:
            return 'EOF'
        else:
            return "'" + chr(t) + "'"
LexerATNSimulator.ERROR = DFAState(2147483647, ATNConfigSet())
del Lexer
from io import StringIO
from antlr4 import Parser, DFA
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.error.ErrorListener import ErrorListener

class DiagnosticErrorListener(ErrorListener):

    def __init__(self, exactOnly: bool=True):
        if False:
            print('Hello World!')
        self.exactOnly = exactOnly

    def reportAmbiguity(self, recognizer: Parser, dfa: DFA, startIndex: int, stopIndex: int, exact: bool, ambigAlts: set, configs: ATNConfigSet):
        if False:
            return 10
        if self.exactOnly and (not exact):
            return
        with StringIO() as buf:
            buf.write('reportAmbiguity d=')
            buf.write(self.getDecisionDescription(recognizer, dfa))
            buf.write(': ambigAlts=')
            buf.write(str(self.getConflictingAlts(ambigAlts, configs)))
            buf.write(", input='")
            buf.write(recognizer.getTokenStream().getText(startIndex, stopIndex))
            buf.write("'")
            recognizer.notifyErrorListeners(buf.getvalue())

    def reportAttemptingFullContext(self, recognizer: Parser, dfa: DFA, startIndex: int, stopIndex: int, conflictingAlts: set, configs: ATNConfigSet):
        if False:
            while True:
                i = 10
        with StringIO() as buf:
            buf.write('reportAttemptingFullContext d=')
            buf.write(self.getDecisionDescription(recognizer, dfa))
            buf.write(", input='")
            buf.write(recognizer.getTokenStream().getText(startIndex, stopIndex))
            buf.write("'")
            recognizer.notifyErrorListeners(buf.getvalue())

    def reportContextSensitivity(self, recognizer: Parser, dfa: DFA, startIndex: int, stopIndex: int, prediction: int, configs: ATNConfigSet):
        if False:
            while True:
                i = 10
        with StringIO() as buf:
            buf.write('reportContextSensitivity d=')
            buf.write(self.getDecisionDescription(recognizer, dfa))
            buf.write(", input='")
            buf.write(recognizer.getTokenStream().getText(startIndex, stopIndex))
            buf.write("'")
            recognizer.notifyErrorListeners(buf.getvalue())

    def getDecisionDescription(self, recognizer: Parser, dfa: DFA):
        if False:
            print('Hello World!')
        decision = dfa.decision
        ruleIndex = dfa.atnStartState.ruleIndex
        ruleNames = recognizer.ruleNames
        if ruleIndex < 0 or ruleIndex >= len(ruleNames):
            return str(decision)
        ruleName = ruleNames[ruleIndex]
        if ruleName is None or len(ruleName) == 0:
            return str(decision)
        return str(decision) + ' (' + ruleName + ')'

    def getConflictingAlts(self, reportedAlts: set, configs: ATNConfigSet):
        if False:
            return 10
        if reportedAlts is not None:
            return reportedAlts
        result = set()
        for config in configs:
            result.add(config.alt)
        return result
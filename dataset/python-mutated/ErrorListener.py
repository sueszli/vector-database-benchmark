import sys

class ErrorListener(object):

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        if False:
            for i in range(10):
                print('nop')
        pass

    def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
        if False:
            print('Hello World!')
        pass

    def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
        if False:
            i = 10
            return i + 15
        pass

class ConsoleErrorListener(ErrorListener):
    INSTANCE = None

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        if False:
            for i in range(10):
                print('nop')
        print('line ' + str(line) + ':' + str(column) + ' ' + msg, file=sys.stderr)
ConsoleErrorListener.INSTANCE = ConsoleErrorListener()

class ProxyErrorListener(ErrorListener):

    def __init__(self, delegates):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if delegates is None:
            raise ReferenceError('delegates')
        self.delegates = delegates

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        if False:
            return 10
        for delegate in self.delegates:
            delegate.syntaxError(recognizer, offendingSymbol, line, column, msg, e)

    def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
        if False:
            return 10
        for delegate in self.delegates:
            delegate.reportAmbiguity(recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs)

    def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
        if False:
            i = 10
            return i + 15
        for delegate in self.delegates:
            delegate.reportAttemptingFullContext(recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs)

    def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
        if False:
            print('Hello World!')
        for delegate in self.delegates:
            delegate.reportContextSensitivity(recognizer, dfa, startIndex, stopIndex, prediction, configs)
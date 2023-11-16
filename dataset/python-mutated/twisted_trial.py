from zope.interface import implementer
from twisted.plugin import IPlugin
from twisted.trial.itrial import IReporter

@implementer(IPlugin, IReporter)
class _Reporter:

    def __init__(self, name, module, description, longOpt, shortOpt, klass):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.module = module
        self.description = description
        self.longOpt = longOpt
        self.shortOpt = shortOpt
        self.klass = klass

    @property
    def stream(self):
        if False:
            i = 10
            return i + 15
        pass

    @property
    def tbformat(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    def args(self):
        if False:
            i = 10
            return i + 15
        pass

    @property
    def shouldStop(self):
        if False:
            while True:
                i = 10
        pass

    @property
    def separator(self):
        if False:
            return 10
        pass

    @property
    def testsRun(self):
        if False:
            i = 10
            return i + 15
        pass

    def addError(self, test, error):
        if False:
            return 10
        pass

    def addExpectedFailure(self, test, failure, todo=None):
        if False:
            while True:
                i = 10
        pass

    def addFailure(self, test, failure):
        if False:
            return 10
        pass

    def addSkip(self, test, reason):
        if False:
            i = 10
            return i + 15
        pass

    def addSuccess(self, test):
        if False:
            i = 10
            return i + 15
        pass

    def addUnexpectedSuccess(self, test, todo=None):
        if False:
            i = 10
            return i + 15
        pass

    def cleanupErrors(self, errs):
        if False:
            while True:
                i = 10
        pass

    def done(self):
        if False:
            print('Hello World!')
        pass

    def endSuite(self, name):
        if False:
            while True:
                i = 10
        pass

    def printErrors(self):
        if False:
            i = 10
            return i + 15
        pass

    def printSummary(self):
        if False:
            print('Hello World!')
        pass

    def startSuite(self, name):
        if False:
            print('Hello World!')
        pass

    def startTest(self, method):
        if False:
            print('Hello World!')
        pass

    def stopTest(self, method):
        if False:
            print('Hello World!')
        pass

    def upDownError(self, userMeth, warn=True, printStatus=True):
        if False:
            for i in range(10):
                print('nop')
        pass

    def wasSuccessful(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def write(self, string):
        if False:
            i = 10
            return i + 15
        pass

    def writeln(self, string):
        if False:
            while True:
                i = 10
        pass
Tree = _Reporter('Tree Reporter', 'twisted.trial.reporter', description='verbose color output (default reporter)', longOpt='verbose', shortOpt='v', klass='TreeReporter')
BlackAndWhite = _Reporter('Black-And-White Reporter', 'twisted.trial.reporter', description='Colorless verbose output', longOpt='bwverbose', shortOpt='o', klass='VerboseTextReporter')
Minimal = _Reporter('Minimal Reporter', 'twisted.trial.reporter', description='minimal summary output', longOpt='summary', shortOpt='s', klass='MinimalReporter')
Classic = _Reporter('Classic Reporter', 'twisted.trial.reporter', description='terse text output', longOpt='text', shortOpt='t', klass='TextReporter')
Timing = _Reporter('Timing Reporter', 'twisted.trial.reporter', description='Timing output', longOpt='timing', shortOpt=None, klass='TimingTextReporter')
Subunit = _Reporter('Subunit Reporter', 'twisted.trial.reporter', description='subunit output', longOpt='subunit', shortOpt=None, klass='SubunitReporter')
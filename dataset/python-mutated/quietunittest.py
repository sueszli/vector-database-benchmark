import os
import sys
from io import IOBase
from unittest import TestResult, TextTestRunner
from unittest.main import main
__unittest = True
if os.getenv('CI'):
    sys.path.remove(os.path.split(__file__)[0])

class PreambuleStream(IOBase):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.preambule = ''
        self.line_before_msg = False

    def set_preambule(self, preambule):
        if False:
            print('Hello World!')
        self.preambule = preambule

    def writable(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def writelines(self, lines):
        if False:
            while True:
                i = 10
        return self.write(''.join(lines))

    def write(self, s):
        if False:
            return 10
        if self.preambule:
            _stdout.write('\n' + self.preambule + '\n')
            self.preambule = ''
        self.line_before_msg = True
        return _stdout.write(s)

    def write_msg(self, s):
        if False:
            while True:
                i = 10
        if self.line_before_msg:
            _stdout.write('\n')
        _stdout.write(self.preambule + ' ... ' + s)
        self.line_before_msg = False
        self.preambule = ''

    def flush(self):
        if False:
            print('Hello World!')
        _stdout.flush()

class QuietTestResult(TestResult):
    separator1 = '=' * 70
    separator2 = '-' * 70

    def startTest(self, test):
        if False:
            print('Hello World!')
        super().startTest(test)
        sys.stdout.set_preambule(self.getDescription(test))

    def stopTest(self, test):
        if False:
            print('Hello World!')
        super().stopTest(test)
        sys.stdout.set_preambule('')

    @staticmethod
    def getDescription(test):
        if False:
            i = 10
            return i + 15
        doc_first_line = test.shortDescription()
        if doc_first_line:
            return '\n'.join((str(test), doc_first_line))
        else:
            return str(test)

    def addError(self, test, err):
        if False:
            print('Hello World!')
        super().addError(test, err)
        sys.stdout.write_msg('ERROR\n')

    def addFailure(self, test, err):
        if False:
            for i in range(10):
                print('nop')
        super().addError(test, err)
        sys.stdout.write_msg('FAIL\n')

    def printErrors(self):
        if False:
            print('Hello World!')
        sys.stdout.set_preambule('')
        print()
        self.printErrorList('ERROR', self.errors)
        self.printErrorList('FAIL', self.failures)

    def printErrorList(self, flavour, errors):
        if False:
            while True:
                i = 10
        for (test, err) in errors:
            print(self.separator1)
            print('%s: %s' % (flavour, self.getDescription(test)))
            print(self.separator2)
            print('%s' % err)
_stdout = sys.stdout
sys.stderr = sys.stdout = PreambuleStream()
testRunner = TextTestRunner(resultclass=QuietTestResult, stream=sys.stdout, verbosity=2)
main(module=None, testRunner=testRunner)
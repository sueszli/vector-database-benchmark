from zope.interface import implementer
from buildbot import interfaces

@implementer(interfaces.ILogObserver)
class LogObserver:

    def setStep(self, step):
        if False:
            while True:
                i = 10
        self.step = step

    def setLog(self, loog):
        if False:
            while True:
                i = 10
        loog.subscribe(self.gotData)

    def gotData(self, stream, data):
        if False:
            while True:
                i = 10
        if data is None:
            self.finishReceived()
        elif stream is None or stream == 'o':
            self.outReceived(data)
        elif stream == 'e':
            self.errReceived(data)
        elif stream == 'h':
            self.headerReceived(data)

    def finishReceived(self):
        if False:
            print('Hello World!')
        pass

    def outReceived(self, data):
        if False:
            i = 10
            return i + 15
        pass

    def errReceived(self, data):
        if False:
            while True:
                i = 10
        pass

    def headerReceived(self, data):
        if False:
            while True:
                i = 10
        pass

class LogLineObserver(LogObserver):
    stdoutDelimiter = '\n'
    stderrDelimiter = '\n'
    headerDelimiter = '\n'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.max_length = 16384

    def setMaxLineLength(self, max_length):
        if False:
            print('Hello World!')
        '\n        Set the maximum line length: lines longer than max_length are\n        dropped.  Default is 16384 bytes.  Use sys.maxint for effective\n        infinity.\n        '
        self.max_length = max_length

    def _lineReceived(self, data, delimiter, funcReceived):
        if False:
            print('Hello World!')
        for line in data.rstrip().split(delimiter):
            if len(line) > self.max_length:
                continue
            funcReceived(line)

    def outReceived(self, data):
        if False:
            i = 10
            return i + 15
        self._lineReceived(data, self.stdoutDelimiter, self.outLineReceived)

    def errReceived(self, data):
        if False:
            i = 10
            return i + 15
        self._lineReceived(data, self.stderrDelimiter, self.errLineReceived)

    def headerReceived(self, data):
        if False:
            print('Hello World!')
        self._lineReceived(data, self.headerDelimiter, self.headerLineReceived)

    def outLineReceived(self, line):
        if False:
            print('Hello World!')
        'This will be called with complete stdout lines (not including the\n        delimiter). Override this in your observer.'

    def errLineReceived(self, line):
        if False:
            i = 10
            return i + 15
        'This will be called with complete lines of stderr (not including\n        the delimiter). Override this in your observer.'

    def headerLineReceived(self, line):
        if False:
            return 10
        'This will be called with complete lines of stderr (not including\n        the delimiter). Override this in your observer.'

class LineConsumerLogObserver(LogLineObserver):

    def __init__(self, consumerFunction):
        if False:
            while True:
                i = 10
        super().__init__()
        self.generator = None
        self.consumerFunction = consumerFunction

    def feed(self, input):
        if False:
            print('Hello World!')
        self.generator = self.consumerFunction()
        next(self.generator)
        self.feed = self.generator.send
        self.feed(input)

    def outLineReceived(self, line):
        if False:
            for i in range(10):
                print('nop')
        self.feed(('o', line))

    def errLineReceived(self, line):
        if False:
            i = 10
            return i + 15
        self.feed(('e', line))

    def headerLineReceived(self, line):
        if False:
            i = 10
            return i + 15
        self.feed(('h', line))

    def finishReceived(self):
        if False:
            for i in range(10):
                print('nop')
        if self.generator:
            self.generator.close()

class OutputProgressObserver(LogObserver):
    length = 0

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name

    def gotData(self, stream, data):
        if False:
            return 10
        if data:
            self.length += len(data)
        self.step.setProgress(self.name, self.length)

class BufferLogObserver(LogObserver):

    def __init__(self, wantStdout=True, wantStderr=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.stdout = [] if wantStdout else None
        self.stderr = [] if wantStderr else None

    def outReceived(self, data):
        if False:
            while True:
                i = 10
        if self.stdout is not None:
            self.stdout.append(data)

    def errReceived(self, data):
        if False:
            print('Hello World!')
        if self.stderr is not None:
            self.stderr.append(data)

    def _get(self, chunks):
        if False:
            print('Hello World!')
        if chunks is None or not chunks:
            return ''
        return ''.join(chunks)

    def getStdout(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get(self.stdout)

    def getStderr(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get(self.stderr)
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
try:
    import syslog as _stdsyslog
except ImportError:
    stdsyslog = None
else:
    stdsyslog = _stdsyslog
    from twisted.python import syslog

class SyslogObserverTests(TestCase):
    """
    Tests for L{SyslogObserver} which sends Twisted log events to the syslog.
    """
    events = None
    if stdsyslog is None:
        skip = 'syslog is not supported on this platform'

    def setUp(self):
        if False:
            print('Hello World!')
        self.patch(syslog.SyslogObserver, 'openlog', self.openlog)
        self.patch(syslog.SyslogObserver, 'syslog', self.syslog)
        self.observer = syslog.SyslogObserver('SyslogObserverTests')

    def openlog(self, prefix, options, facility):
        if False:
            return 10
        self.logOpened = (prefix, options, facility)
        self.events = []

    def syslog(self, options, message):
        if False:
            return 10
        self.events.append((options, message))

    def test_emitWithoutMessage(self):
        if False:
            while True:
                i = 10
        "\n        L{SyslogObserver.emit} ignores events with an empty value for the\n        C{'message'} key.\n        "
        self.observer.emit({'message': (), 'isError': False, 'system': '-'})
        self.assertEqual(self.events, [])

    def test_emitCustomPriority(self):
        if False:
            print('Hello World!')
        "\n        L{SyslogObserver.emit} uses the value of the C{'syslogPriority'} as the\n        syslog priority, if that key is present in the event dictionary.\n        "
        self.observer.emit({'message': ('hello, world',), 'isError': False, 'system': '-', 'syslogPriority': stdsyslog.LOG_DEBUG})
        self.assertEqual(self.events, [(stdsyslog.LOG_DEBUG, '[-] hello, world')])

    def test_emitErrorPriority(self):
        if False:
            return 10
        '\n        L{SyslogObserver.emit} uses C{LOG_ALERT} if the event represents an\n        error.\n        '
        self.observer.emit({'message': ('hello, world',), 'isError': True, 'system': '-', 'failure': Failure(Exception('foo'))})
        self.assertEqual(self.events, [(stdsyslog.LOG_ALERT, '[-] hello, world')])

    def test_emitCustomPriorityOverridesError(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{SyslogObserver.emit} uses the value of the C{'syslogPriority'} key if\n        it is specified even if the event dictionary represents an error.\n        "
        self.observer.emit({'message': ('hello, world',), 'isError': True, 'system': '-', 'syslogPriority': stdsyslog.LOG_NOTICE, 'failure': Failure(Exception('bar'))})
        self.assertEqual(self.events, [(stdsyslog.LOG_NOTICE, '[-] hello, world')])

    def test_emitCustomFacility(self):
        if False:
            while True:
                i = 10
        "\n        L{SyslogObserver.emit} uses the value of the C{'syslogPriority'} as the\n        syslog priority, if that key is present in the event dictionary.\n        "
        self.observer.emit({'message': ('hello, world',), 'isError': False, 'system': '-', 'syslogFacility': stdsyslog.LOG_CRON})
        self.assertEqual(self.events, [(stdsyslog.LOG_INFO | stdsyslog.LOG_CRON, '[-] hello, world')])

    def test_emitCustomSystem(self):
        if False:
            print('Hello World!')
        "\n        L{SyslogObserver.emit} uses the value of the C{'system'} key to prefix\n        the logged message.\n        "
        self.observer.emit({'message': ('hello, world',), 'isError': False, 'system': 'nonDefaultSystem'})
        self.assertEqual(self.events, [(stdsyslog.LOG_INFO, '[nonDefaultSystem] hello, world')])

    def test_emitMessage(self):
        if False:
            print('Hello World!')
        "\n        L{SyslogObserver.emit} logs the value of the C{'message'} key of the\n        event dictionary it is passed to the syslog.\n        "
        self.observer.emit({'message': ('hello, world',), 'isError': False, 'system': '-'})
        self.assertEqual(self.events, [(stdsyslog.LOG_INFO, '[-] hello, world')])

    def test_emitMultilineMessage(self):
        if False:
            i = 10
            return i + 15
        '\n        Each line of a multiline message is emitted separately to the syslog.\n        '
        self.observer.emit({'message': ('hello,\nworld',), 'isError': False, 'system': '-'})
        self.assertEqual(self.events, [(stdsyslog.LOG_INFO, '[-] hello,'), (stdsyslog.LOG_INFO, '[-] \tworld')])

    def test_emitStripsTrailingEmptyLines(self):
        if False:
            return 10
        '\n        Trailing empty lines of a multiline message are omitted from the\n        messages sent to the syslog.\n        '
        self.observer.emit({'message': ('hello,\nworld\n\n',), 'isError': False, 'system': '-'})
        self.assertEqual(self.events, [(stdsyslog.LOG_INFO, '[-] hello,'), (stdsyslog.LOG_INFO, '[-] \tworld')])
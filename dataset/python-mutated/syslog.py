"""
Classes and utility functions for integrating Twisted and syslog.

You probably want to call L{startLogging}.
"""
syslog = __import__('syslog')
from twisted.python import log
DEFAULT_OPTIONS = 0
DEFAULT_FACILITY = syslog.LOG_USER

class SyslogObserver:
    """
    A log observer for logging to syslog.

    See L{twisted.python.log} for context.

    This logObserver will automatically use LOG_ALERT priority for logged
    failures (such as from C{log.err()}), but you can use any priority and
    facility by setting the 'C{syslogPriority}' and 'C{syslogFacility}' keys in
    the event dict.
    """
    openlog = syslog.openlog
    syslog = syslog.syslog

    def __init__(self, prefix, options=DEFAULT_OPTIONS, facility=DEFAULT_FACILITY):
        if False:
            for i in range(10):
                print('nop')
        '\n        @type prefix: C{str}\n        @param prefix: The syslog prefix to use.\n\n        @type options: C{int}\n        @param options: A bitvector represented as an integer of the syslog\n            options to use.\n\n        @type facility: C{int}\n        @param facility: An indication to the syslog daemon of what sort of\n            program this is (essentially, an additional arbitrary metadata\n            classification for messages sent to syslog by this observer).\n        '
        self.openlog(prefix, options, facility)

    def emit(self, eventDict):
        if False:
            while True:
                i = 10
        "\n        Send a message event to the I{syslog}.\n\n        @param eventDict: The event to send.  If it has no C{'message'} key, it\n            will be ignored.  Otherwise, if it has C{'syslogPriority'} and/or\n            C{'syslogFacility'} keys, these will be used as the syslog priority\n            and facility.  If it has no C{'syslogPriority'} key but a true\n            value for the C{'isError'} key, the B{LOG_ALERT} priority will be\n            used; if it has a false value for C{'isError'}, B{LOG_INFO} will be\n            used.  If the C{'message'} key is multiline, each line will be sent\n            to the syslog separately.\n        "
        text = log.textFromEventDict(eventDict)
        if text is None:
            return
        priority = syslog.LOG_INFO
        facility = 0
        if eventDict['isError']:
            priority = syslog.LOG_ALERT
        if 'syslogPriority' in eventDict:
            priority = int(eventDict['syslogPriority'])
        if 'syslogFacility' in eventDict:
            facility = int(eventDict['syslogFacility'])
        lines = text.split('\n')
        while lines[-1:] == ['']:
            lines.pop()
        firstLine = True
        for line in lines:
            if firstLine:
                firstLine = False
            else:
                line = '\t' + line
            self.syslog(priority | facility, '[{}] {}'.format(eventDict['system'], line))

def startLogging(prefix='Twisted', options=DEFAULT_OPTIONS, facility=DEFAULT_FACILITY, setStdout=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Send all Twisted logging output to syslog from now on.\n\n    The prefix, options and facility arguments are passed to\n    C{syslog.openlog()}, see the Python syslog documentation for details. For\n    other parameters, see L{twisted.python.log.startLoggingWithObserver}.\n    '
    obs = SyslogObserver(prefix, options, facility)
    log.startLoggingWithObserver(obs.emit, setStdout=setStdout)
"""
Example of an interface to Courier's mail filter.
"""
LOGFILE = '/tmp/filter.log'
from twisted.python import log
log.startLogging(open(LOGFILE, 'a'))
import sys
sys.stderr = log.logfile
from twisted.internet import reactor, stdio
from twisted.internet.protocol import Factory, Protocol
from twisted.protocols import basic
FILTERS = '/var/lib/courier/filters'
ALLFILTERS = '/var/lib/courier/allfilters'
FILTERNAME = 'twistedfilter'
import email.message
import email.parser
import os
import os.path
from syslog import LOG_MAIL, openlog, syslog

def trace_dump():
    if False:
        while True:
            i = 10
    (t, v, tb) = sys.exc_info()
    openlog(FILTERNAME, 0, LOG_MAIL)
    syslog(f'Unhandled exception: {v} - {t}')
    while tb:
        syslog('Trace: {}:{} {}'.format(tb.tb_frame.f_code.co_filename, tb.tb_frame.f_code.co_name, tb.tb_lineno))
        tb = tb.tb_next
    del tb

def safe_del(file):
    if False:
        for i in range(10):
            print('nop')
    try:
        if os.path.isdir(file):
            os.removedirs(file)
        else:
            os.remove(file)
    except OSError:
        pass

class DieWhenLost(Protocol):

    def connectionLost(self, reason=None):
        if False:
            for i in range(10):
                print('nop')
        reactor.stop()

class MailProcessor(basic.LineReceiver):
    """
    I process a mail message.

    Override filterMessage to do any filtering you want.
    """
    messageFilename = None
    delimiter = '\n'

    def connectionMade(self):
        if False:
            return 10
        log.msg(f'Connection from {self.transport}')
        self.state = 'connected'
        self.metaInfo = []

    def lineReceived(self, line):
        if False:
            while True:
                i = 10
        if self.state == 'connected':
            self.messageFilename = line
            self.state = 'gotMessageFilename'
        if self.state == 'gotMessageFilename':
            if line:
                self.metaInfo.append(line)
            else:
                if not self.metaInfo:
                    self.transport.loseConnection()
                    return
                self.filterMessage()

    def filterMessage(self):
        if False:
            return 10
        'Override this.\n\n        A trivial example is included.\n        '
        try:
            emailParser = email.parser.Parser()
            with open(self.messageFilename) as f:
                emailParser.parse(f)
            self.sendLine(b'200 Ok')
        except BaseException:
            trace_dump()
            self.sendLine(b'435 ' + FILTERNAME.encode('ascii') + b' processing error')

def main():
    if False:
        return 10
    f = Factory()
    f.protocol = MailProcessor
    safe_del(f'{ALLFILTERS}/{FILTERNAME}')
    reactor.listenUNIX(f'{ALLFILTERS}/{FILTERNAME}', f, 10)
    reactor.callLater(0, os.close, 3)
    stdio.StandardIO(DieWhenLost())
    reactor.run()
if __name__ == '__main__':
    main()
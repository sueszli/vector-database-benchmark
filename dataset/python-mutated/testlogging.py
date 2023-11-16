"""Test logging.

Message should only be printed second time around.
"""
import sys
import warnings
from twisted.internet import reactor
from twisted.python import log

def test(i):
    if False:
        i = 10
        return i + 15
    print('printed', i)
    log.msg(f'message {i}')
    warnings.warn(f'warning {i}')
    try:
        raise RuntimeError(f'error {i}')
    except BaseException:
        log.err()

def startlog():
    if False:
        for i in range(10):
            print('nop')
    log.startLogging(sys.stdout)

def end():
    if False:
        return 10
    reactor.stop()
test(1)
reactor.callLater(0.1, test, 2)
reactor.callLater(0.2, startlog)
reactor.callLater(0.3, test, 3)
reactor.callLater(0.4, end)
reactor.run()
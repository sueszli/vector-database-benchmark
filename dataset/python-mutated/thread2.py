import string, sys, time
try:
    from _thread import get_ident
except:
    from thread import get_ident
from threading import Thread, Lock
import libxml2
THREADS_COUNT = 15
failed = 0

class ErrorHandler:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.errors = []
        self.lock = Lock()

    def handler(self, ctx, str):
        if False:
            return 10
        self.lock.acquire()
        self.errors.append(str)
        self.lock.release()

def getLineNumbersDefault():
    if False:
        for i in range(10):
            print('nop')
    old = libxml2.lineNumbersDefault(0)
    libxml2.lineNumbersDefault(old)
    return old

def test(expectedLineNumbersDefault):
    if False:
        i = 10
        return i + 15
    time.sleep(1)
    global failed
    if expectedLineNumbersDefault != getLineNumbersDefault():
        failed = 1
        print('FAILED to obtain correct value for lineNumbersDefault in thread %d' % get_ident())
    try:
        doc = libxml2.parseFile('bad.xml')
    except:
        pass
    else:
        assert 'failed'
eh = ErrorHandler()
libxml2.registerErrorHandler(eh.handler, '')
libxml2.lineNumbersDefault(1)
test(1)
ec = len(eh.errors)
if ec == 0:
    print('FAILED: should have obtained errors')
    sys.exit(1)
ts = []
for i in range(THREADS_COUNT):
    ts.append(Thread(target=test, args=(0,)))
for t in ts:
    t.start()
for t in ts:
    t.join()
if len(eh.errors) != ec + THREADS_COUNT * ec:
    print('FAILED: did not obtain the correct number of errors')
    sys.exit(1)
libxml2.thrDefLineNumbersDefaultValue(1)
ts = []
for i in range(THREADS_COUNT):
    ts.append(Thread(target=test, args=(1,)))
for t in ts:
    t.start()
for t in ts:
    t.join()
if len(eh.errors) != ec + THREADS_COUNT * ec * 2:
    print('FAILED: did not obtain the correct number of errors')
    sys.exit(1)
if failed:
    print('FAILED')
    sys.exit(1)
libxml2.cleanupParser()
if libxml2.debugMemory(1) == 0:
    print('OK')
else:
    print('Memory leak %d bytes' % libxml2.debugMemory(1))
    libxml2.dumpMemory()
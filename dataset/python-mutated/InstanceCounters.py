""" Instance counter primitives

We don't use a meta class as it's unnecessary complex, and portable meta classes
have their difficulties, and want to count classes, who already have a meta
class.

This is going to expanded with time.

"""
from nuitka.Options import isShowMemory
from nuitka.Tracing import printIndented, printLine
counted_inits = {}
counted_dels = {}

def isCountingInstances():
    if False:
        for i in range(10):
            print('nop')
    return isShowMemory()

def counted_init(init):
    if False:
        while True:
            i = 10
    if isShowMemory():

        def wrapped_init(self, *args, **kw):
            if False:
                for i in range(10):
                    print('nop')
            name = self.__class__.__name__
            assert type(name) is str
            if name not in counted_inits:
                counted_inits[name] = 0
            counted_inits[name] += 1
            init(self, *args, **kw)
        return wrapped_init
    else:
        return init

def _wrapped_del(self):
    if False:
        i = 10
        return i + 15
    if counted_dels is None:
        return
    name = self.__class__.__name__
    assert type(name) is str
    if name not in counted_dels:
        counted_dels[name] = 0
    counted_dels[name] += 1

def counted_del():
    if False:
        for i in range(10):
            print('nop')
    assert isShowMemory()
    return _wrapped_del

def printStats():
    if False:
        i = 10
        return i + 15
    printLine('Init/del/alive calls:')
    for (name, count) in sorted(counted_inits.items()):
        dels = counted_dels.get(name, 0)
        printIndented(1, name, count, dels, count - dels)
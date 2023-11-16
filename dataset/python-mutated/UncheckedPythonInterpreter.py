import sys
__saved_context__ = {}

def saveContext():
    if False:
        i = 10
        return i + 15
    __saved_context__.update(sys.modules[__name__].__dict__)

def restoreContext():
    if False:
        for i in range(10):
            print('nop')
    names = list(sys.modules[__name__].__dict__.keys())
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]
saveContext()
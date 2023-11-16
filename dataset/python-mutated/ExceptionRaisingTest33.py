import sys
print('Testing exception changes between generator switches:')

def yieldExceptionInteraction():
    if False:
        for i in range(10):
            print('nop')

    def yield_raise():
        if False:
            for i in range(10):
                print('nop')
        try:
            raise KeyError('caught')
        except KeyError:
            yield from sys.exc_info()
            yield from sys.exc_info()
        yield from sys.exc_info()
    g = yield_raise()
    print('Initial yield from catch in generator', next(g))
    print('Checking from outside of generator', sys.exc_info()[0])
    print('Second yield from the catch reentered', next(g))
    print('Checking from outside of generator', sys.exc_info()[0])
    print('After leaving the catch generator yielded', next(g))
yieldExceptionInteraction()
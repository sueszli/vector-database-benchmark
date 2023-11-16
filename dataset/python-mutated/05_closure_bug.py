from functools import wraps

def contextmanager(func):
    if False:
        print('Hello World!')

    @wraps(func)
    def helper(*args, **kwds):
        if False:
            print('Hello World!')
        return _GeneratorContextManager(func, *args, **kwds)
    return helper
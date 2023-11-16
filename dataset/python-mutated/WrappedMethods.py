from functools import wraps

def decorator(f):
    if False:
        return 10

    @wraps(f)
    def wrapper(*args, **kws):
        if False:
            print('Hello World!')
        return f(*args, **kws)
    return wrapper

class WrappedMethods:

    @decorator
    def wrapped_method(self):
        if False:
            while True:
                i = 10
        pass

    @decorator
    def wrapped_method_with_arguments(self, a, b=2):
        if False:
            while True:
                i = 10
        pass
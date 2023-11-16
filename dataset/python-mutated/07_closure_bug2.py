from functools import wraps

class ContextDecorator(object):

    def __call__(self, func):
        if False:
            return 10

        @wraps(func)
        def inner(*args, **kwds):
            if False:
                for i in range(10):
                    print('nop')
            with self._recreate_cm():
                return func(*args, **kwds)
        return inner
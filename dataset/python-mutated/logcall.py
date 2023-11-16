from functools import wraps

def logformat(fmt):
    if False:
        return 10

    def logged(func):
        if False:
            while True:
                i = 10
        print('Adding logging to', func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            print(fmt.format(func=func))
            return func(*args, **kwargs)
        return wrapper
    return logged
logged = logformat('Calling {func.__name__}')
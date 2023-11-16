from functools import wraps

def decorator(f):
    if False:
        i = 10
        return i + 15

    @wraps(f)
    def wrapper(*args, **kws):
        if False:
            print('Hello World!')
        return f(*args, **kws)
    return wrapper

@decorator
def wrapped_function():
    if False:
        for i in range(10):
            print('nop')
    pass

@decorator
def wrapped_function_with_arguments(a, b=2):
    if False:
        return 10
    pass
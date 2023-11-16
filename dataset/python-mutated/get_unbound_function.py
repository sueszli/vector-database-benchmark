def get_unbound_function(func):
    if False:
        while True:
            i = 10
    if not getattr(func, '__self__', True):
        return func.__func__
    return func
def decorator_factory(foo):
    if False:
        while True:
            i = 10

    def decorator(function):
        if False:
            return 10

        def function_wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return function(*args, **kwargs)
        return function_wrapper
    return decorator

@decorator_factory('bar')
def test():
    if False:
        for i in range(10):
            print('nop')
    ' Simple reproducer. '
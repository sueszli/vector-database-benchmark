import contextlib
from ulauncher.utils.timer import timer

def debounce(wait):
    if False:
        while True:
            i = 10
    'Decorator that will postpone a functions\n    execution until after wait seconds\n    have elapsed since the last time it was invoked.'

    def decorator(fn):
        if False:
            return 10

        def debounced(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')

            def call_it():
                if False:
                    i = 10
                    return i + 15
                fn(*args, **kwargs)
            with contextlib.suppress(AttributeError):
                debounced.t.cancel()
            debounced.t = timer(wait, call_it)
        return debounced
    return decorator
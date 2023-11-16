import time
from functools import wraps
from inspect import signature
import pkg_resources

def entry_point(name):
    if False:
        while True:
            i = 10

    def inner_function(func):
        if False:
            while True:
                i = 10

        @wraps(func)
        def function_wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            'function_wrapper of greeting'
            on_call_kwargs = kwargs.copy()
            sig = signature(func)
            for (arg, parameter) in zip(args, sig.parameters):
                on_call_kwargs[parameter] = arg
            entry_points = []
            for entry_point in pkg_resources.iter_entry_points(name):
                entry_point = entry_point.load()
                entry_points.append(entry_point())
            for ep in entry_points:
                ep.on_call(on_call_kwargs)
            try:
                start = time.time()
                return_value = func(*args, **kwargs)
                runtime = time.time() - start
            except Exception as e:
                runtime = time.time() - start
                for ep in entry_points:
                    ep.on_error(error=e, runtime=runtime)
                raise e
            for ep in entry_points:
                ep.on_return(return_value=return_value, runtime=runtime)
            return return_value
        return function_wrapper
    return inner_function
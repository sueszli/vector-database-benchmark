import sys
from typing import Any, Callable, Iterable, List, Tuple
__all__ = ['trace_dependencies']

def trace_dependencies(callable: Callable[[Any], Any], inputs: Iterable[Tuple[Any, ...]]) -> List[str]:
    if False:
        i = 10
        return i + 15
    "Trace the execution of a callable in order to determine which modules it uses.\n\n    Args:\n        callable: The callable to execute and trace.\n        inputs: The input to use during tracing. The modules used by 'callable' when invoked by each set of inputs\n            are union-ed to determine all modules used by the callable for the purpooses of packaging.\n\n    Returns: A list of the names of all modules used during callable execution.\n    "
    modules_used = set()

    def record_used_modules(frame, event, arg):
        if False:
            print('Hello World!')
        if event != 'call':
            return
        name = frame.f_code.co_name
        module = None
        if name in frame.f_globals:
            module = frame.f_globals[name].__module__
        elif name in frame.f_locals:
            module = frame.f_locals[name].__module__
        elif 'self' in frame.f_locals:
            method = getattr(frame.f_locals['self'], name, None)
            module = method.__module__ if method else None
        if module:
            modules_used.add(module)
    try:
        sys.setprofile(record_used_modules)
        for inp in inputs:
            callable(*inp)
    finally:
        sys.setprofile(None)
    return list(modules_used)
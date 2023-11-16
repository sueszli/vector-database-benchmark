import inspect
from pyodide.ffi.wrappers import add_event_listener
from pyscript.magic_js import document

def when(event_type=None, selector=None):
    if False:
        return 10
    '\n    Decorates a function and passes py-* events to the decorated function\n    The events might or not be an argument of the decorated function\n    '

    def decorator(func):
        if False:
            while True:
                i = 10
        if isinstance(selector, str):
            elements = document.querySelectorAll(selector)
        else:
            from pyweb import pydom
            if isinstance(selector, pydom.Element):
                elements = [selector._js]
            elif isinstance(selector, pydom.ElementCollection):
                elements = [el._js for el in selector]
            else:
                raise ValueError(f'Invalid selector: {selector}. Selector must be a string, a pydom.Element or a pydom.ElementCollection.')
        sig = inspect.signature(func)
        if not sig.parameters:

            def wrapper(*args, **kwargs):
                if False:
                    return 10
                func()
            for el in elements:
                add_event_listener(el, event_type, wrapper)
        else:
            for el in elements:
                add_event_listener(el, event_type, func)
        return func
    return decorator
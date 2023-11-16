"""Generic utilities not strictly related to autograph that are moved here or just implemented
as no-op placeholders if the actual functionality doesn't matter - for example the scope of API
export and it's management is not important if we import the autograph as internal symbol.
"""
import inspect

def _remove_undocumented(module_name, allowed_exception_list=None, doc_string_modules=None):
    if False:
        for i in range(10):
            print('nop')
    pass

def export_symbol(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    'No-op replacement for @tf_export. This is decorator factory that accepts arguments'

    def actual_decorator(function):
        if False:
            return 10
        return function
    return actual_decorator

def make_decorator(target, decorator_func, decorator_name=None, decorator_doc='', decorator_argspec=None):
    if False:
        while True:
            i = 10
    'Make a decorator from a wrapper and a target.\n\n  Args:\n    target: The final callable to be wrapped.\n    decorator_func: The wrapper function.\n    decorator_name: The name of the decorator. If `None`, the name of the\n      function calling make_decorator.\n    decorator_doc: Documentation specific to this application of\n      `decorator_func` to `target`.\n    decorator_argspec: The new callable signature of this decorator.\n\n  Returns:\n    The `decorator_func` argument with new metadata attached.\n    Note that we just wrap the function and adjust the members but do not insert the special\n    member TFDecorator\n  '
    if decorator_name is None:
        decorator_name = inspect.currentframe().f_back.f_code.co_name
    if hasattr(target, '__name__'):
        decorator_func.__name__ = target.__name__
    if hasattr(target, '__qualname__'):
        decorator_func.__qualname__ = target.__qualname__
    if hasattr(target, '__module__'):
        decorator_func.__module__ = target.__module__
    if hasattr(target, '__dict__'):
        for name in target.__dict__:
            if name not in decorator_func.__dict__:
                decorator_func.__dict__[name] = target.__dict__[name]
    decorator_func.__wrapped__ = target
    decorator_func.__original_wrapped__ = target
    return decorator_func

def custom_constant(val, shape=None, dtype=None):
    if False:
        for i in range(10):
            print('nop')
    'Customization point to introduce library-specific argument to the control flow.\n  Currently those tests fallback to Python implementation'
    return val
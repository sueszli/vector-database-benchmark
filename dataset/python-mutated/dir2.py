"""A fancy version of Python's builtin :func:`dir` function.
"""
import inspect
import types

def safe_hasattr(obj, attr):
    if False:
        while True:
            i = 10
    'In recent versions of Python, hasattr() only catches AttributeError.\n    This catches all errors.\n    '
    try:
        getattr(obj, attr)
        return True
    except:
        return False

def dir2(obj):
    if False:
        while True:
            i = 10
    'dir2(obj) -> list of strings\n\n    Extended version of the Python builtin dir(), which does a few extra\n    checks.\n\n    This version is guaranteed to return only a list of true strings, whereas\n    dir() returns anything that objects inject into themselves, even if they\n    are later not really valid for attribute access (many extension libraries\n    have such bugs).\n    '
    try:
        words = set(dir(obj))
    except Exception:
        words = set()
    if safe_hasattr(obj, '__class__'):
        words |= set(dir(obj.__class__))
    words = [w for w in words if isinstance(w, str)]
    return sorted(words)

def get_real_method(obj, name):
    if False:
        for i in range(10):
            print('nop')
    'Like getattr, but with a few extra sanity checks:\n\n    - If obj is a class, ignore everything except class methods\n    - Check if obj is a proxy that claims to have all attributes\n    - Catch attribute access failing with any exception\n    - Check that the attribute is a callable object\n\n    Returns the method or None.\n    '
    try:
        canary = getattr(obj, '_ipython_canary_method_should_not_exist_', None)
    except Exception:
        return None
    if canary is not None:
        return None
    try:
        m = getattr(obj, name, None)
    except Exception:
        return None
    if inspect.isclass(obj) and (not isinstance(m, types.MethodType)):
        return None
    if callable(m):
        return m
    return None
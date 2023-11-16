"""
Functions for working with "safe strings": strings that can be displayed safely
without further escaping in HTML. Marking something as a "safe string" means
that the producer of the string has already turned characters that should not
be interpreted by the HTML engine (e.g. '<') into the appropriate entities.
"""
from functools import wraps
from django.utils.functional import keep_lazy

class SafeData:
    __slots__ = ()

    def __html__(self):
        if False:
            while True:
                i = 10
        "\n        Return the html representation of a string for interoperability.\n\n        This allows other template engines to understand Django's SafeData.\n        "
        return self

class SafeString(str, SafeData):
    """
    A str subclass that has been specifically marked as "safe" for HTML output
    purposes.
    """
    __slots__ = ()

    def __add__(self, rhs):
        if False:
            print('Hello World!')
        '\n        Concatenating a safe string with another safe bytestring or\n        safe string is safe. Otherwise, the result is no longer safe.\n        '
        t = super().__add__(rhs)
        if isinstance(rhs, SafeData):
            return SafeString(t)
        return t

    def __str__(self):
        if False:
            print('Hello World!')
        return self
SafeText = SafeString

def _safety_decorator(safety_marker, func):
    if False:
        print('Hello World!')

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        return safety_marker(func(*args, **kwargs))
    return wrapper

@keep_lazy(SafeString)
def mark_safe(s):
    if False:
        for i in range(10):
            print('nop')
    '\n    Explicitly mark a string as safe for (HTML) output purposes. The returned\n    object can be used everywhere a string is appropriate.\n\n    If used on a method as a decorator, mark the returned data as safe.\n\n    Can be called multiple times on a single string.\n    '
    if hasattr(s, '__html__'):
        return s
    if callable(s):
        return _safety_decorator(mark_safe, s)
    return SafeString(s)
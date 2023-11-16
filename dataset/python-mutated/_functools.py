import types
import functools

def method_cache(method, cache_wrapper=None):
    if False:
        print('Hello World!')
    "\n    Wrap lru_cache to support storing the cache data in the object instances.\n\n    Abstracts the common paradigm where the method explicitly saves an\n    underscore-prefixed protected property on first call and returns that\n    subsequently.\n\n    >>> class MyClass:\n    ...     calls = 0\n    ...\n    ...     @method_cache\n    ...     def method(self, value):\n    ...         self.calls += 1\n    ...         return value\n\n    >>> a = MyClass()\n    >>> a.method(3)\n    3\n    >>> for x in range(75):\n    ...     res = a.method(x)\n    >>> a.calls\n    75\n\n    Note that the apparent behavior will be exactly like that of lru_cache\n    except that the cache is stored on each instance, so values in one\n    instance will not flush values from another, and when an instance is\n    deleted, so are the cached values for that instance.\n\n    >>> b = MyClass()\n    >>> for x in range(35):\n    ...     res = b.method(x)\n    >>> b.calls\n    35\n    >>> a.method(0)\n    0\n    >>> a.calls\n    75\n\n    Note that if method had been decorated with ``functools.lru_cache()``,\n    a.calls would have been 76 (due to the cached value of 0 having been\n    flushed by the 'b' instance).\n\n    Clear the cache with ``.cache_clear()``\n\n    >>> a.method.cache_clear()\n\n    Same for a method that hasn't yet been called.\n\n    >>> c = MyClass()\n    >>> c.method.cache_clear()\n\n    Another cache wrapper may be supplied:\n\n    >>> cache = functools.lru_cache(maxsize=2)\n    >>> MyClass.method2 = method_cache(lambda self: 3, cache_wrapper=cache)\n    >>> a = MyClass()\n    >>> a.method2()\n    3\n\n    Caution - do not subsequently wrap the method with another decorator, such\n    as ``@property``, which changes the semantics of the function.\n\n    See also\n    http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/\n    for another implementation and additional justification.\n    "
    cache_wrapper = cache_wrapper or functools.lru_cache()

    def wrapper(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        bound_method = types.MethodType(method, self)
        cached_method = cache_wrapper(bound_method)
        setattr(self, method.__name__, cached_method)
        return cached_method(*args, **kwargs)
    wrapper.cache_clear = lambda : None
    return wrapper

def pass_none(func):
    if False:
        return 10
    "\n    Wrap func so it's not called if its first param is None\n\n    >>> print_text = pass_none(print)\n    >>> print_text('text')\n    text\n    >>> print_text(None)\n    "

    @functools.wraps(func)
    def wrapper(param, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if param is not None:
            return func(param, *args, **kwargs)
    return wrapper
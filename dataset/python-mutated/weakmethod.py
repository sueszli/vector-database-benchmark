"""
Weak Method
===========

The :class:`WeakMethod` is used by the :class:`~kivy.clock.Clock` class to
allow references to a bound method that permits the associated object to
be garbage collected. Please refer to
`examples/core/clock_method.py` for more information.

This WeakMethod class is taken from the recipe
http://code.activestate.com/recipes/81253/, based on the nicodemus version.
Many thanks nicodemus!
"""
import weakref

class WeakMethod:
    """Implementation of a
    `weakref <http://en.wikipedia.org/wiki/Weak_reference>`_
    for functions and bound methods.
    """

    def __init__(self, method):
        if False:
            print('Hello World!')
        self.method = None
        self.method_name = None
        try:
            if method.__self__ is not None:
                self.method_name = method.__func__.__name__
                self.proxy = weakref.proxy(method.__self__)
            else:
                self.method = method
                self.proxy = None
        except AttributeError:
            self.method = method
            self.proxy = None

    def __call__(self):
        if False:
            return 10
        "Return a new bound-method like the original, or the\n        original function if it was just a function or unbound\n        method.\n        Returns None if the original object doesn't exist.\n        "
        if self.proxy is not None:
            try:
                return getattr(self.proxy, self.method_name)
            except ReferenceError:
                return None
        return self.method

    def is_dead(self):
        if False:
            while True:
                i = 10
        'Returns True if the referenced callable was a bound method and\n        the instance no longer exists. Otherwise, return False.\n        '
        if self.proxy is None:
            return False
        try:
            getattr(self.proxy, self.method_name)
            return False
        except ReferenceError:
            return True

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if type(self) is not type(other):
            return False
        s = self()
        return s is not None and s == other()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<WeakMethod proxy={} method={} method_name={}>'.format(self.proxy, self.method, self.method_name)
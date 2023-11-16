"""
An object subclass for Python 2 that gives new-style classes written in the
style of Python 3 (with ``__next__`` and unicode-returning ``__str__`` methods)
the appropriate Python 2-style ``next`` and ``__unicode__`` methods for compatible.

Example use::

    from builtins import object

    my_unicode_str = u'Unicode string: 孔子'

    class A(object):
        def __str__(self):
            return my_unicode_str

    a = A()
    print(str(a))

    # On Python 2, these relations hold:
    assert unicode(a) == my_unicode_string
    assert str(a) == my_unicode_string.encode('utf-8')


Another example::

    from builtins import object

    class Upper(object):
        def __init__(self, iterable):
            self._iter = iter(iterable)
        def __next__(self):                 # note the Py3 interface
            return next(self._iter).upper()
        def __iter__(self):
            return self

    assert list(Upper('hello')) == list('HELLO')

"""

class newobject(object):
    """
    A magical object class that provides Python 2 compatibility methods::
        next
        __unicode__
        __nonzero__

    Subclasses of this class can merely define the Python 3 methods (__next__,
    __str__, and __bool__).
    """

    def next(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, '__next__'):
            return type(self).__next__(self)
        raise TypeError('newobject is not an iterator')

    def __unicode__(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '__str__'):
            s = type(self).__str__(self)
        else:
            s = str(self)
        if isinstance(s, unicode):
            return s
        else:
            return s.decode('utf-8')

    def __nonzero__(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '__bool__'):
            return type(self).__bool__(self)
        if hasattr(self, '__len__'):
            return type(self).__len__(self)
        return True

    def __long__(self):
        if False:
            print('Hello World!')
        if not hasattr(self, '__int__'):
            return NotImplemented
        return self.__int__()

    def __native__(self):
        if False:
            return 10
        '\n        Hook for the future.utils.native() function\n        '
        return object(self)
    __slots__ = []
__all__ = ['newobject']
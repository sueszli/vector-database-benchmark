import enum
from io import StringIO
__all__ = ['Class']
integer = 1

def raises(exc, func, *args, **kwds):
    if False:
        i = 10
        return i + 15
    'Raise AssertionError if ``func(*args, **kwds)`` does not raise *exc*.'
    pass

class CustomEx(Exception):
    """My custom exception."""

    def f(self):
        if False:
            for i in range(10):
                print('nop')
        'Exception method.'

class CustomDataDescriptor:
    """Descriptor class docstring."""

    def __init__(self, doc):
        if False:
            while True:
                i = 10
        self.__doc__ = doc

    def __get__(self, obj, type=None):
        if False:
            print('Hello World!')
        if obj is None:
            return self
        return 42

    def meth(self):
        if False:
            i = 10
            return i + 15
        'Function.'
        return 'The Answer'

class CustomDataDescriptorMeta(type):
    """Descriptor metaclass docstring."""

class CustomDataDescriptor2(CustomDataDescriptor):
    """Descriptor class with custom metaclass docstring."""
    __metaclass__ = CustomDataDescriptorMeta

def _funky_classmethod(name, b, c, d, docstring=None):
    if False:
        return 10
    'Generates a classmethod for a class from a template by filling out\n    some arguments.'

    def template(cls, a, b, c, d=4, e=5, f=6):
        if False:
            for i in range(10):
                print('nop')
        return (a, b, c, d, e, f)
    from functools import partial
    function = partial(template, b=b, c=c, d=d)
    function.__name__ = name
    function.__doc__ = docstring
    return classmethod(function)

class Base:

    def inheritedmeth(self):
        if False:
            i = 10
            return i + 15
        'Inherited function.'

class Derived(Base):

    def inheritedmeth(self):
        if False:
            return 10
        pass

class Class(Base):
    """Class to document."""
    descr = CustomDataDescriptor('Descriptor instance docstring.')

    def meth(self):
        if False:
            i = 10
            return i + 15
        'Function.'

    def undocmeth(self):
        if False:
            i = 10
            return i + 15
        pass

    def skipmeth(self):
        if False:
            while True:
                i = 10
        'Method that should be skipped.'

    def excludemeth(self):
        if False:
            return 10
        'Method that should be excluded.'
    skipattr = 'foo'
    attr = 'bar'

    @property
    def prop(self):
        if False:
            for i in range(10):
                print('nop')
        'Property.'
    docattr = 'baz'
    'should likewise be documented -- süß'
    udocattr = 'quux'
    'should be documented as well - süß'
    mdocattr = StringIO()
    'should be documented as well - süß'
    roger = _funky_classmethod('roger', 2, 3, 4)
    moore = _funky_classmethod('moore', 9, 8, 7, docstring='moore(a, e, f) -> happiness')

    def __init__(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.inst_attr_inline = None
        self.inst_attr_comment = None
        self.inst_attr_string = None
        'a documented instance attribute'
        self._private_inst_attr = None

    def __special1__(self):
        if False:
            while True:
                i = 10
        'documented special method'

    def __special2__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class CustomDict(dict):
    """Docstring."""

def function(foo, *args, **kwds):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return spam.\n    '
    pass

class Outer:
    """Foo"""

    class Inner:
        """Foo"""

        def meth(self):
            if False:
                while True:
                    i = 10
            'Foo'
    factory = dict

class DocstringSig:

    def meth(self):
        if False:
            for i in range(10):
                print('nop')
        'meth(FOO, BAR=1) -> BAZ\nFirst line of docstring\n\n        rest of docstring\n        '

    def meth2(self):
        if False:
            i = 10
            return i + 15
        'First line, no signature\n        Second line followed by indentation::\n\n            indented line\n        '

    @property
    def prop1(self):
        if False:
            for i in range(10):
                print('nop')
        'DocstringSig.prop1(self)\n        First line of docstring\n        '
        return 123

    @property
    def prop2(self):
        if False:
            print('Hello World!')
        'First line of docstring\n        Second line of docstring\n        '
        return 456

class StrRepr(str):

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

class AttCls:
    a1 = StrRepr('hello\nworld')
    a2 = None

class InstAttCls:
    """Class with documented class and instance attributes."""
    ca1 = 'a'
    ca2 = 'b'
    ca3 = 'c'
    'Docstring for class attribute InstAttCls.ca3.'

    def __init__(self):
        if False:
            print('Hello World!')
        self.ia1 = 'd'
        self.ia2 = 'e'
        'Docstring for instance attribute InstAttCls.ia2.'

class EnumCls(enum.Enum):
    """
    this is enum class
    """
    val1 = 12
    val2 = 23
    val3 = 34
    'doc for val3'
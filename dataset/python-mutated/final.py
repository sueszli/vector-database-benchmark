from abc import ABCMeta, abstractmethod
from six import with_metaclass, iteritems
_type_error = TypeError('Cannot override final attribute')

def bases_mro(bases):
    if False:
        while True:
            i = 10
    '\n    Yield classes in the order that methods should be looked up from the\n    base classes of an object.\n    '
    for base in bases:
        for class_ in base.__mro__:
            yield class_

def is_final(name, mro):
    if False:
        print('Hello World!')
    '\n    Checks if `name` is a `final` object in the given `mro`.\n    We need to check the mro because we need to directly go into the __dict__\n    of the classes. Because `final` objects are descriptor, we need to grab\n    them _BEFORE_ the `__call__` is invoked.\n    '
    return any((isinstance(getattr(c, '__dict__', {}).get(name), final) for c in bases_mro(mro)))

class FinalMeta(type):
    """A metaclass template for classes the want to prevent subclassess from
    overriding a some methods or attributes.
    """

    def __new__(mcls, name, bases, dict_):
        if False:
            while True:
                i = 10
        for (k, v) in iteritems(dict_):
            if is_final(k, bases):
                raise _type_error
        setattr_ = dict_.get('__setattr__')
        if setattr_ is None:
            setattr_ = bases[0].__setattr__
        if not is_final('__setattr__', bases) and (not isinstance(setattr_, final)):
            dict_['__setattr__'] = final(setattr_)
        return super(FinalMeta, mcls).__new__(mcls, name, bases, dict_)

    def __setattr__(self, name, value):
        if False:
            i = 10
            return i + 15
        'This stops the `final` attributes from being reassigned on the\n        class object.\n        '
        if is_final(name, self.__mro__):
            raise _type_error
        super(FinalMeta, self).__setattr__(name, value)

class final(with_metaclass(ABCMeta)):
    """
    An attribute that cannot be overridden.
    This is like the final modifier in Java.

    Example usage:
    >>> from six import with_metaclass
    >>> class C(with_metaclass(FinalMeta, object)):
    ...    @final
    ...    def f(self):
    ...        return 'value'
    ...

    This constructs a class with final method `f`. This cannot be overridden
    on the class object or on any instance. You cannot override this by
    subclassing `C`; attempting to do so will raise a `TypeError` at class
    construction time.
    """

    def __new__(cls, attr):
        if False:
            while True:
                i = 10
        if hasattr(attr, '__get__'):
            return object.__new__(finaldescriptor)
        else:
            return object.__new__(finalvalue)

    def __init__(self, attr):
        if False:
            print('Hello World!')
        self._attr = attr

    def __set__(self, instance, value):
        if False:
            print('Hello World!')
        '\n        `final` objects cannot be reassigned. This is the most import concept\n        about `final`s.\n\n        Unlike a `property` object, this will raise a `TypeError` when you\n        attempt to reassign it.\n        '
        raise _type_error

    @abstractmethod
    def __get__(self, instance, owner):
        if False:
            print('Hello World!')
        raise NotImplementedError('__get__')

class finalvalue(final):
    """
    A wrapper for a non-descriptor attribute.
    """

    def __get__(self, instance, owner):
        if False:
            while True:
                i = 10
        return self._attr

class finaldescriptor(final):
    """
    A final wrapper around a descriptor.
    """

    def __get__(self, instance, owner):
        if False:
            print('Hello World!')
        return self._attr.__get__(instance, owner)
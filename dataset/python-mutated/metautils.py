from operator import attrgetter
import six

def compose_types(a, *cs):
    if False:
        print('Hello World!')
    "Compose multiple classes together.\n\n    Parameters\n    ----------\n    *mcls : tuple[type]\n        The classes that you would like to compose\n\n    Returns\n    -------\n    cls : type\n        A type that subclasses all of the types in ``mcls``.\n\n    Notes\n    -----\n    A common use case for this is to build composed metaclasses, for example,\n    imagine you have some simple metaclass ``M`` and some instance of ``M``\n    named ``C`` like so:\n\n    .. code-block:: python\n\n       >>> class M(type):\n       ...     def __new__(mcls, name, bases, dict_):\n       ...         dict_['ayy'] = 'lmao'\n       ...         return super(M, mcls).__new__(mcls, name, bases, dict_)\n\n\n       >>> from six import with_metaclass\n       >>> class C(with_metaclass(M, object)):\n       ...     pass\n\n\n    We now want to create a sublclass of ``C`` that is also an abstract class.\n    We can use ``compose_types`` to create a new metaclass that is a subclass\n    of ``M`` and ``ABCMeta``. This is needed because a subclass of a class\n    with a metaclass must have a metaclass which is a subclass of the metaclass\n    of the superclass.\n\n\n    .. code-block:: python\n\n       >>> from abc import ABCMeta, abstractmethod\n       >>> class D(with_metaclass(compose_types(M, ABCMeta), C)):\n       ...     @abstractmethod\n       ...     def f(self):\n       ...         raise NotImplementedError('f')\n\n\n    We can see that this class has both metaclasses applied to it:\n\n    .. code-block:: python\n\n       >>> D.ayy\n       'lmao'\n       >>> D()\n       Traceback (most recent call last):\n          ...\n       TypeError: Can't instantiate abstract class D with abstract methods f\n\n\n    An important note here is that ``M`` did not use ``type.__new__`` and\n    instead used ``super()``. This is to support cooperative multiple\n    inheritance which is needed for ``compose_types`` to work as intended.\n    After we have composed these types ``M.__new__``'s super will actually\n    go to ``ABCMeta.__new__`` and not ``type.__new__``.\n\n    Always using ``super()`` to dispatch to your superclass is best practices\n    anyways so most classes should compose without much special considerations.\n    "
    if not cs:
        return a
    mcls = (a,) + cs
    return type('compose_types(%s)' % ', '.join(map(attrgetter('__name__'), mcls)), mcls, {})

def with_metaclasses(metaclasses, *bases):
    if False:
        for i in range(10):
            print('nop')
    'Make a class inheriting from ``bases`` whose metaclass inherits from\n    all of ``metaclasses``.\n\n    Like :func:`six.with_metaclass`, but allows multiple metaclasses.\n\n    Parameters\n    ----------\n    metaclasses : iterable[type]\n        A tuple of types to use as metaclasses.\n    *bases : tuple[type]\n        A tuple of types to use as bases.\n\n    Returns\n    -------\n    base : type\n        A subtype of ``bases`` whose metaclass is a subtype of ``metaclasses``.\n\n    Notes\n    -----\n    The metaclasses must be written to support cooperative multiple\n    inheritance. This means that they must delegate all calls to ``super()``\n    instead of inlining their super class by name.\n    '
    return six.with_metaclass(compose_types(*metaclasses), *bases)
from .core import unify, reify
from .dispatch import dispatch

def unifiable(cls):
    if False:
        while True:
            i = 10
    " Register standard unify and reify operations on class\n    This uses the type and __dict__ or __slots__ attributes to define the\n    nature of the term\n    See Also:\n    >>> # xdoctest: +SKIP\n    >>> class A(object):\n    ...     def __init__(self, a, b):\n    ...         self.a = a\n    ...         self.b = b\n    >>> unifiable(A)\n    <class 'unification.more.A'>\n    >>> x = var('x')\n    >>> a = A(1, 2)\n    >>> b = A(1, x)\n    >>> unify(a, b, {})\n    {~x: 2}\n    "
    _unify.add((cls, cls, dict), unify_object)
    _reify.add((cls, dict), reify_object)
    return cls

def reify_object(o, s):
    if False:
        return 10
    ' Reify a Python object with a substitution\n    >>> # xdoctest: +SKIP\n    >>> class Foo(object):\n    ...     def __init__(self, a, b):\n    ...         self.a = a\n    ...         self.b = b\n    ...     def __str__(self):\n    ...         return "Foo(%s, %s)"%(str(self.a), str(self.b))\n    >>> x = var(\'x\')\n    >>> f = Foo(1, x)\n    >>> print(f)\n    Foo(1, ~x)\n    >>> print(reify_object(f, {x: 2}))\n    Foo(1, 2)\n    '
    if hasattr(o, '__slots__'):
        return _reify_object_slots(o, s)
    else:
        return _reify_object_dict(o, s)

def _reify_object_dict(o, s):
    if False:
        print('Hello World!')
    obj = object.__new__(type(o))
    d = reify(o.__dict__, s)
    if d == o.__dict__:
        return o
    obj.__dict__.update(d)
    return obj

def _reify_object_slots(o, s):
    if False:
        i = 10
        return i + 15
    attrs = [getattr(o, attr) for attr in o.__slots__]
    new_attrs = reify(attrs, s)
    if attrs == new_attrs:
        return o
    else:
        newobj = object.__new__(type(o))
        for (slot, attr) in zip(o.__slots__, new_attrs):
            setattr(newobj, slot, attr)
        return newobj

@dispatch(slice, dict)
def _reify(o, s):
    if False:
        while True:
            i = 10
    ' Reify a Python ``slice`` object '
    return slice(*reify((o.start, o.stop, o.step), s))

def unify_object(u, v, s):
    if False:
        while True:
            i = 10
    ' Unify two Python objects\n    Unifies their type and ``__dict__`` attributes\n    >>> # xdoctest: +SKIP\n    >>> class Foo(object):\n    ...     def __init__(self, a, b):\n    ...         self.a = a\n    ...         self.b = b\n    ...     def __str__(self):\n    ...         return "Foo(%s, %s)"%(str(self.a), str(self.b))\n    >>> x = var(\'x\')\n    >>> f = Foo(1, x)\n    >>> g = Foo(1, 2)\n    >>> unify_object(f, g, {})\n    {~x: 2}\n    '
    if type(u) != type(v):
        return False
    if hasattr(u, '__slots__'):
        return unify([getattr(u, slot) for slot in u.__slots__], [getattr(v, slot) for slot in v.__slots__], s)
    else:
        return unify(u.__dict__, v.__dict__, s)

@dispatch(slice, slice, dict)
def _unify(u, v, s):
    if False:
        for i in range(10):
            print('nop')
    ' Unify a Python ``slice`` object '
    return unify((u.start, u.stop, u.step), (v.start, v.stop, v.step), s)
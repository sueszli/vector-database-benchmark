import sys

def immutable(members='', name='Immutable', verbose=False):
    if False:
        print('Hello World!')
    "\n    Produces a class that either can be used standalone or as a base class for persistent classes.\n\n    This is a thin wrapper around a named tuple.\n\n    Constructing a type and using it to instantiate objects:\n\n    >>> Point = immutable('x, y', name='Point')\n    >>> p = Point(1, 2)\n    >>> p2 = p.set(x=3)\n    >>> p\n    Point(x=1, y=2)\n    >>> p2\n    Point(x=3, y=2)\n\n    Inheriting from a constructed type. In this case no type name needs to be supplied:\n\n    >>> class PositivePoint(immutable('x, y')):\n    ...     __slots__ = tuple()\n    ...     def __new__(cls, x, y):\n    ...         if x > 0 and y > 0:\n    ...             return super(PositivePoint, cls).__new__(cls, x, y)\n    ...         raise Exception('Coordinates must be positive!')\n    ...\n    >>> p = PositivePoint(1, 2)\n    >>> p.set(x=3)\n    PositivePoint(x=3, y=2)\n    >>> p.set(y=-3)\n    Traceback (most recent call last):\n    Exception: Coordinates must be positive!\n\n    The persistent class also supports the notion of frozen members. The value of a frozen member\n    cannot be updated. For example it could be used to implement an ID that should remain the same\n    over time. A frozen member is denoted by a trailing underscore.\n\n    >>> Point = immutable('x, y, id_', name='Point')\n    >>> p = Point(1, 2, id_=17)\n    >>> p.set(x=3)\n    Point(x=3, y=2, id_=17)\n    >>> p.set(id_=18)\n    Traceback (most recent call last):\n    AttributeError: Cannot set frozen members id_\n    "
    if isinstance(members, str):
        members = members.replace(',', ' ').split()

    def frozen_member_test():
        if False:
            for i in range(10):
                print('nop')
        frozen_members = ["'%s'" % f for f in members if f.endswith('_')]
        if frozen_members:
            return "\n        frozen_fields = fields_to_modify & set([{frozen_members}])\n        if frozen_fields:\n            raise AttributeError('Cannot set frozen members %s' % ', '.join(frozen_fields))\n            ".format(frozen_members=', '.join(frozen_members))
        return ''
    verbose_string = ''
    if sys.version_info < (3, 7):
        verbose_string = ', verbose={verbose}'.format(verbose=verbose)
    quoted_members = ', '.join(("'%s'" % m for m in members))
    template = '\nclass {class_name}(namedtuple(\'ImmutableBase\', [{quoted_members}]{verbose_string})):\n    __slots__ = tuple()\n\n    def __repr__(self):\n        return super({class_name}, self).__repr__().replace(\'ImmutableBase\', self.__class__.__name__)\n\n    def set(self, **kwargs):\n        if not kwargs:\n            return self\n\n        fields_to_modify = set(kwargs.keys())\n        if not fields_to_modify <= {member_set}:\n            raise AttributeError("\'%s\' is not a member" % \', \'.join(fields_to_modify - {member_set}))\n\n        {frozen_member_test}\n\n        return self.__class__.__new__(self.__class__, *map(kwargs.pop, [{quoted_members}], self))\n'.format(quoted_members=quoted_members, member_set='set([%s])' % quoted_members if quoted_members else 'set()', frozen_member_test=frozen_member_test(), verbose_string=verbose_string, class_name=name)
    if verbose:
        print(template)
    from collections import namedtuple
    namespace = dict(namedtuple=namedtuple, __name__='pyrsistent_immutable')
    try:
        exec(template, namespace)
    except SyntaxError as e:
        raise SyntaxError(str(e) + ':\n' + template) from e
    return namespace[name]
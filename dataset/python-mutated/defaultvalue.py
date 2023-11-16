"""This module contains the DefaultValue class.

.. versionchanged:: 20.0
   Previously, the contents of this module were available through the (no longer existing)
   module ``telegram._utils.helpers``.

Warning:
    Contents of this module are intended to be used internally by the library and *not* by the
    user. Changes to this module are not considered breaking changes and may not be documented in
    the changelog.
"""
from typing import Generic, TypeVar, Union, overload
DVType = TypeVar('DVType', bound=object)
OT = TypeVar('OT', bound=object)

class DefaultValue(Generic[DVType]):
    """Wrapper for immutable default arguments that allows to check, if the default value was set
    explicitly. Usage::

        default_one = DefaultValue(1)
        def f(arg=default_one):
            if arg is default_one:
                print('`arg` is the default')
                arg = arg.value
            else:
                print('`arg` was set explicitly')
            print(f'`arg` = {str(arg)}')

    This yields::

        >>> f()
        `arg` is the default
        `arg` = 1
        >>> f(1)
        `arg` was set explicitly
        `arg` = 1
        >>> f(2)
        `arg` was set explicitly
        `arg` = 2

    Also allows to evaluate truthiness::

        default = DefaultValue(value)
        if default:
            ...

    is equivalent to::

        default = DefaultValue(value)
        if value:
            ...

    ``repr(DefaultValue(value))`` returns ``repr(value)`` and ``str(DefaultValue(value))`` returns
    ``f'DefaultValue({value})'``.

    Args:
        value (:class:`object`): The value of the default argument
    Attributes:
        value (:class:`object`): The value of the default argument

    """
    __slots__ = ('value',)

    def __init__(self, value: DVType):
        if False:
            for i in range(10):
                print('nop')
        self.value: DVType = value

    def __bool__(self) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self.value)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'DefaultValue({self.value})'

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return repr(self.value)

    @overload
    @staticmethod
    def get_value(obj: 'DefaultValue[OT]') -> OT:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    @staticmethod
    def get_value(obj: OT) -> OT:
        if False:
            i = 10
            return i + 15
        ...

    @staticmethod
    def get_value(obj: Union[OT, 'DefaultValue[OT]']) -> OT:
        if False:
            print('Hello World!')
        'Shortcut for::\n\n            return obj.value if isinstance(obj, DefaultValue) else obj\n\n        Args:\n            obj (:obj:`object`): The object to process\n\n        Returns:\n            Same type as input, or the value of the input: The value\n        '
        return obj.value if isinstance(obj, DefaultValue) else obj
DEFAULT_NONE: DefaultValue[None] = DefaultValue(None)
':class:`DefaultValue`: Default :obj:`None`'
DEFAULT_FALSE: DefaultValue[bool] = DefaultValue(False)
':class:`DefaultValue`: Default :obj:`False`'
DEFAULT_TRUE: DefaultValue[bool] = DefaultValue(True)
':class:`DefaultValue`: Default :obj:`True`\n\n.. versionadded:: 20.0\n'
DEFAULT_20: DefaultValue[int] = DefaultValue(20)
':class:`DefaultValue`: Default :obj:`20`'
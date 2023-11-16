""" Provide ``Nullable`` and ``NonNullable`` properties. """
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any, TypeVar, Union
from ...util.deprecation import deprecated
from ._sphinx import property_link, register_type_link, type_link
from .bases import Init, Property, SingleParameterizedProperty, TypeOrInst
from .required import Required
from .singletons import Undefined
__all__ = ('NonNullable', 'Nullable')
T = TypeVar('T')

class Nullable(SingleParameterizedProperty[Union[T, None]]):
    """ A property accepting ``None`` or a value of some other type. """

    def __init__(self, type_param: TypeOrInst[Property[T]], *, default: Init[T | None]=None, help: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(type_param, default=default, help=help)

    def transform(self, value: Any) -> T | None:
        if False:
            for i in range(10):
                print('nop')
        return None if value is None else super().transform(value)

    def wrap(self, value: Any) -> Any:
        if False:
            i = 10
            return i + 15
        return None if value is None else super().wrap(value)

    def validate(self, value: Any, detail: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            return
        try:
            super().validate(value, detail=False)
        except ValueError:
            pass
        else:
            return
        msg = '' if not detail else f'expected either None or a value of type {self.type_param}, got {value!r}'
        raise ValueError(msg)

class NonNullable(Required[T]):
    """
    A property accepting a value of some other type while having undefined default.

    .. deprecated:: 3.0.0

        Use ``bokeh.core.property.required.Required`` instead.
    """

    def __init__(self, type_param: TypeOrInst[Property[T]], *, default: Init[T]=Undefined, help: str | None=None) -> None:
        if False:
            print('Hello World!')
        deprecated((3, 0, 0), 'NonNullable(Type)', 'Required(Type)')
        super().__init__(type_param, default=default, help=help)

@register_type_link(Nullable)
@register_type_link(NonNullable)
def _sphinx_type_link(obj: SingleParameterizedProperty[Any]) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'{property_link(obj)}({type_link(obj.type_param)})'
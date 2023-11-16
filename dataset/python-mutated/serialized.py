""" Provide ``NotSerialized`` property. """
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any, TypeVar
from ._sphinx import property_link, register_type_link, type_link
from .bases import Init, Property, SingleParameterizedProperty, TypeOrInst
from .singletons import Intrinsic
__all__ = ('NotSerialized',)
T = TypeVar('T')

class NotSerialized(SingleParameterizedProperty[T]):
    """
    A property which state won't be synced with the browser.
    """
    _serialized = False

    def __init__(self, type_param: TypeOrInst[Property[T]], *, default: Init[T]=Intrinsic, help: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(type_param, default=default, help=help)

@register_type_link(NotSerialized)
def _sphinx_type_link(obj: SingleParameterizedProperty[Any]) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'{property_link(obj)}({type_link(obj.type_param)})'
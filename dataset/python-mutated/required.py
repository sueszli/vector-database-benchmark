""" Provide ``Required`` property. """
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any, TypeVar
from ._sphinx import property_link, register_type_link, type_link
from .bases import Init, Property, SingleParameterizedProperty, TypeOrInst
from .singletons import Undefined
__all__ = ('Required',)
T = TypeVar('T')

class Required(SingleParameterizedProperty[T]):
    """ A property accepting a value of some other type while having undefined default. """

    def __init__(self, type_param: TypeOrInst[Property[T]], *, default: Init[T]=Undefined, help: str | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(type_param, default=default, help=help)

@register_type_link(Required)
def _sphinx_type_link(obj: SingleParameterizedProperty[Any]) -> str:
    if False:
        i = 10
        return i + 15
    return f'{property_link(obj)}({type_link(obj.type_param)})'
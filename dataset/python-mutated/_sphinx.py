""" Functions useful for generating rich sphinx links for properties

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, Callable
__all__ = ('model_link', 'property_link', 'register_type_link', 'type_link')
_type_links: dict[type[Any], Callable[[Any], str]] = {}
if TYPE_CHECKING:
    from typing_extensions import TypeAlias

def model_link(fullname: str) -> str:
    if False:
        i = 10
        return i + 15
    return f':class:`~{fullname}`\\ '

def property_link(obj: Any) -> str:
    if False:
        return 10
    return f':class:`~bokeh.core.properties.{obj.__class__.__name__}`\\ '
Fn: TypeAlias = Callable[[Any], str]

def register_type_link(cls: type[Any]) -> Callable[[Fn], Fn]:
    if False:
        for i in range(10):
            print('nop')

    def decorator(func: Fn):
        if False:
            for i in range(10):
                print('nop')
        _type_links[cls] = func
        return func
    return decorator

def type_link(obj: Any) -> str:
    if False:
        i = 10
        return i + 15
    return _type_links.get(obj.__class__, property_link)(obj)
"""Setstate and getstate functions for objects with __slots__, allowing
compatibility with default pickling protocol
"""
from __future__ import annotations
from typing import Any, Mapping

def _setstate_slots(self: Any, state: Any) -> None:
    if False:
        while True:
            i = 10
    for (slot, value) in state.items():
        setattr(self, slot, value)

def _mangle_name(name: str, prefix: str) -> str:
    if False:
        while True:
            i = 10
    if name.startswith('__'):
        prefix = '_' + prefix
    else:
        prefix = ''
    return prefix + name

def _getstate_slots(self: Any) -> Mapping[Any, Any]:
    if False:
        return 10
    prefix = self.__class__.__name__
    ret = {}
    for name in self.__slots__:
        mangled_name = _mangle_name(name, prefix)
        if hasattr(self, mangled_name):
            ret[mangled_name] = getattr(self, mangled_name)
    return ret
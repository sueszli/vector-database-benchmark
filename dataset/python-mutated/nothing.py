""" """
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any, NoReturn
from .bases import Property, Undefined
__all__ = ('Nothing',)

class Nothing(Property[NoReturn]):
    """ The bottom type of bokeh's type system. It doesn't accept any values. """

    def __init__(self, *, help: str | None=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(default=Undefined, help=help)

    def validate(self, value: Any, detail: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise ValueError('no value is allowed')
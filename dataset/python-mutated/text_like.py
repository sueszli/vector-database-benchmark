""" ``TextLike`` is a shortcut for properties that accepts strings, parsed
strings, and text-like objects, e.g.:

* :class:`~bokeh.models.text.MathText`
* :class:`~bokeh.models.text.PlainText`

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
from .bases import Init
from .either import Either
from .instance import Instance
from .singletons import Intrinsic
from .string import MathString
if TYPE_CHECKING:
    from ...models.text import BaseText
__all__ = ('TextLike',)

class TextLike(Either):
    """ Accept a string that may be interpreted into text models or the models themselves.

    """

    def __init__(self, default: Init[str | BaseText]=Intrinsic, help: str | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(MathString(), Instance('bokeh.models.text.BaseText'), default=default, help=help)
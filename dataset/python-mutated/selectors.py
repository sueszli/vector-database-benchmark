"""
Models representing selector queries for UI components.
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.has_props import abstract
from ..core.properties import Required, String
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..model import Model
__all__ = ('ByID', 'ByClass', 'ByCSS', 'ByXPath')

@abstract
class Selector(Model):
    """ Base class for selector queries. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class ByID(Selector):
    """ Represents a CSS ID selector query. """

    def __init__(self, query: Init[str]=Intrinsic, **kwargs) -> None:
        if False:
            return 10
        super().__init__(query=query, **kwargs)
    query = Required(String, help='\n    Element CSS ID without ``#`` prefix. Alternatively use ``ByCSS("#id")``.\n    ')

class ByClass(Selector):
    """ Represents a CSS class selector query. """

    def __init__(self, query: Init[str]=Intrinsic, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(query=query, **kwargs)
    query = Required(String, help='\n    CSS class name without ``.`` prefix. Alternatively use ``ByCSS(".class")``.\n    ')

class ByCSS(Selector):
    """ Represents a CSS selector query. """

    def __init__(self, query: Init[str]=Intrinsic, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(query=query, **kwargs)
    query = Required(String, help='\n    CSS selector query (see https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Selectors).\n    ')

class ByXPath(Selector):
    """ Represents an XPath selector query. """

    def __init__(self, query: Init[str]=Intrinsic, **kwargs) -> None:
        if False:
            return 10
        super().__init__(query=query, **kwargs)
    query = Required(String, help='\n    XPath selector query (see https://developer.mozilla.org/en-US/docs/Web/XPath).\n    ')
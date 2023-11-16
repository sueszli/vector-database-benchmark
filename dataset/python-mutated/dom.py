""" An abstraction over the document object model (DOM).

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any
from ..core.has_props import HasProps, abstract
from ..core.properties import Bool, Dict, Either, Instance, List, Nullable, Required, String
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..core.validation import error
from ..core.validation.errors import NOT_A_PROPERTY_OF
from ..model import Model, Qualified
from .css import Styles
from .renderers import RendererGroup
from .ui.ui_element import UIElement
__all__ = ('Div', 'HTML', 'Span', 'Table', 'TableRow', 'Text')

@abstract
class DOMNode(Model, Qualified):
    """ Base class for DOM nodes. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class Text(DOMNode):
    """ DOM text node. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    content = String('')

@abstract
class DOMElement(DOMNode):
    """ Base class for DOM elements. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    style = Nullable(Either(Instance(Styles), Dict(String, String)))
    children = List(Either(String, Instance(DOMNode), Instance(UIElement)), default=[])

class Span(DOMElement):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)

class Div(DOMElement):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)

class Table(DOMElement):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

class TableRow(DOMElement):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

@abstract
class Action(Model, Qualified):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class Template(DOMElement):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    actions = List(Instance(Action))

class ToggleGroup(Action):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    groups = List(Instance(RendererGroup))

@abstract
class Placeholder(DOMNode):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class ValueOf(Placeholder):
    """ A placeholder for the value of a model's property. """

    def __init__(self, obj: Init[HasProps]=Intrinsic, attr: Init[str]=Intrinsic, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(obj=obj, attr=attr, **kwargs)
    obj: HasProps = Required(Instance(HasProps), help='\n    The object whose property will be observed.\n    ')
    attr: str = Required(String, help='\n    The name of the property whose value will be observed.\n    ')

    @error(NOT_A_PROPERTY_OF)
    def _check_if_an_attribute_is_a_property_of_a_model(self):
        if False:
            while True:
                i = 10
        if self.obj.lookup(self.attr, raises=False):
            return None
        else:
            return f'{self.attr} is not a property of {self.obj}'

class Index(Placeholder):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)

class ValueRef(Placeholder):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    field = Required(String)

class ColorRef(ValueRef):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    hex = Bool(default=True)
    swatch = Bool(default=True)

class HTML(Model, Qualified):
    """ A parsed HTML fragment with optional references to DOM nodes and UI elements. """

    def __init__(self, *html: str | DOMNode | UIElement, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        if html and 'html' in kwargs:
            raise TypeError("'html' argument specified multiple times")
        processed_html: Init[str | list[str | DOMNode | UIElement]]
        if not html:
            processed_html = kwargs.pop('html', Intrinsic)
        else:
            processed_html = list(html)
        super().__init__(html=processed_html, **kwargs)
    html = Required(Either(String, List(Either(String, Instance(DOMNode), Instance(UIElement)))), help='\n    Either a parsed HTML string with optional references to Bokeh objects using\n    ``<ref id="..."></ref>`` syntax. Or a list of parsed HTML interleaved with\n    Bokeh\'s objects. Any DOM node or UI element (even a plot) can be referenced\n    here.\n    ')
    refs = List(Either(String, Instance(DOMNode), Instance(UIElement)), default=[], help='\n    A collection of objects referenced by ``<ref id="..."></ref>`` from `the `html`` property.\n    Objects already included by instance in ``html`` don\'t have to be repeated here.\n    ')
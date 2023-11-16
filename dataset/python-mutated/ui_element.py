"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.has_props import abstract
from ...core.properties import Bool, Dict, Either, Instance, List, Nullable, Seq, String
from ...model import Model
from ..css import Styles, StyleSheet
__all__ = ('UIElement',)

@abstract
class UIElement(Model):
    """ Base class for user interface elements.
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    visible = Bool(default=True, help='\n    Whether the component should be displayed on screen.\n    ')
    css_classes = List(String, default=[], help='\n    A list of additional CSS classes to add to the underlying DOM element.\n    ').accepts(Seq(String), lambda x: list(x))
    styles = Either(Dict(String, Nullable(String)), Instance(Styles), default={}, help='\n    Inline CSS styles applied to the underlying DOM element.\n    ')
    stylesheets = List(Either(Instance(StyleSheet), String, Dict(String, Either(Dict(String, Nullable(String)), Instance(Styles)))), help="\n    Additional style-sheets to use for the underlying DOM element.\n\n    Note that all bokeh's components use shadow DOM, thus any included style\n    sheets must reflect that, e.g. use ``:host`` CSS pseudo selector to access\n    the root DOM element.\n    ")
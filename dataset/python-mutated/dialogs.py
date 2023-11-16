"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.properties import Bool, Either, Instance, List, Nullable, Required, String
from ..dom import DOMNode
from .ui_element import UIElement
__all__ = ('Dialog',)
Button = UIElement

class Dialog(UIElement):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    title = Nullable(Either(String, Instance(DOMNode)), default=None, help='\n    ')
    content = Required(Either(String, Instance(DOMNode), Instance(UIElement)), help='\n    ')
    buttons = List(Instance(Button), default=[], help='\n    ')
    modal = Bool(default=False, help='\n    ')
    closable = Bool(default=True, help='\n    Whether to show close (x) button in the title bar.\n    ')
    draggable = Bool(default=True, help='\n    ')
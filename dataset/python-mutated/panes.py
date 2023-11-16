""" Various kinds of panes. """
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.properties import Either, Instance, List, String
from .ui_element import UIElement
__all__ = ('Pane',)

class Pane(UIElement):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    children = List(Either(String, Instance(UIElement)), default=[], help='\n    ')
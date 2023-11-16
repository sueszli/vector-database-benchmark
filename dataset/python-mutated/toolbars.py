"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ....core.properties import Instance
from .html_annotation import HTMLAnnotation
__all__ = ('ToolbarPanel',)

class ToolbarPanel(HTMLAnnotation):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    toolbar = Instance('.models.tools.Toolbar', help='\n    A toolbar to display.\n    ')
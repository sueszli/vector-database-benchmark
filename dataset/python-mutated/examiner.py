"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.has_props import HasProps
from ...core.properties import Instance, Nullable
from .ui_element import UIElement
__all__ = ('Examiner',)

class Examiner(UIElement):
    """ A diagnostic tool for examining documents, models, properties, etc. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    target = Nullable(Instance(HasProps), help='\n    The model and its references to inspect. If not specified, then all models\n    in the document the inpector model belongs to will be inspected.\n    ')
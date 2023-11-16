"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ....core.has_props import abstract
from ..annotation import Annotation
__all__ = ('HTMLAnnotation',)

@abstract
class HTMLAnnotation(Annotation):
    """ Base class for HTML-based annotations.

    .. note::
        All annotations that inherit from this base class can be attached to
        a canvas, but are not rendered to it, thus they won't appear in saved
        plots. Only ``export_png()`` function can preserve HTML annotations.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
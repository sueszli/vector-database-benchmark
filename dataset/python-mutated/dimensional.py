"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any
from ...core.has_props import abstract
from ...core.properties import Float, List, Nullable, Override, String
from ...model import Model
__all__ = ('MetricLength',)

@abstract
class Dimensional(Model):
    """
    A base class for models defining units of measurement.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    ticks = List(Float, help='\n    Preferred values to choose from in non-exact mode.\n    ')
    include = Nullable(List(String), default=None, help='\n    An optional subset of preferred units from the basis.\n    ')
    exclude = List(String, default=[], help='\n    A subset of units from the basis to avoid.\n    ')

@abstract
class Metric(Dimensional):
    """
    A base class defining metric units of measurement.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    ticks = Override(default=[1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 250, 500, 750])

class MetricLength(Metric):
    """
    Units of metric length.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    exclude = Override(default=['dm', 'hm'])
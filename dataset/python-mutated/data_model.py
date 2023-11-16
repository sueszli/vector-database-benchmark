""" Provide a base class for all objects (called Bokeh Models) that can go in
a Bokeh |Document|.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.has_props import abstract
from .model import Model
__all__ = ('DataModel',)

@abstract
class DataModel(Model):
    __data_model__ = True

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
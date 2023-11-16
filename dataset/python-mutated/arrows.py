"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.enums import CoordinateUnits
from ...core.has_props import abstract
from ...core.properties import Enum, Include, Instance, InstanceDefault, Nullable, NumberSpec, Override, field
from ...core.property_mixins import FillProps, LineProps
from ..graphics import Marking
from .annotation import DataAnnotation
__all__ = ('Arrow', 'ArrowHead', 'NormalHead', 'OpenHead', 'TeeHead', 'VeeHead')

@abstract
class ArrowHead(Marking):
    """ Base class for arrow heads.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    size = NumberSpec(default=25, help='\n    The size, in pixels, of the arrow head.\n    ')

class OpenHead(ArrowHead):
    """ Render an open-body arrow head.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    line_props = Include(LineProps, help='\n\n    The {prop} values for the arrow head outline.\n    ')

class NormalHead(ArrowHead):
    """ Render a closed-body arrow head.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    line_props = Include(LineProps, help='\n    The {prop} values for the arrow head outline.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the arrow head interior.\n    ')
    fill_color = Override(default='black')

class TeeHead(ArrowHead):
    """ Render a tee-style arrow head.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    line_props = Include(LineProps, help='\n    The {prop} values for the arrow head outline.\n    ')

class VeeHead(ArrowHead):
    """ Render a vee-style arrow head.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    line_props = Include(LineProps, help='\n    The {prop} values for the arrow head outline.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the arrow head interior.\n    ')
    fill_color = Override(default='black')

class Arrow(DataAnnotation):
    """ Render arrows as an annotation.

    See :ref:`ug_basic_annotations_arrows` for information on plotting arrows.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    x_start = NumberSpec(default=field('x_start'), help='\n    The x-coordinates to locate the start of the arrows.\n    ')
    y_start = NumberSpec(default=field('y_start'), help='\n    The y-coordinates to locate the start of the arrows.\n    ')
    start_units = Enum(CoordinateUnits, default='data', help='\n    The unit type for the start_x and start_y attributes. Interpreted as "data\n    space" units by default.\n    ')
    start = Nullable(Instance(ArrowHead), help='\n    Instance of ``ArrowHead``.\n    ')
    x_end = NumberSpec(default=field('x_end'), help='\n    The x-coordinates to locate the end of the arrows.\n    ')
    y_end = NumberSpec(default=field('y_end'), help='\n    The y-coordinates to locate the end of the arrows.\n    ')
    end_units = Enum(CoordinateUnits, default='data', help='\n    The unit type for the end_x and end_y attributes. Interpreted as "data\n    space" units by default.\n    ')
    end = Nullable(Instance(ArrowHead), default=InstanceDefault(OpenHead), help='\n    Instance of ``ArrowHead``.\n    ')
    body_props = Include(LineProps, help='\n    The {prop} values for the arrow body.\n    ')
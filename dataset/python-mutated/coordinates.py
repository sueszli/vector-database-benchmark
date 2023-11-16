from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.properties import Either, Enum, Instance, InstanceDefault, Int, Required, String
from ..core.property.singletons import Optional, Undefined
from ..model import Model
from .ranges import DataRange1d, Range
from .scales import LinearScale, Scale
__all__ = ('CoordinateMapping', 'Node')

class CoordinateMapping(Model):
    """ A mapping between two coordinate systems. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    x_source = Instance(Range, default=InstanceDefault(DataRange1d), help='\n    The source range of the horizontal dimension of the new coordinate space.\n    ')
    y_source = Instance(Range, default=InstanceDefault(DataRange1d), help='\n    The source range of the vertical dimension of the new coordinate space.\n    ')
    x_scale = Instance(Scale, default=InstanceDefault(LinearScale), help='\n    What kind of scale to use to convert x-coordinates from the source (data)\n    space into x-coordinates in the target (possibly screen) coordinate space.\n    ')
    y_scale = Instance(Scale, default=InstanceDefault(LinearScale), help='\n    What kind of scale to use to convert y-coordinates from the source (data)\n    space into y-coordinates in the target (possibly screen) coordinate space.\n    ')
    x_target = Instance(Range, help='\n    The horizontal range to map x-coordinates in the target coordinate space.\n    ')
    y_target = Instance(Range, help='\n    The vertical range to map y-coordinates in the target coordinate space.\n    ')

class Node(Model):
    """
    Represents a symbolic coordinate (by name).

    .. note::
        This model is experimental and may change at any point.
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    target = Required(Either(Instance(Model), Enum('canvas', 'plot', 'frame', 'parent')), help="\n    The provider of coordinates for this node.\n\n    This can be either a concrete model that can provide its coordinates (e.g.\n    a renderer, a frame or a canvas) or an implicit target defined by the\n    enum, which is resolved as the nearest parent of the given type. If the\n    provider cannot be determined or it isn't able to provide coordinates,\n    then the node resolved to an invalid coordinate (with x and y components\n    being ``NaN``).\n    ")
    symbol = Required(String, help='\n    A symbolic name of a coordinate to provide.\n\n    The allowed terms are dependent on the target of this node. For example,\n    for box-like targets this will comprise of box anchors (e.g. center, top\n    left) and box edges (e.g. top, left).\n    ')
    offset = Int(default=0, help='\n    Optional pixel offset for the computed coordinate.\n    ')

def FrameLeft(*, offset: Optional[int]=Undefined) -> Node:
    if False:
        return 10
    return Node(target='frame', symbol='left', offset=offset)

def FrameRight(*, offset: Optional[int]=Undefined) -> Node:
    if False:
        return 10
    return Node(target='frame', symbol='right', offset=offset)

def FrameTop(*, offset: Optional[int]=Undefined) -> Node:
    if False:
        print('Hello World!')
    return Node(target='frame', symbol='top', offset=offset)

def FrameBottom(*, offset: Optional[int]=Undefined) -> Node:
    if False:
        for i in range(10):
            print('nop')
    return Node(target='frame', symbol='bottom', offset=offset)
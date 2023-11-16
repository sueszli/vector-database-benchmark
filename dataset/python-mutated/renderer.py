""" Base classes for the various kinds of renderer types.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.enums import RenderLevel
from ...core.has_props import abstract
from ...core.properties import Bool, Enum, Instance, Nullable, Override, String
from ...model import Model
from ..coordinates import CoordinateMapping
__all__ = ('DataRenderer', 'GuideRenderer', 'Renderer', 'RendererGroup')

class RendererGroup(Model):
    """A collection of renderers.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    visible = Bool(default=True, help='\n    Makes all groupped renderers visible or not.\n    ')

@abstract
class Renderer(Model):
    """An abstract base class for renderer types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    level = Enum(RenderLevel, help='\n    Specifies the level in which to paint this renderer.\n    ')
    visible = Bool(default=True, help='\n    Is the renderer visible.\n    ')
    coordinates = Nullable(Instance(CoordinateMapping))
    x_range_name = String('default', help='\n    A particular (named) x-range to use for computing screen locations when\n    rendering glyphs on the plot. If unset, use the default x-range.\n    ')
    y_range_name = String('default', help='\n    A particular (named) y-range to use for computing screen locations when\n    rendering glyphs on the plot. If unset, use the default y-range.\n    ')
    group = Nullable(Instance(RendererGroup))
    propagate_hover = Bool(default=False, help='\n    Allows to propagate hover events to the parent renderer, frame or canvas.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')

@abstract
class DataRenderer(Renderer):
    """ An abstract base class for data renderer types (e.g. ``GlyphRenderer``, ``GraphRenderer``).

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    level = Override(default='glyph')

@abstract
class GuideRenderer(Renderer):
    """ A base class for all guide renderer types. ``GuideRenderer`` is
    not generally useful to instantiate on its own.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    level = Override(default='guide')
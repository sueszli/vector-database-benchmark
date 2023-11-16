""" Display a variety of visual shapes whose attributes can be associated
with data columns from ``ColumnDataSources``.

All these glyphs share a minimal common interface through their base class
``Glyph``:

.. autoclass:: Glyph
    :members:

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.has_props import abstract
from ..core.properties import Instance, List
from ..model import Model
from .graphics import Decoration
__all__ = ('ConnectedXYGlyph', 'Glyph', 'XYGlyph')

@abstract
class Glyph(Model):
    """ Base class for all glyph models.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    decorations = List(Instance(Decoration), default=[], help="\n    A collection of glyph decorations, e.g. arrow heads.\n\n    Use ``GlyphRenderer.add_decoration()`` for easy setup for all glyphs\n    of a glyph renderer. Use this property when finer control is needed.\n\n    .. note::\n\n        Decorations are only for aiding visual appearance of a glyph,\n        but they don't participate in hit testing, etc.\n    ")

@abstract
class XYGlyph(Glyph):
    """ Base class of glyphs with `x` and `y` coordinate attributes.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

@abstract
class ConnectedXYGlyph(XYGlyph):
    """ Base class of glyphs with `x` and `y` coordinate attributes and
    a connected topology.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

@abstract
class LineGlyph(Glyph):
    """ Glyphs with line properties

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

@abstract
class FillGlyph(Glyph):
    """ Glyphs with fill properties

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

@abstract
class TextGlyph(Glyph):
    """ Glyphs with text properties

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)

@abstract
class HatchGlyph(Glyph):
    """ Glyphs with Hatch properties

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
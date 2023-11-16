from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.has_props import abstract
from ..core.properties import Dict, Int, List, Seq, String, Struct
from ..model import Model
__all__ = ('IntersectRenderers', 'Selection', 'SelectionPolicy', 'UnionRenderers')

class Selection(Model):
    """
    A Selection represents a portion of the data in a ``DataSource``, which
    can be visually manipulated in a plot.

    Selections are typically created by selecting points in a plot with
    a ``SelectTool``, but can also be programmatically specified.

    For most glyphs, the ``indices`` property is the relevant value to use.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    indices = Seq(Int, default=[], help='\n    The "scatter" level indices included in a selection. For example, for a\n    selection on a ``Circle`` glyph, this list records the indices of which\n    individual circles are selected.\n\n    For "multi" glyphs such as ``Patches``, ``MultiLine``, ``MultiPolygons``,\n    etc, this list records the indices of which entire sub-items are selected.\n    For example, which indidual polygons of a ``MultiPolygon`` are selected.\n    ')
    line_indices = Seq(Int, default=[], help='\n    The point indices included in a selection on a ``Line`` glyph.\n\n    This value records the indices of the individual points on a ``Line`` that\n    were selected by a selection tool.\n    ')
    multiline_indices = Dict(String, Seq(Int), default={}, help='\n    The detailed point indices included in a selection on a ``MultiLine``.\n\n    This value records which points, on which lines, are part of a seletion on\n    a ``MulitLine``. The keys are the top level indices (i.e., which line)\n    which map to lists of indices (i.e. which points on that line).\n\n    If you only need to know which lines are selected, without knowing what\n    individual points on those lines are selected, then you can look at the\n    keys of this dictionary (converted to ints).\n    ')
    image_indices = List(Struct(index=Int, i=Int, j=Int, flat_index=Int), help='\n\n    ')

@abstract
class SelectionPolicy(Model):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

class IntersectRenderers(SelectionPolicy):
    """
    When a data source is shared between multiple renderers, a row in the data
    source will only be selected if that point for each renderer is selected. The
    selection is made from the intersection of hit test results from all renderers.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class UnionRenderers(SelectionPolicy):
    """
    When a data source is shared between multiple renderers, selecting a point on
    from any renderer will cause that row in the data source to be selected. The
    selection is made from the union of hit test results from all renderers.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
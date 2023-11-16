"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.properties import Bool, Float, Instance, InstanceDefault, Override
from ..tiles import TileSource, WMTSTileSource
from .renderer import Renderer
__all__ = ('TileRenderer',)

class TileRenderer(Renderer):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    tile_source = Instance(TileSource, default=InstanceDefault(WMTSTileSource), help='\n    Local data source to use when rendering glyphs on the plot.\n    ')
    alpha = Float(1.0, help='\n    tile opacity 0.0 - 1.0\n    ')
    smoothing = Bool(default=True, help='\n    Enable image smoothing for the rendered tiles.\n    ')
    render_parents = Bool(default=True, help='\n    Flag enable/disable drawing of parent tiles while waiting for new tiles to arrive. Default value is True.\n    ')
    level = Override(default='image')
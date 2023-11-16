"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.properties import Any, Bool, Dict, Float, Int, Nullable, Override, Required, String
from ..model import Model
__all__ = ('TileSource', 'MercatorTileSource', 'TMSTileSource', 'WMTSTileSource', 'QUADKEYTileSource', 'BBoxTileSource')

class TileSource(Model):
    """ A base class for all tile source types.

    In general, tile sources are used as a required input for ``TileRenderer``.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    _args = ('url', 'tile_size', 'min_zoom', 'max_zoom', 'extra_url_vars')
    url = String('', help='\n    Tile service url e.g., http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png\n    ')
    tile_size = Int(default=256, help='\n    Tile size in pixels (e.g. 256)\n    ')
    min_zoom = Int(default=0, help='\n    A minimum zoom level for the tile layer. This is the most zoomed-out level.\n    ')
    max_zoom = Int(default=30, help='\n    A maximum zoom level for the tile layer. This is the most zoomed-in level.\n    ')
    extra_url_vars = Dict(String, Any, help='\n    A dictionary that maps url variable template keys to values.\n\n    These variables are useful for parts of tile urls which do not change from\n    tile to tile (e.g. server host name, or layer name).\n    ')
    attribution = String('', help='\n    Data provider attribution content. This can include HTML content.\n    ')
    x_origin_offset = Required(Float, help='\n    An x-offset in plot coordinates\n    ')
    y_origin_offset = Required(Float, help='\n    A y-offset in plot coordinates\n    ')
    initial_resolution = Nullable(Float, help='\n    Resolution (plot_units / pixels) of minimum zoom level of tileset\n    projection. None to auto-compute.\n    ')

class MercatorTileSource(TileSource):
    """ A base class for Mercator tile services (e.g. ``WMTSTileSource``).

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    _args = ('url', 'tile_size', 'min_zoom', 'max_zoom', 'x_origin_offset', 'y_origin_offset', 'extra_url_vars', 'initial_resolution')
    x_origin_offset = Override(default=20037508.34)
    y_origin_offset = Override(default=20037508.34)
    initial_resolution = Override(default=156543.03392804097)
    snap_to_zoom = Bool(default=False, help='\n    Forces initial extents to snap to the closest larger zoom level.')
    wrap_around = Bool(default=True, help='\n    Enables continuous horizontal panning by wrapping the x-axis based on\n    bounds of map.\n\n    .. note::\n        Axis coordinates are not wrapped. To toggle axis label visibility,\n        use ``plot.axis.visible = False``.\n\n    ')

class TMSTileSource(MercatorTileSource):
    """ Contains tile config info and provides urls for tiles based on a
    templated url e.g. ``http://your.tms.server.host/{Z}/{X}/{Y}.png``. The
    defining feature of TMS is the tile-origin in located at the bottom-left.

    ``TMSTileSource`` can also be helpful in implementing tile renderers for
    custom tile sets, including non-spatial datasets.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)

class WMTSTileSource(MercatorTileSource):
    """ Behaves much like ``TMSTileSource`` but has its tile-origin in the
    top-left.

    This is the most common used tile source for web mapping applications.
    Such companies as Google, MapQuest, Stadia, Esri, and OpenStreetMap provide
    service which use the WMTS specification e.g. ``http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png``.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class QUADKEYTileSource(MercatorTileSource):
    """ Has the same tile origin as the ``WMTSTileSource`` but requests tiles using
    a `quadkey` argument instead of X, Y, Z e.g.
    ``http://your.quadkey.tile.host/{Q}.png``

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class BBoxTileSource(MercatorTileSource):
    """ Has the same default tile origin as the ``WMTSTileSource`` but requested
    tiles use a ``{XMIN}``, ``{YMIN}``, ``{XMAX}``, ``{YMAX}`` e.g.
    ``http://your.custom.tile.service?bbox={XMIN},{YMIN},{XMAX},{YMAX}``.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    use_latlon = Bool(default=False, help='\n    Flag which indicates option to output ``{XMIN}``, ``{YMIN}``, ``{XMAX}``, ``{YMAX}`` in meters or latitude and longitude.\n    ')
""" Pre-configured tile sources for common third party tile services.

.. autofunction:: bokeh.tile_providers.get_provider

The available built-in tile providers are listed in the ``Vendors`` enum:

.. bokeh-enum:: Vendors
    :module: bokeh.tile_providers
    :noindex:

.. warning::
    The built-in Vendors are deprecated as of Bokeh 3.0.0 and will be removed in a future
    release. You can pass the same strings to ``add_tile`` directly.

Any of these values may be be passed to the ``get_provider`` function in order
to obtain a tile provider to use with a Bokeh plot. Representative samples of
each tile provider are shown below.

CARTODBPOSITRON
---------------

Tile Source for CartoDB Tile Service

.. raw:: html

    <img src="https://tiles.basemaps.cartocdn.com/light_all/14/2627/6331.png" />

CARTODBPOSITRON_RETINA
----------------------

Tile Source for CartoDB Tile Service (tiles at 'retina' resolution)

.. raw:: html

    <img src="https://tiles.basemaps.cartocdn.com/light_all/14/2627/6331@2x.png" />

ESRI_IMAGERY
------------

Tile Source for ESRI public tiles.

.. raw:: html

    <img src="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/14/6331/2627.jpg" />

OSM
---

Tile Source for Open Street Maps.

.. raw:: html

    <img src="https://c.tile.openstreetmap.org/14/2627/6331.png" />

STAMEN_TERRAIN
--------------

Tile Source for Stadia/Stamen Terrain Service

.. raw:: html

    <img src="https://tiles.stadiamaps.com/tiles/stamen_terrain/14/2627/6331.png" />

STAMEN_TERRAIN_RETINA
---------------------

Tile Source for Stadia/Stamen Terrain Service (tiles at 2x or 'retina' resolution)

.. raw:: html

    <img src="https://tiles.stadiamaps.com/tiles/stamen_terrain/14/2627/6331@2x.png" />

STAMEN_TONER
------------

Tile Source for Stadia/Stamen Toner Service

.. raw:: html

    <img src="https://tiles.stadiamaps.com/tiles/stamen_toner/14/2627/6331.png" />

STAMEN_TONER_BACKGROUND
-----------------------

Tile Source for Stadia/Stamen Toner Background Service which does not include labels

.. raw:: html

    <img src="https://tiles.stadiamaps.com/tiles/stamen_toner_background/14/2627/6331.png" />

STAMEN_TONER_LABELS
-------------------

Tile Source for Stadia/Stamen Toner Service which includes only labels

.. raw:: html

    <img src="https://tiles.stadiamaps.com/tiles/stamen_toner_labels/14/2627/6331.png" />

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import sys
import types
import xyzservices
from bokeh.core.enums import enumeration
from .util.deprecation import deprecated

class _TileProvidersModule(types.ModuleType):

    def deprecated_vendors():
        if False:
            return 10
        deprecated((3, 0, 0), 'tile_providers module', 'add_tile directly')
        return enumeration('CARTODBPOSITRON', 'CARTODBPOSITRON_RETINA', 'STAMEN_TERRAIN', 'STAMEN_TERRAIN_RETINA', 'STAMEN_TONER', 'STAMEN_TONER_BACKGROUND', 'STAMEN_TONER_LABELS', 'OSM', 'ESRI_IMAGERY', case_sensitive=True)
    Vendors = deprecated_vendors()

    def get_provider(self, provider_name: str | Vendors | xyzservices.TileProvider):
        if False:
            while True:
                i = 10
        "Use this function to retrieve an instance of a predefined tile provider.\n\n        .. warning::\n            get_provider is deprecated as of Bokeh 3.0.0 and will be removed in a future\n            release. Use ``add_tile`` directly instead.\n\n        Args:\n            provider_name (Union[str, Vendors, xyzservices.TileProvider]):\n                Name of the tile provider to supply.\n\n                Use a ``tile_providers.Vendors`` enumeration value, or the string\n                name of one of the known providers. Use\n                :class:`xyzservices.TileProvider` to pass custom tile providers.\n\n        Returns:\n            WMTSTileProviderSource: The desired tile provider instance.\n\n        Raises:\n            ValueError: if the specified provider can not be found.\n\n        Example:\n\n            .. code-block:: python\n\n                    >>> from bokeh.tile_providers import get_provider, Vendors\n                    >>> get_provider(Vendors.CARTODBPOSITRON)\n                    <class 'bokeh.models.tiles.WMTSTileSource'>\n                    >>> get_provider('CARTODBPOSITRON')\n                    <class 'bokeh.models.tiles.WMTSTileSource'>\n\n                    >>> import xyzservices.providers as xyz\n                    >>> get_provider(xyz.CartoDB.Positron)\n                    <class 'bokeh.models.tiles.WMTSTileSource'>\n        "
        deprecated((3, 0, 0), 'get_provider', 'add_tile directly')
        from bokeh.models import WMTSTileSource
        if isinstance(provider_name, WMTSTileSource):
            return WMTSTileSource(url=provider_name.url, attribution=provider_name.attribution)
        if isinstance(provider_name, str):
            provider_name = provider_name.lower()
            if provider_name == 'esri_imagery':
                provider_name = 'esri_worldimagery'
            if provider_name == 'osm':
                provider_name = 'openstreetmap_mapnik'
            if provider_name.startswith('stamen'):
                provider_name = f'stadia.{provider_name}'
            if 'retina' in provider_name:
                provider_name = provider_name.replace('retina', '')
                retina = True
            else:
                retina = False
            scale_factor = '@2x' if retina else None
            provider_name = xyzservices.providers.query_name(provider_name)
        else:
            scale_factor = None
        if isinstance(provider_name, xyzservices.TileProvider):
            return WMTSTileSource(url=provider_name.build_url(scale_factor=scale_factor), attribution=provider_name.html_attribution, min_zoom=provider_name.get('min_zoom', 0), max_zoom=provider_name.get('max_zoom', 30))
    CARTODBPOSITRON = Vendors.CARTODBPOSITRON
    CARTODBPOSITRON_RETINA = Vendors.CARTODBPOSITRON_RETINA
    STAMEN_TERRAIN = Vendors.STAMEN_TERRAIN
    STAMEN_TERRAIN_RETINA = Vendors.STAMEN_TERRAIN_RETINA
    STAMEN_TONER = Vendors.STAMEN_TONER
    STAMEN_TONER_BACKGROUND = Vendors.STAMEN_TONER_BACKGROUND
    STAMEN_TONER_LABELS = Vendors.STAMEN_TONER_LABELS
    OSM = Vendors.OSM
    ESRI_IMAGERY = Vendors.ESRI_IMAGERY
_mod = _TileProvidersModule('bokeh.tile_providers')
_mod.__doc__ = __doc__
_mod.__all__ = ('CARTODBPOSITRON', 'CARTODBPOSITRON_RETINA', 'STAMEN_TERRAIN', 'STAMEN_TERRAIN_RETINA', 'STAMEN_TONER', 'STAMEN_TONER_BACKGROUND', 'STAMEN_TONER_LABELS', 'OSM', 'ESRI_IMAGERY', 'get_provider', 'Vendors')
sys.modules['bokeh.tile_providers'] = _mod
del _mod, sys, types
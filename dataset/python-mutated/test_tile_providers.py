from __future__ import annotations
import pytest
pytest
import xyzservices.providers as xyz
from bokeh.models import WMTSTileSource
import bokeh.tile_providers as bt
ALL = ('CARTODBPOSITRON', 'CARTODBPOSITRON_RETINA', 'STAMEN_TERRAIN', 'STAMEN_TERRAIN_RETINA', 'STAMEN_TONER', 'STAMEN_TONER_BACKGROUND', 'STAMEN_TONER_LABELS', 'OSM', 'ESRI_IMAGERY', 'get_provider', 'Vendors')
_CARTO_URLS = {'CARTODBPOSITRON': xyz.CartoDB.Positron.build_url(), 'CARTODBPOSITRON_RETINA': xyz.CartoDB.Positron.build_url(scale_factor='@2x')}
_STAMEN_URLS = {'STAMEN_TERRAIN': xyz.Stadia.StamenTerrain.build_url(), 'STAMEN_TERRAIN_RETINA': xyz.Stadia.StamenTerrain.build_url(scale_factor='@2x'), 'STAMEN_TONER': xyz.Stadia.StamenToner.build_url(), 'STAMEN_TONER_BACKGROUND': xyz.Stadia.StamenTonerBackground.build_url(), 'STAMEN_TONER_LABELS': xyz.Stadia.StamenTonerLabels.build_url()}
_STAMEN_ATTR = {'STAMEN_TERRAIN': xyz.Stadia.StamenTerrain.html_attribution, 'STAMEN_TERRAIN_RETINA': xyz.Stadia.StamenTerrain.html_attribution, 'STAMEN_TONER': xyz.Stadia.StamenToner.html_attribution, 'STAMEN_TONER_BACKGROUND': xyz.Stadia.StamenTonerBackground.html_attribution, 'STAMEN_TONER_LABELS': xyz.Stadia.StamenTonerLabels.html_attribution}
_OSM_URLS = {'OSM': xyz.OpenStreetMap.Mapnik.build_url()}
_ESRI_URLS = {'ESRI_IMAGERY': xyz.Esri.WorldImagery.build_url()}

@pytest.mark.parametrize('name', ['STAMEN_TERRAIN', 'STAMEN_TERRAIN_RETINA', 'STAMEN_TONER', 'STAMEN_TONER_BACKGROUND', 'STAMEN_TONER_LABELS'])
class Test_StamenProviders:

    def test_type(self, name) -> None:
        if False:
            return 10
        p = getattr(bt, name)
        assert isinstance(p, str)

    def test_url(self, name) -> None:
        if False:
            print('Hello World!')
        p = bt.get_provider(getattr(bt, name))
        assert p.url == _STAMEN_URLS[name]

    def test_attribution(self, name) -> None:
        if False:
            return 10
        p = bt.get_provider(getattr(bt, name))
        assert p.attribution == _STAMEN_ATTR[name]

    def test_copies(self, name) -> None:
        if False:
            for i in range(10):
                print('nop')
        p1 = bt.get_provider(getattr(bt, name))
        p2 = bt.get_provider(getattr(bt, name))
        assert p1 is not p2

@pytest.mark.parametrize('name', ['CARTODBPOSITRON', 'CARTODBPOSITRON_RETINA'])
class Test_CartoProviders:

    def test_type(self, name) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = getattr(bt, name)
        assert isinstance(p, str)

    def test_url(self, name) -> None:
        if False:
            print('Hello World!')
        p = bt.get_provider(getattr(bt, name))
        assert p.url == _CARTO_URLS[name]

    def test_attribution(self, name) -> None:
        if False:
            print('Hello World!')
        p = bt.get_provider(getattr(bt, name))
        assert p.attribution == xyz.CartoDB.Positron.html_attribution

    def test_copies(self, name) -> None:
        if False:
            for i in range(10):
                print('nop')
        p1 = bt.get_provider(getattr(bt, name))
        p2 = bt.get_provider(getattr(bt, name))
        assert p1 is not p2

@pytest.mark.parametrize('name', ['OSM'])
class Test_OsmProvider:

    def test_type(self, name) -> None:
        if False:
            while True:
                i = 10
        p = getattr(bt, name)
        assert isinstance(p, str)

    def test_url(self, name) -> None:
        if False:
            print('Hello World!')
        p = bt.get_provider(getattr(bt, name))
        assert p.url == _OSM_URLS[name]

    def test_attribution(self, name) -> None:
        if False:
            return 10
        p = bt.get_provider(getattr(bt, name))
        assert p.attribution == xyz.OpenStreetMap.Mapnik.html_attribution

    def test_copies(self, name) -> None:
        if False:
            i = 10
            return i + 15
        p1 = bt.get_provider(getattr(bt, name))
        p2 = bt.get_provider(getattr(bt, name))
        assert p1 is not p2

@pytest.mark.parametrize('name', ['ESRI_IMAGERY'])
class Test_EsriProvider:

    def test_type(self, name) -> None:
        if False:
            i = 10
            return i + 15
        p = getattr(bt, name)
        assert isinstance(p, str)

    def test_url(self, name) -> None:
        if False:
            return 10
        p = bt.get_provider(getattr(bt, name))
        assert p.url == _ESRI_URLS[name]

    def test_attribution(self, name) -> None:
        if False:
            i = 10
            return i + 15
        p = bt.get_provider(getattr(bt, name))
        assert p.attribution == xyz.Esri.WorldImagery.html_attribution

    def test_copies(self, name) -> None:
        if False:
            for i in range(10):
                print('nop')
        p1 = bt.get_provider(getattr(bt, name))
        p2 = bt.get_provider(getattr(bt, name))
        assert p1 is not p2

class Test_GetProvider:

    @pytest.mark.parametrize('name', ['CARTODBPOSITRON', 'CARTODBPOSITRON_RETINA', 'STAMEN_TERRAIN', 'STAMEN_TERRAIN_RETINA', 'STAMEN_TONER', 'STAMEN_TONER_BACKGROUND', 'STAMEN_TONER_LABELS', 'OSM', 'ESRI_IMAGERY'])
    def test_get_provider(self, name) -> None:
        if False:
            while True:
                i = 10
        assert name in bt.Vendors
        enum_member = getattr(bt.Vendors, name)
        assert hasattr(bt, name)
        mod_member = getattr(bt, name)
        p1 = bt.get_provider(enum_member)
        p2 = bt.get_provider(name)
        p3 = bt.get_provider(name.lower())
        p4 = bt.get_provider(mod_member)
        assert isinstance(p1, WMTSTileSource)
        assert isinstance(p2, WMTSTileSource)
        assert isinstance(p3, WMTSTileSource)
        assert isinstance(p4, WMTSTileSource)
        assert p1 is not p2
        assert p2 is not p3
        assert p2 is not p4
        assert p4 is not p1
        assert p1.url == p2.url == p3.url == p4.url
        assert p1.attribution == p2.attribution == p3.attribution == p4.attribution

    def test_unknown_vendor(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            bt.get_provider('This is not a valid tile vendor')

    def test_xyzservices(self) -> None:
        if False:
            i = 10
            return i + 15
        xyzservices = pytest.importorskip('xyzservices')
        provider_data = xyzservices.providers.CartoDB.Positron
        provider = bt.get_provider(provider_data)
        assert isinstance(provider, WMTSTileSource)
        assert provider.url == provider_data.build_url()
        assert provider.attribution == provider_data.html_attribution
        assert provider.min_zoom == provider_data.get('min_zoom', 0)
        assert provider.max_zoom == provider_data.get('max_zoom', 30)
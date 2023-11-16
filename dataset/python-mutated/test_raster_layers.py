"""
Test raster_layers
------------------

"""
import pytest
import xyzservices
from jinja2 import Template
import folium
from folium.utilities import normalize

def test_tile_layer():
    if False:
        for i in range(10):
            print('nop')
    m = folium.Map([48.0, 5.0], zoom_start=6)
    layer = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
    folium.raster_layers.TileLayer(tiles=layer, name='OpenStreetMap', attr='attribution').add_to(m)
    folium.raster_layers.TileLayer(tiles=layer, name='OpenStreetMap2', attr='attribution2', overlay=True).add_to(m)
    folium.LayerControl().add_to(m)
    m._repr_html_()
    bounds = m.get_bounds()
    assert bounds == [[None, None], [None, None]], bounds

def _is_working_zoom_level(zoom, tiles, session):
    if False:
        while True:
            i = 10
    'Check if the zoom level works for the given tileset.'
    url = tiles.format(s='a', x=0, y=0, z=zoom)
    response = session.get(url, timeout=5)
    if response.status_code < 400:
        return True
    return False

def test_custom_tile_subdomains():
    if False:
        return 10
    'Test custom tile subdomains.'
    url = 'http://{s}.custom_tiles.org/{z}/{x}/{y}.png'
    m = folium.Map()
    folium.TileLayer(tiles=url, name='subdomains2', attr='attribution', subdomains='mytilesubdomain').add_to(m)
    out = m._parent.render()
    assert 'mytilesubdomain' in out

def test_wms():
    if False:
        for i in range(10):
            print('nop')
    m = folium.Map([40, -100], zoom_start=4)
    url = 'http://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0r.cgi'
    w = folium.raster_layers.WmsTileLayer(url=url, name='test', fmt='image/png', layers='nexrad-n0r-900913', attr='Weather data © 2012 IEM Nexrad', transparent=True, cql_filter='something')
    w.add_to(m)
    html = m.get_root().render()
    assert '"cql_filter": "something",' in html
    assert 'cqlFilter' not in html
    bounds = m.get_bounds()
    assert bounds == [[None, None], [None, None]], bounds

def test_image_overlay():
    if False:
        return 10
    'Test image overlay.'
    data = [[[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [[1, 1, 0, 0.5], [0, 0, 1, 1], [0, 0, 1, 1]]]
    m = folium.Map()
    io = folium.raster_layers.ImageOverlay(data, [[0, -180], [90, 180]], mercator_project=True)
    io.add_to(m)
    m._repr_html_()
    out = m._parent.render()
    url = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAYAAACddGYaAAAAF0lEQVR42mP4z8AARFDw/z/DeiA5H4QBV60H6ABl9ZIAAAAASUVORK5CYII='
    assert io.url == url
    tmpl = Template('\n        var {{this.get_name()}} = L.imageOverlay(\n            "{{ this.url }}",\n            {{ this.bounds }},\n            {{ this.options }}\n        );\n        {{ this.get_name() }}.addTo({{this._parent.get_name()}});\n    ')
    assert normalize(tmpl.render(this=io)) in normalize(out)
    bounds = m.get_bounds()
    assert bounds == [[0, -180], [90, 180]], bounds

@pytest.mark.parametrize('tiles', ['CartoDB DarkMatter', xyzservices.providers.CartoDB.DarkMatter])
def test_xyzservices(tiles):
    if False:
        for i in range(10):
            print('nop')
    m = folium.Map([48.0, 5.0], tiles=tiles, zoom_start=6)
    folium.raster_layers.TileLayer(tiles=xyzservices.providers.CartoDB.Positron).add_to(m)
    folium.LayerControl().add_to(m)
    out = m._parent.render()
    assert xyzservices.providers.CartoDB.DarkMatter.build_url(fill_subdomain=False, scale_factor='{r}') in out
    assert xyzservices.providers.CartoDB.Positron.build_url(fill_subdomain=False, scale_factor='{r}') in out
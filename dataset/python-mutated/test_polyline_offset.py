"""
Test PolyLineOffset
-------------------
"""
import pytest
import folium
from folium import plugins
from folium.utilities import normalize

@pytest.mark.parametrize('offset', [0, 10, -10])
def test_polylineoffset(offset):
    if False:
        for i in range(10):
            print('nop')
    m = folium.Map([20.0, 0.0], zoom_start=3)
    locations = [[59.3556, -31.99219], [55.17887, -42.89062], [47.7541, -43.94531], [38.27269, -37.96875], [27.05913, -41.13281], [16.29905, -36.5625], [8.40717, -30.23437], [1.05463, -22.5], [-8.75479, -18.28125], [-21.61658, -20.03906], [-31.35364, -24.25781], [-39.90974, -30.9375], [-43.83453, -41.13281], [-47.7541, -49.92187], [-50.95843, -54.14062], [-55.9738, -56.60156]]
    polylineoffset = plugins.PolyLineOffset(locations=locations, offset=offset)
    polylineoffset.add_to(m)
    m._repr_html_()
    out = m._parent.render()
    script = '<script src="https://cdn.jsdelivr.net/npm/leaflet-polylineoffset@1.1.1/leaflet.polylineoffset.min.js"></script>'
    assert script in out
    expected_rendered = f'\n    var {polylineoffset.get_name()} = L.polyline(\n    {locations},\n    {{\n    "bubblingMouseEvents": true,\n    "color": "#3388ff",\n    "dashArray": null,\n    "dashOffset": null,\n    "fill": false,\n    "fillColor": "#3388ff",\n    "fillOpacity": 0.2,\n    "fillRule": "evenodd",\n    "lineCap": "round",\n    "lineJoin": "round",\n    "noClip": false,\n    "offset": {offset},\n    "opacity": 1.0,\n    "smoothFactor": 1.0,\n    "stroke": true,\n    "weight": 3\n    }}\n    )\n    .addTo({m.get_name()});\n    '
    rendered = polylineoffset._template.module.script(polylineoffset)
    assert normalize(expected_rendered) == normalize(rendered)

def test_polylineoffset_without_offset():
    if False:
        return 10
    m = folium.Map([20.0, 0.0], zoom_start=3)
    locations = [[59.3556, -31.99219], [55.17887, -42.89062]]
    polylineoffset = plugins.PolyLineOffset(locations=locations)
    polylineoffset.add_to(m)
    m._repr_html_()
    out = m._parent.render()
    script = '<script src="https://cdn.jsdelivr.net/npm/leaflet-polylineoffset@1.1.1/leaflet.polylineoffset.min.js"></script>'
    assert script in out
    expected_rendered = f'\n    var {polylineoffset.get_name()} = L.polyline(\n    {locations},\n    {{\n    "bubblingMouseEvents": true,\n    "color": "#3388ff",\n    "dashArray": null,\n    "dashOffset": null,\n    "fill": false,\n    "fillColor": "#3388ff",\n    "fillOpacity": 0.2,\n    "fillRule": "evenodd",\n    "lineCap": "round",\n    "lineJoin": "round",\n    "noClip": false,\n    "offset": 0,\n    "opacity": 1.0,\n    "smoothFactor": 1.0,\n    "stroke": true,\n    "weight": 3\n    }}\n    )\n    .addTo({m.get_name()});\n    '
    rendered = polylineoffset._template.module.script(polylineoffset)
    assert normalize(expected_rendered) == normalize(rendered)
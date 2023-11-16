"""
Test HeatMap
------------
"""
import numpy as np
import pytest
from jinja2 import Template
import folium
from folium.plugins import HeatMap
from folium.utilities import normalize

def test_heat_map():
    if False:
        return 10
    np.random.seed(3141592)
    data = np.random.normal(size=(100, 2)) * np.array([[1, 1]]) + np.array([[48, 5]])
    m = folium.Map([48.0, 5.0], zoom_start=6)
    hm = HeatMap(data)
    m.add_child(hm)
    m._repr_html_()
    out = normalize(m._parent.render())
    script = '<script src="https://cdn.jsdelivr.net/gh/python-visualization/folium@main/folium/templates/leaflet_heat.min.js"></script>'
    assert script in out
    tmpl = Template('\n            var {{this.get_name()}} = L.heatLayer(\n                {{this.data}},\n                {\n                    minOpacity: {{this.min_opacity}},\n                    maxZoom: {{this.max_zoom}},\n                    radius: {{this.radius}},\n                    blur: {{this.blur}},\n                    gradient: {{this.gradient}}\n                    })\n                .addTo({{this._parent.get_name()}});\n    ')
    assert tmpl.render(this=hm)
    bounds = m.get_bounds()
    np.testing.assert_allclose(bounds, [[46.218566840847025, 3.0302801394447734], [50.75345011431167, 7.132453997672826]])

def test_heatmap_data():
    if False:
        for i in range(10):
            print('nop')
    data = HeatMap(np.array([[3, 4, 1], [5, 6, 1], [7, 8, 0.5]])).data
    assert isinstance(data, list)
    assert len(data) == 3
    for i in range(len(data)):
        assert isinstance(data[i], list)
        assert len(data[i]) == 3

def test_heat_map_exception():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        HeatMap(np.array([[4, 5, 1], [3, 6, np.nan]]))
    with pytest.raises(Exception):
        HeatMap(np.array([3, 4, 5]))
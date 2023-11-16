"""
Test PolyLineTextPath
---------------
"""
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_polyline_text_path():
    if False:
        for i in range(10):
            print('nop')
    m = folium.Map([20.0, 0.0], zoom_start=3)
    wind_locations = [[59.3556, -31.99219], [55.17887, -42.89062], [47.7541, -43.94531], [38.27269, -37.96875], [27.05913, -41.13281], [16.29905, -36.5625], [8.40717, -30.23437], [1.05463, -22.5], [-8.75479, -18.28125], [-21.61658, -20.03906], [-31.35364, -24.25781], [-39.90974, -30.9375], [-43.83453, -41.13281], [-47.7541, -49.92187], [-50.95843, -54.14062], [-55.9738, -56.60156]]
    wind_line = folium.PolyLine(wind_locations, weight=15, color='#8EE9FF')
    attr = {'fill': '#007DEF', 'font-weight': 'bold', 'font-size': '24'}
    wind_textpath = plugins.PolyLineTextPath(wind_line, ') ', repeat=True, offset=7, attributes=attr)
    m.add_child(wind_line)
    m.add_child(wind_textpath)
    out = normalize(m._parent.render())
    script = '<script src="https://cdn.jsdelivr.net/npm/leaflet-textpath@1.2.3/leaflet.textpath.min.js"></script>'
    assert script in out
    tmpl = Template('\n        {{ this.polyline.get_name() }}.setText(\n            "{{this.text}}",\n            {{ this.options|tojson }}\n        );\n        ')
    expected = normalize(tmpl.render(this=wind_textpath))
    assert expected in out
"""
Test AntPath
-------------
"""
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_antpath():
    if False:
        print('Hello World!')
    m = folium.Map([20.0, 0.0], zoom_start=3)
    locations = [[59.3556, -31.99219], [55.17887, -42.89062], [47.7541, -43.94531], [38.27269, -37.96875], [27.05913, -41.13281], [16.29905, -36.5625], [8.40717, -30.23437], [1.05463, -22.5], [-8.75479, -18.28125], [-21.61658, -20.03906], [-31.35364, -24.25781], [-39.90974, -30.9375], [-43.83453, -41.13281], [-47.7541, -49.92187], [-50.95843, -54.14062], [-55.9738, -56.60156]]
    antpath = plugins.AntPath(locations=locations)
    antpath.add_to(m)
    out = m._parent.render()
    script = '<script src="https://cdn.jsdelivr.net/npm/leaflet-ant-path@1.1.2/dist/leaflet-ant-path.min.js"></script>'
    assert script in out
    tmpl = Template('\n          {{this.get_name()}} = L.polyline.antPath(\n                  {{ this.locations|tojson }},\n                  {{ this.options|tojson }}\n                )\n                .addTo({{this._parent.get_name()}});\n        ')
    expected_rendered = tmpl.render(this=antpath)
    rendered = antpath._template.module.script(antpath)
    assert normalize(expected_rendered) == normalize(rendered)
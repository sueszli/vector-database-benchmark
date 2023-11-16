"""
Test Fullscreen
----------------

"""
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_fullscreen():
    if False:
        return 10
    m = folium.Map([47, 3], zoom_start=1)
    fs = plugins.Fullscreen().add_to(m)
    out = normalize(m._parent.render())
    tmpl = Template('\n        L.control.fullscreen(\n            {{ this.options|tojson }}\n        ).addTo({{this._parent.get_name()}});\n    ')
    assert normalize(tmpl.render(this=fs)) in out
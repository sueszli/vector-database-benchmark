"""
Test TagFilterButton
------------
"""
import random
import numpy as np
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_tag_filter_button():
    if False:
        i = 10
        return i + 15
    np.random.seed(3141592)
    initial_data = np.random.normal(size=(100, 2)) * np.array([[1, 1]]) + np.array([[48, 5]])
    n = 5
    categories = [f'category{i + 1}' for i in range(n)]
    category_column = [random.choice(categories) for i in range(len(initial_data))]
    m = folium.Map([48.0, 5.0], zoom_start=6)
    for (i, latlng) in enumerate(initial_data):
        category = category_column[i]
        folium.Marker(tuple(latlng), tags=[category]).add_to(m)
    hm = plugins.TagFilterButton(categories).add_to(m)
    out = normalize(m._parent.render())
    script = '<script src="https://cdn.jsdelivr.net/npm/leaflet-tag-filter-button/src/leaflet-tag-filter-button.js"></script>'
    assert script in out
    script = '<script src="https://cdn.jsdelivr.net/npm/leaflet-easybutton@2/src/easy-button.js"></script>'
    assert script in out
    script = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-tag-filter-button/src/leaflet-tag-filter-button.css"/>'
    assert script in out
    script = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-easybutton@2/src/easy-button.css"/>'
    assert script in out
    script = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/css-ripple-effect@1.0.5/dist/ripple.min.css"/>'
    assert script in out
    tmpl = Template('\n            var {{this.get_name()}} = L.control.tagFilterButton(\n                {\n                    data: {{this.options.data}},\n                    icon: "{{this.options.icon}}",\n                    clearText: {{this.options.clear_text}},\n                    filterOnEveryClick: {{this.options.filter_on_every_click}},\n                    openPopupOnHover: {{this.options.open_popup_on_hover}}\n                    })\n                .addTo({{this._parent.get_name()}});\n    ')
    assert normalize(tmpl.render(this=hm))
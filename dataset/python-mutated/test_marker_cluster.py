"""
Test MarkerCluster
------------------
"""
import numpy as np
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_marker_cluster():
    if False:
        for i in range(10):
            print('nop')
    N = 100
    np.random.seed(seed=26082009)
    data = np.array([np.random.uniform(low=35, high=60, size=N), np.random.uniform(low=-12, high=30, size=N)]).T
    m = folium.Map([45.0, 3.0], zoom_start=4)
    mc = plugins.MarkerCluster(data).add_to(m)
    out = normalize(m._parent.render())
    assert '<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/leaflet.markercluster.js"></script>' in out
    assert '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.css"/>' in out
    assert '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.Default.css"/>' in out
    tmpl = Template('\n        var {{this.get_name()}} = L.markerClusterGroup(\n            {{ this.options|tojson }}\n        );\n        {%- if this.icon_create_function is not none %}\n            {{ this.get_name() }}.options.iconCreateFunction =\n                {{ this.icon_create_function.strip() }};\n            {%- endif %}\n\n        {% for marker in this._children.values() %}\n            var {{marker.get_name()}} = L.marker(\n                {{ marker.location|tojson }},\n                {}\n            ).addTo({{this.get_name()}});\n        {% endfor %}\n\n        {{ this.get_name() }}.addTo({{ this._parent.get_name() }});\n    ')
    expected = normalize(tmpl.render(this=mc))
    assert expected in out
    bounds = m.get_bounds()
    np.testing.assert_allclose(bounds, [[35.147332572663785, -11.520684337300109], [59.839718052359274, 29.94931046497927]])
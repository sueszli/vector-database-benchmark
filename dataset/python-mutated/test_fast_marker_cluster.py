"""
Test FastMarkerCluster
----------------------
"""
import numpy as np
import pandas as pd
import pytest
from jinja2 import Template
import folium
from folium.plugins import FastMarkerCluster
from folium.utilities import normalize

def test_fast_marker_cluster():
    if False:
        print('Hello World!')
    n = 100
    np.random.seed(seed=26082009)
    data = np.array([np.random.uniform(low=35, high=60, size=n), np.random.uniform(low=-12, high=30, size=n)]).T
    m = folium.Map([45.0, 3.0], zoom_start=4)
    mc = FastMarkerCluster(data).add_to(m)
    out = normalize(m._parent.render())
    assert '<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/leaflet.markercluster.js"></script>' in out
    assert '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.css"/>' in out
    assert '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.Default.css"/>' in out
    tmpl = Template('\n        var {{ this.get_name() }} = (function(){\n            {{ this.callback }}\n\n            var data = {{ this.data|tojson }};\n            var cluster = L.markerClusterGroup({{ this.options|tojson }});\n            {%- if this.icon_create_function is not none %}\n            cluster.options.iconCreateFunction =\n                {{ this.icon_create_function.strip() }};\n            {%- endif %}\n\n            for (var i = 0; i < data.length; i++) {\n                var row = data[i];\n                var marker = callback(row);\n                marker.addTo(cluster);\n            }\n\n            cluster.addTo({{ this._parent.get_name() }});\n            return cluster;\n        })();\n    ')
    expected = normalize(tmpl.render(this=mc))
    assert expected in out

@pytest.mark.parametrize('case', [np.array([[0, 5, 1], [1, 6, 1], [2, 7, 0.5]]), [[0, 5, 'red'], (1, 6, 'blue'), [2, 7, {'this': 'also works'}]], pd.DataFrame([[0, 5, 'red'], [1, 6, 'blue'], [2, 7, 'something']], columns=['lat', 'lng', 'color'])])
def test_fast_marker_cluster_data(case):
    if False:
        for i in range(10):
            print('nop')
    data = FastMarkerCluster(case).data
    assert isinstance(data, list)
    assert len(data) == 3
    for i in range(len(data)):
        assert isinstance(data[i], list)
        assert len(data[i]) == 3
        assert data[i][0] == float(i)
        assert data[i][1] == float(i + 5)
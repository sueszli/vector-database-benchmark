"""
Test VectorGridProtobuf
---------------
"""
import json
import folium
from folium.plugins import VectorGridProtobuf
from folium.utilities import normalize

def test_vectorgrid():
    if False:
        for i in range(10):
            print('nop')
    m = folium.Map(location=(30, 20), zoom_start=4)
    url = 'https://free-{s}.tilehosting.com/data/v3/{z}/{x}/{y}.pbf?token={token}'
    vc = VectorGridProtobuf(url, 'test').add_to(m)
    out = normalize(m._parent.render())
    expected = normalize(VectorGridProtobuf._template.render(this=vc))
    assert expected in out
    script = f'<script src="{VectorGridProtobuf.default_js[0][1]}"></script>'
    assert script in out
    assert url in out
    assert 'L.vectorGrid.protobuf' in out

def test_vectorgrid_str_options():
    if False:
        for i in range(10):
            print('nop')
    m = folium.Map(location=(30, 20), zoom_start=4)
    url = 'https://free-{s}.tilehosting.com/data/v3/{z}/{x}/{y}.pbf?token={token}'
    options = '{\n        "subdomain": "test",\n        "token": "test_token",\n        "vectorTileLayerStyles": {\n            "all": {\n                "fill": true,\n                "weight": 1,\n                "fillColor": "green",\n                "color": "black",\n                "fillOpacity": 0.6,\n                "opacity": 0.6\n                }\n            }\n        }'
    vc = VectorGridProtobuf(url, 'test', options)
    m.add_child(vc)
    dict_options = json.loads(options)
    out = normalize(m._parent.render())
    script = f'<script src="{VectorGridProtobuf.default_js[0][1]}"></script>'
    assert script in out
    assert url in out
    assert 'L.vectorGrid.protobuf' in out
    assert '"token": "test_token"' in out
    assert '"subdomain": "test"' in out
    for (k, v) in dict_options['vectorTileLayerStyles']['all'].items():
        if type(v) == bool:
            assert f'"{k}": {str(v).lower()}' in out
            continue
        if type(v) == str:
            assert f'"{k}": "{v}"' in out
            continue
        assert f'"{k}": {v}' in out

def test_vectorgrid_dict_options():
    if False:
        return 10
    m = folium.Map(location=(30, 20), zoom_start=4)
    url = 'https://free-{s}.tilehosting.com/data/v3/{z}/{x}/{y}.pbf?token={token}'
    options = {'subdomain': 'test', 'token': 'test_token', 'vectorTileLayerStyles': {'all': {'fill': True, 'weight': 1, 'fillColor': 'grey', 'color': 'purple', 'fillOpacity': 0.3, 'opacity': 0.6}}}
    vc = VectorGridProtobuf(url, 'test', options)
    m.add_child(vc)
    out = normalize(m._parent.render())
    script = f'<script src="{VectorGridProtobuf.default_js[0][1]}"></script>'
    assert script in out
    assert url in out
    assert 'L.vectorGrid.protobuf' in out
    assert '"token": "test_token"' in out
    assert '"subdomain": "test"' in out
    for (k, v) in options['vectorTileLayerStyles']['all'].items():
        if type(v) == bool:
            assert f'"{k}": {str(v).lower()}' in out
            continue
        if type(v) == str:
            assert f'"{k}": "{v}"' in out
            continue
        assert f'"{k}": {v}' in out
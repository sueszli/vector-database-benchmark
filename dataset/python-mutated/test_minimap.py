"""
Test MiniMap
---------------
"""
import folium
from folium import plugins
from folium.utilities import normalize

def test_minimap():
    if False:
        while True:
            i = 10
    m = folium.Map(location=(30, 20), zoom_start=4)
    minimap = plugins.MiniMap()
    m.add_child(minimap)
    out = normalize(m._parent.render())
    assert 'new L.Control.MiniMap' in out
    m = folium.Map(tiles=None, location=(30, 20), zoom_start=4)
    minimap = plugins.MiniMap()
    minimap.add_to(m)
    out = normalize(m._parent.render())
    assert 'https://tile.openstreetmap.org' in out
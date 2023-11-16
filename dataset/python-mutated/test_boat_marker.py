"""
Test BoatMarker
---------------

"""
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_boat_marker():
    if False:
        return 10
    m = folium.Map([30.0, 0.0], zoom_start=3)
    bm1 = plugins.BoatMarker((34, -43), heading=45, wind_heading=150, wind_speed=45, color='#8f8')
    bm2 = plugins.BoatMarker((46, -30), heading=-20, wind_heading=46, wind_speed=25, color='#88f')
    m.add_child(bm1)
    m.add_child(bm2)
    m._repr_html_()
    out = normalize(m._parent.render())
    script = '<script src="https://unpkg.com/leaflet.boatmarker/leaflet.boatmarker.min.js"></script>'
    assert script in out
    tmpl = Template('\n        var {{ this.get_name() }} = L.boatMarker(\n            {{ this.location|tojson }},\n            {{ this.options|tojson }}\n        ).addTo({{ this._parent.get_name() }});\n        {{ this.get_name() }}.setHeadingWind(\n            {{ this.heading }},\n            {{ this.wind_speed }},\n            {{ this.wind_heading }}\n        );\n    ')
    assert normalize(tmpl.render(this=bm1)) in out
    assert normalize(tmpl.render(this=bm2)) in out
    bounds = m.get_bounds()
    assert bounds == [[34, -43], [46, -30]], bounds

def test_boat_marker_with_no_wind_speed_or_heading():
    if False:
        for i in range(10):
            print('nop')
    m = folium.Map([30.0, 0.0], zoom_start=3)
    bm1 = plugins.BoatMarker((34, -43), heading=45, color='#8f8')
    m.add_child(bm1)
    out = normalize(m._parent.render())
    tmpl = Template('\n        var {{ this.get_name() }} = L.boatMarker(\n            {{ this.location|tojson }},\n            {{ this.options|tojson }}\n        ).addTo({{ this._parent.get_name() }});\n        {{ this.get_name() }}.setHeading({{ this.heading }});\n    ')
    assert normalize(tmpl.render(this=bm1)) in out
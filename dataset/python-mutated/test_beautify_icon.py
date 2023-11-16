"""
Test BeautifyIcon
---------------

"""
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_beautify_icon():
    if False:
        while True:
            i = 10
    m = folium.Map([30.0, 0.0], zoom_start=3)
    ic1 = plugins.BeautifyIcon(icon='plane', border_color='#b3334f', text_color='#b3334f')
    ic2 = plugins.BeautifyIcon(border_color='#00ABDC', text_color='#00ABDC', number=10, inner_icon_style='margin-top:0;')
    bm1 = folium.Marker(location=[46, -122], popup='Portland, OR', icon=ic1).add_to(m)
    bm2 = folium.Marker(location=[50, -121], icon=ic2).add_to(m)
    m.add_child(bm1)
    m.add_child(bm2)
    m._repr_html_()
    out = normalize(m._parent.render())
    script = '<script src="https://cdn.jsdelivr.net/gh/marslan390/BeautifyMarker/leaflet-beautify-marker-icon.min.js"></script>'
    assert script in out
    css = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/marslan390/BeautifyMarker/leaflet-beautify-marker-icon.min.css"/>'
    assert css in out
    tmpl = Template('\n                var {{this.get_name()}} = new L.BeautifyIcon.icon({{ this.options|tojson }})\n                {{this._parent.get_name()}}.setIcon({{this.get_name()}});\n            ')
    assert normalize(tmpl.render(this=ic1)) in out
    assert normalize(tmpl.render(this=ic2)) in out
"""
Test FloatImage
---------------
"""
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_float_image():
    if False:
        print('Hello World!')
    m = folium.Map([45.0, 3.0], zoom_start=4)
    url = 'https://raw.githubusercontent.com/SECOORA/static_assets/master/maps/img/rose.png'
    szt = plugins.FloatImage(url, bottom=60, left=70, width='20%')
    m.add_child(szt)
    m._repr_html_()
    out = normalize(m._parent.render())
    tmpl = Template('\n        <img id="{{this.get_name()}}" alt="float_image"\n        src="https://raw.githubusercontent.com/SECOORA/static_assets/master/maps/img/rose.png"\n        style="z-index: 999999">\n        </img>\n    ')
    assert normalize(tmpl.render(this=szt)) in out
    tmpl = Template('\n        <style>\n            #{{this.get_name()}} {\n                position: absolute;\n                bottom: 60%;\n                left: 70%;\n                width: 20%;\n                }\n        </style>\n    ')
    assert normalize(tmpl.render(this=szt)) in out
    bounds = m.get_bounds()
    assert bounds == [[None, None], [None, None]], bounds
"""
Test ScrollZoomToggler
----------------------
"""
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_scroll_zoom_toggler():
    if False:
        print('Hello World!')
    m = folium.Map([45.0, 3.0], zoom_start=4)
    szt = plugins.ScrollZoomToggler()
    m.add_child(szt)
    out = normalize(m._parent.render())
    tmpl = Template('\n        <img id="{{this.get_name()}}" alt="scroll"\n        src="https://cdnjs.cloudflare.com/ajax/libs/ionicons/2.0.1/png/512/arrow-move.png"\n        style="z-index: 999999"\n        onclick="{{this._parent.get_name()}}.toggleScroll()"></img>\n    ')
    assert ''.join(tmpl.render(this=szt).split()) in ''.join(out.split())
    tmpl = Template('\n        <style>\n            #{{this.get_name()}} {\n                position:absolute;\n                width:35px;\n                bottom:10px;\n                height:35px;\n                left:10px;\n                background-color:#fff;\n                text-align:center;\n                line-height:35px;\n                vertical-align: middle;\n                }\n        </style>\n    ')
    expected = normalize(tmpl.render(this=szt))
    assert expected in out
    tmpl = Template('\n        {{this._parent.get_name()}}.scrollEnabled = true;\n\n        {{this._parent.get_name()}}.toggleScroll = function() {\n            if (this.scrollEnabled) {\n                this.scrollEnabled = false;\n                this.scrollWheelZoom.disable();\n            } else {\n                this.scrollEnabled = true;\n                this.scrollWheelZoom.enable();\n            }\n        };\n\n        {{this._parent.get_name()}}.toggleScroll();\n    ')
    expected = normalize(tmpl.render(this=szt))
    assert expected in out
    bounds = m.get_bounds()
    assert bounds == [[None, None], [None, None]], bounds
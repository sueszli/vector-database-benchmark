from branca.element import MacroElement
from jinja2 import Template
from folium.elements import JSCSSMixin

class Terminator(JSCSSMixin, MacroElement):
    """
    Leaflet.Terminator is a simple plug-in to the Leaflet library to
    overlay day and night regions on maps.

    """
    _template = Template('\n        {% macro script(this, kwargs) %}\n            L.terminator().addTo({{this._parent.get_name()}});\n        {% endmacro %}\n        ')
    default_js = [('terminator', 'https://unpkg.com/@joergdietrich/leaflet.terminator')]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._name = 'Terminator'
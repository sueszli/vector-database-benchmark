from branca.element import MacroElement
from jinja2 import Template
from folium.elements import JSCSSMixin
from folium.utilities import parse_options

class MeasureControl(JSCSSMixin, MacroElement):
    """Add a measurement widget on the map.

    Parameters
    ----------
    position: str, default 'topright'
        Location of the widget.
    primary_length_unit: str, default 'meters'
    secondary_length_unit: str, default 'miles'
    primary_area_unit: str, default 'sqmeters'
    secondary_area_unit: str, default 'acres'

    See https://github.com/ljagis/leaflet-measure for more information.

    """
    _template = Template('\n        {% macro script(this, kwargs) %}\n            var {{ this.get_name() }} = new L.Control.Measure(\n                {{ this.options|tojson }});\n            {{this._parent.get_name()}}.addControl({{this.get_name()}});\n\n            // Workaround for using this plugin with Leaflet>=1.8.0\n            // https://github.com/ljagis/leaflet-measure/issues/171\n            L.Control.Measure.include({\n                _setCaptureMarkerIcon: function () {\n                    // disable autopan\n                    this._captureMarker.options.autoPanOnFocus = false;\n                    // default function\n                    this._captureMarker.setIcon(\n                        L.divIcon({\n                            iconSize: this._map.getSize().multiplyBy(2)\n                        })\n                    );\n                },\n            });\n\n        {% endmacro %}\n        ')
    default_js = [('leaflet_measure_js', 'https://cdn.jsdelivr.net/gh/ljagis/leaflet-measure@2.1.7/dist/leaflet-measure.min.js')]
    default_css = [('leaflet_measure_css', 'https://cdn.jsdelivr.net/gh/ljagis/leaflet-measure@2.1.7/dist/leaflet-measure.min.css')]

    def __init__(self, position='topright', primary_length_unit='meters', secondary_length_unit='miles', primary_area_unit='sqmeters', secondary_area_unit='acres', **kwargs):
        if False:
            while True:
                i = 10
        super().__init__()
        self._name = 'MeasureControl'
        self.options = parse_options(position=position, primary_length_unit=primary_length_unit, secondary_length_unit=secondary_length_unit, primary_area_unit=primary_area_unit, secondary_area_unit=secondary_area_unit, **kwargs)
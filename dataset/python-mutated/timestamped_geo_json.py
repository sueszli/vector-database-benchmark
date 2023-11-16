import json
from branca.element import MacroElement
from jinja2 import Template
from folium.elements import JSCSSMixin
from folium.folium import Map
from folium.utilities import get_bounds, parse_options

class TimestampedGeoJson(JSCSSMixin, MacroElement):
    """
    Creates a TimestampedGeoJson plugin from timestamped GeoJSONs to append
    into a map with Map.add_child.

    A geo-json is timestamped if:

    * it contains only features of types LineString, MultiPoint, MultiLineString,
      Polygon and MultiPolygon.
    * each feature has a 'times' property with the same length as the
      coordinates array.
    * each element of each 'times' property is a timestamp in ms since epoch,
      or in ISO string.

    Eventually, you may have Point features with a 'times' property being an
    array of length 1.

    Parameters
    ----------
    data: file, dict or str.
        The timestamped geo-json data you want to plot.

        * If file, then data will be read in the file and fully embedded in
          Leaflet's javascript.
        * If dict, then data will be converted to json and embedded in the
          javascript.
        * If str, then data will be passed to the javascript as-is.
    transition_time: int, default 200.
        The duration in ms of a transition from between timestamps.
    loop: bool, default True
        Whether the animation shall loop.
    auto_play: bool, default True
        Whether the animation shall start automatically at startup.
    add_last_point: bool, default True
        Whether a point is added at the last valid coordinate of a LineString.
    period: str, default "P1D"
        Used to construct the array of available times starting
        from the first available time. Format: ISO8601 Duration
        ex: 'P1M' 1/month, 'P1D' 1/day, 'PT1H' 1/hour, and 'PT1M' 1/minute
    duration: str, default None
        Period of time which the features will be shown on the map after their
        time has passed. If None, all previous times will be shown.
        Format: ISO8601 Duration
        ex: 'P1M' 1/month, 'P1D' 1/day, 'PT1H' 1/hour, and 'PT1M' 1/minute

    Examples
    --------
    >>> TimestampedGeoJson(
    ...     {
    ...         "type": "FeatureCollection",
    ...         "features": [
    ...             {
    ...                 "type": "Feature",
    ...                 "geometry": {
    ...                     "type": "LineString",
    ...                     "coordinates": [[-70, -25], [-70, 35], [70, 35]],
    ...                 },
    ...                 "properties": {
    ...                     "times": [1435708800000, 1435795200000, 1435881600000],
    ...                     "tooltip": "my tooltip text",
    ...                 },
    ...             }
    ...         ],
    ...     }
    ... )

    See https://github.com/socib/Leaflet.TimeDimension for more information.

    """
    _template = Template('\n        {% macro script(this, kwargs) %}\n            L.Control.TimeDimensionCustom = L.Control.TimeDimension.extend({\n                _getDisplayDateFormat: function(date){\n                    var newdate = new moment(date);\n                    console.log(newdate)\n                    return newdate.format("{{this.date_options}}");\n                }\n            });\n            {{this._parent.get_name()}}.timeDimension = L.timeDimension(\n                {\n                    period: {{ this.period|tojson }},\n                }\n            );\n            var timeDimensionControl = new L.Control.TimeDimensionCustom(\n                {{ this.options|tojson }}\n            );\n            {{this._parent.get_name()}}.addControl(this.timeDimensionControl);\n\n            var geoJsonLayer = L.geoJson({{this.data}}, {\n                    pointToLayer: function (feature, latLng) {\n                        if (feature.properties.icon == \'marker\') {\n                            if(feature.properties.iconstyle){\n                                return new L.Marker(latLng, {\n                                    icon: L.icon(feature.properties.iconstyle)});\n                            }\n                            //else\n                            return new L.Marker(latLng);\n                        }\n                        if (feature.properties.icon == \'circle\') {\n                            if (feature.properties.iconstyle) {\n                                return new L.circleMarker(latLng, feature.properties.iconstyle)\n                                };\n                            //else\n                            return new L.circleMarker(latLng);\n                        }\n                        //else\n\n                        return new L.Marker(latLng);\n                    },\n                    style: function (feature) {\n                        return feature.properties.style;\n                    },\n                    onEachFeature: function(feature, layer) {\n                        if (feature.properties.popup) {\n                        layer.bindPopup(feature.properties.popup);\n                        }\n                        if (feature.properties.tooltip) {\n                        layer.bindTooltip(feature.properties.tooltip);\n                        }\n                    }\n                })\n\n            var {{this.get_name()}} = L.timeDimension.layer.geoJson(\n                geoJsonLayer,\n                {\n                    updateTimeDimension: true,\n                    addlastPoint: {{ this.add_last_point|tojson }},\n                    duration: {{ this.duration }},\n                }\n            ).addTo({{this._parent.get_name()}});\n        {% endmacro %}\n        ')
    default_js = [('jquery3.7.1', 'https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js'), ('jqueryui1.10.2', 'https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.10.2/jquery-ui.min.js'), ('iso8601', 'https://cdn.jsdelivr.net/npm/iso8601-js-period@0.2.1/iso8601.min.js'), ('leaflet.timedimension', 'https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.min.js'), ('moment', 'https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.18.1/moment.min.js')]
    default_css = [('highlight.js_css', 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/8.4/styles/default.min.css'), ('leaflet.timedimension_css', 'https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.control.css')]

    def __init__(self, data, transition_time=200, loop=True, auto_play=True, add_last_point=True, period='P1D', min_speed=0.1, max_speed=10, loop_button=False, date_options='YYYY-MM-DD HH:mm:ss', time_slider_drag_update=False, duration=None, speed_slider=True):
        if False:
            print('Hello World!')
        super().__init__()
        self._name = 'TimestampedGeoJson'
        if 'read' in dir(data):
            self.embed = True
            self.data = data.read()
        elif type(data) is dict:
            self.embed = True
            self.data = json.dumps(data)
        else:
            self.embed = False
            self.data = data
        self.add_last_point = bool(add_last_point)
        self.period = period
        self.date_options = date_options
        self.duration = 'undefined' if duration is None else '"' + duration + '"'
        self.options = parse_options(position='bottomleft', min_speed=min_speed, max_speed=max_speed, auto_play=auto_play, loop_button=loop_button, time_slider_drag_update=time_slider_drag_update, speed_slider=speed_slider, player_options={'transitionTime': int(transition_time), 'loop': loop, 'startOver': True})

    def render(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(self._parent, Map), 'TimestampedGeoJson can only be added to a Map object.'
        super().render(**kwargs)

    def _get_self_bounds(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Computes the bounds of the object itself (not including it's children)\n        in the form [[lat_min, lon_min], [lat_max, lon_max]].\n\n        "
        if not self.embed:
            raise ValueError('Cannot compute bounds of non-embedded GeoJSON.')
        data = json.loads(self.data)
        if 'features' not in data.keys():
            if not (isinstance(data, dict) and 'geometry' in data.keys()):
                data = {'type': 'Feature', 'geometry': data}
            data = {'type': 'FeatureCollection', 'features': [data]}
        return get_bounds(data, lonlat=True)
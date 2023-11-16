"""
Test TimestampedGeoJson
-----------------------

"""
import numpy as np
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_timestamped_geo_json():
    if False:
        i = 10
        return i + 15
    coordinates = [[[[lon - 8 * np.sin(theta), -47 + 6 * np.cos(theta)] for theta in np.linspace(0, 2 * np.pi, 25)], [[lon - 4 * np.sin(theta), -47 + 3 * np.cos(theta)] for theta in np.linspace(0, 2 * np.pi, 25)]] for lon in np.linspace(-150, 150, 7)]
    data = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [0, 0]}, 'properties': {'times': [1435708800000 + 12 * 86400000]}}, {'type': 'Feature', 'geometry': {'type': 'MultiPoint', 'coordinates': [[lon, -25] for lon in np.linspace(-150, 150, 49)]}, 'properties': {'times': [1435708800000 + i * 86400000 for i in np.linspace(0, 25, 49)]}}, {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': [[lon, 25] for lon in np.linspace(-150, 150, 25)]}, 'properties': {'times': [1435708800000 + i * 86400000 for i in np.linspace(0, 25, 25)], 'style': {'color': 'red'}}}, {'type': 'Feature', 'geometry': {'type': 'MultiLineString', 'coordinates': [[[lon - 4 * np.sin(theta), 47 + 3 * np.cos(theta)] for theta in np.linspace(0, 2 * np.pi, 25)] for lon in np.linspace(-150, 150, 13)]}, 'properties': {'times': [1435708800000 + i * 86400000 for i in np.linspace(0, 25, 13)]}}, {'type': 'Feature', 'geometry': {'type': 'MultiPolygon', 'coordinates': coordinates}, 'properties': {'times': [1435708800000 + i * 86400000 for i in np.linspace(0, 25, 7)]}}]}
    m = folium.Map([47, 3], zoom_start=1)
    tgj = plugins.TimestampedGeoJson(data).add_to(m)
    out = normalize(m._parent.render())
    assert '<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>' in out
    assert '<script src="https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.10.2/jquery-ui.min.js"></script>' in out
    assert '<script src="https://cdn.jsdelivr.net/npm/iso8601-js-period@0.2.1/iso8601.min.js"></script>' in out
    assert '<script src="https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.min.js"></script>' in out
    assert '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/8.4/styles/default.min.css"/>' in out
    assert '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.control.css"/>' in out
    assert '<script src="https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.18.1/moment.min.js"></script>' in out
    tmpl = Template('\n        L.Control.TimeDimensionCustom = L.Control.TimeDimension.extend({\n            _getDisplayDateFormat: function(date){\n                var newdate = new moment(date);\n                console.log(newdate)\n                return newdate.format("{{this.date_options}}");\n            }\n        });\n        {{this._parent.get_name()}}.timeDimension = L.timeDimension(\n            {\n                period: {{ this.period|tojson }},\n            }\n        );\n        var timeDimensionControl = new L.Control.TimeDimensionCustom(\n            {{ this.options|tojson }}\n        );\n        {{this._parent.get_name()}}.addControl(this.timeDimensionControl);\n\n        var geoJsonLayer = L.geoJson({{this.data}}, {\n                pointToLayer: function (feature, latLng) {\n                    if (feature.properties.icon == \'marker\') {\n                        if(feature.properties.iconstyle){\n                            return new L.Marker(latLng, {\n                                icon: L.icon(feature.properties.iconstyle)});\n                        }\n                        //else\n                        return new L.Marker(latLng);\n                    }\n                    if (feature.properties.icon == \'circle\') {\n                        if (feature.properties.iconstyle) {\n                            return new L.circleMarker(latLng, feature.properties.iconstyle)\n                            };\n                        //else\n                        return new L.circleMarker(latLng);\n                    }\n                    //else\n\n                    return new L.Marker(latLng);\n                },\n                style: function (feature) {\n                    return feature.properties.style;\n                },\n                onEachFeature: function(feature, layer) {\n                    if (feature.properties.popup) {\n                    layer.bindPopup(feature.properties.popup);\n                    }\n                    if (feature.properties.tooltip) {\n                        layer.bindTooltip(feature.properties.tooltip);\n                    }\n                }\n            })\n\n        var {{this.get_name()}} = L.timeDimension.layer.geoJson(\n            geoJsonLayer,\n            {\n                updateTimeDimension: true,\n                addlastPoint: {{ this.add_last_point|tojson }},\n                duration: {{ this.duration }},\n            }\n        ).addTo({{this._parent.get_name()}});\n    ')
    expected = normalize(tmpl.render(this=tgj))
    assert expected in out
    bounds = m.get_bounds()
    assert bounds == [[-53.0, -158.0], [50.0, 158.0]], bounds
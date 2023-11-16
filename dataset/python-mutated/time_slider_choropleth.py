from jinja2 import Template
from folium.elements import JSCSSMixin
from folium.features import GeoJson
from folium.map import Layer

class TimeSliderChoropleth(JSCSSMixin, Layer):
    """
    Create a choropleth with a timeslider for timestamped data.

    Visualize timestamped data, allowing users to view the choropleth at
    different timestamps using a slider.

    Parameters
    ----------
    data: str
        geojson string
    styledict: dict
        A dictionary where the keys are the geojson feature ids and the values are
        dicts of `{time: style_options_dict}`
    highlight: bool, default False
        Whether to show a visual effect on mouse hover and click.
    name : string, default None
        The name of the Layer, as it will appear in LayerControls.
    overlay : bool, default False
        Adds the layer as an optional overlay (True) or the base layer (False).
    control : bool, default True
        Whether the Layer will be included in LayerControls.
    show: bool, default True
        Whether the layer will be shown on opening.
    init_timestamp: int, default 0
        Initial time-stamp index on the slider. Must be in the range
        `[-L, L-1]`, where `L` is the maximum number of time stamps in
        `styledict`. For example, use `-1` to initialize the slider to the
        latest timestamp.
    """
    _template = Template('\n        {% macro script(this, kwargs) %}\n        {\n            let timestamps = {{ this.timestamps|tojson }};\n            let styledict = {{ this.styledict|tojson }};\n            let current_timestamp = timestamps[{{ this.init_timestamp }}];\n\n            let slider_body = d3.select("body").insert("div", "div.folium-map")\n                .attr("id", "slider_{{ this.get_name() }}");\n            $("#slider_{{ this.get_name() }}").hide();\n            // insert time slider label\n            slider_body.append("output")\n                .attr("width", "100")\n                .style(\'font-size\', \'18px\')\n                .style(\'text-align\', \'center\')\n                .style(\'font-weight\', \'500%\')\n                .style(\'margin\', \'5px\');\n            // insert time slider\n            slider_body.append("input")\n                .attr("type", "range")\n                .attr("width", "100px")\n                .attr("min", 0)\n                .attr("max", timestamps.length - 1)\n                .attr("value", {{ this.init_timestamp }})\n                .attr("step", "1")\n                .style(\'align\', \'center\');\n\n            let datestring = new Date(parseInt(current_timestamp)*1000).toDateString();\n            d3.select("#slider_{{ this.get_name() }} > output").text(datestring);\n\n            let fill_map = function(){\n                for (var feature_id in styledict){\n                    let style = styledict[feature_id]//[current_timestamp];\n                    var fillColor = \'white\';\n                    var opacity = 0;\n                    if (current_timestamp in style){\n                        fillColor = style[current_timestamp][\'color\'];\n                        opacity = style[current_timestamp][\'opacity\'];\n                        d3.selectAll(\'#{{ this.get_name() }}-feature-\'+feature_id\n                        ).attr(\'fill\', fillColor)\n                        .style(\'fill-opacity\', opacity);\n                    }\n                }\n            }\n\n            d3.select("#slider_{{ this.get_name() }} > input").on("input", function() {\n                current_timestamp = timestamps[this.value];\n                var datestring = new Date(parseInt(current_timestamp)*1000).toDateString();\n                d3.select("#slider_{{ this.get_name() }} > output").text(datestring);\n                fill_map();\n            });\n\n            let onEachFeature;\n            {% if this.highlight %}\n                 onEachFeature = function(feature, layer) {\n                    layer.on({\n                        mouseout: function(e) {\n                        if (current_timestamp in styledict[e.target.feature.id]){\n                            var opacity = styledict[e.target.feature.id][current_timestamp][\'opacity\'];\n                            d3.selectAll(\'#{{ this.get_name() }}-feature-\'+e.target.feature.id).style(\'fill-opacity\', opacity);\n                        }\n                    },\n                        mouseover: function(e) {\n                        if (current_timestamp in styledict[e.target.feature.id]){\n                            d3.selectAll(\'#{{ this.get_name() }}-feature-\'+e.target.feature.id).style(\'fill-opacity\', 1);\n                        }\n                    },\n                        click: function(e) {\n                            {{this._parent.get_name()}}.fitBounds(e.target.getBounds());\n                    }\n                    });\n                };\n            {% endif %}\n\n            var {{ this.get_name() }} = L.geoJson(\n                {{ this.data|tojson }},\n                {onEachFeature: onEachFeature}\n            );\n\n            {{ this.get_name() }}.setStyle(function(feature) {\n                if (feature.properties.style !== undefined){\n                    return feature.properties.style;\n                }\n                else{\n                    return "";\n                }\n            });\n\n            let onOverlayAdd = function(e) {\n                {{ this.get_name() }}.eachLayer(function (layer) {\n                    layer._path.id = \'{{ this.get_name() }}-feature-\' + layer.feature.id;\n                });\n\n                $("#slider_{{ this.get_name() }}").show();\n\n                d3.selectAll(\'path\')\n                .attr(\'stroke\', \'white\')\n                .attr(\'stroke-width\', 0.8)\n                .attr(\'stroke-dasharray\', \'5,5\')\n                .attr(\'fill-opacity\', 0);\n\n                fill_map();\n            }\n            {{ this.get_name() }}.on(\'add\', onOverlayAdd);\n            {{ this.get_name() }}.on(\'remove\', function() {\n                $("#slider_{{ this.get_name() }}").hide();\n            })\n\n            {%- if this.show %}\n            {{ this.get_name() }}.addTo({{ this._parent.get_name() }});\n            $("#slider_{{ this.get_name() }}").show();\n            {%- endif %}\n        }\n        {% endmacro %}\n        ')
    default_js = [('d3v4', 'https://d3js.org/d3.v4.min.js')]

    def __init__(self, data, styledict, highlight: bool=False, name=None, overlay=True, control=True, show=True, init_timestamp=0):
        if False:
            i = 10
            return i + 15
        super().__init__(name=name, overlay=overlay, control=control, show=show)
        self.data = GeoJson.process_data(GeoJson({}), data)
        self.highlight = highlight
        if not isinstance(styledict, dict):
            raise ValueError(f'styledict must be a dictionary, got {styledict!r}')
        for val in styledict.values():
            if not isinstance(val, dict):
                raise ValueError(f'Each item in styledict must be a dictionary, got {val!r}')
        timestamps_set = set()
        for feature in styledict.values():
            timestamps_set.update(set(feature.keys()))
        try:
            timestamps = sorted(timestamps_set, key=int)
        except (TypeError, ValueError):
            timestamps = sorted(timestamps_set)
        self.timestamps = timestamps
        self.styledict = styledict
        assert -len(timestamps) <= init_timestamp < len(timestamps), f'init_timestamp must be in the range [-{len(timestamps)}, {len(timestamps)}) but got {init_timestamp}'
        if init_timestamp < 0:
            init_timestamp = len(timestamps) + init_timestamp
        self.init_timestamp = init_timestamp
"""
Test HeatMapWithTime
------------
"""
import numpy as np
from jinja2 import Template
import folium
from folium import plugins
from folium.utilities import normalize

def test_heat_map_with_time():
    if False:
        print('Hello World!')
    np.random.seed(3141592)
    initial_data = np.random.normal(size=(100, 2)) * np.array([[1, 1]]) + np.array([[48, 5]])
    move_data = np.random.normal(size=(100, 2)) * 0.01
    data = [(initial_data + move_data * i).tolist() for i in range(100)]
    m = folium.Map([48.0, 5.0], zoom_start=6)
    hm = plugins.HeatMapWithTime(data).add_to(m)
    out = normalize(m._parent.render())
    script = '<script src="https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.min.js"></script>'
    assert script in out
    script = '<script src="https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/pa7_hm.min.js"></script>'
    assert script in out
    script = '<script src="https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/pa7_leaflet_hm.min.js"></script>'
    assert script in out
    script = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.control.css"/>'
    assert script in out
    tmpl = Template('\n        var times = {{this.times}};\n\n        {{this._parent.get_name()}}.timeDimension = L.timeDimension(\n            {times : times, currentTime: new Date(1)}\n        );\n\n        var {{this._control_name}} = new L.Control.TimeDimensionCustom({{this.index}}, {\n            autoPlay: {{this.auto_play}},\n            backwardButton: {{this.backward_button}},\n            displayDate: {{this.display_index}},\n            forwardButton: {{this.forward_button}},\n            limitMinimumRange: {{this.limit_minimum_range}},\n            limitSliders: {{this.limit_sliders}},\n            loopButton: {{this.loop_button}},\n            maxSpeed: {{this.max_speed}},\n            minSpeed: {{this.min_speed}},\n            playButton: {{this.play_button}},\n            playReverseButton: {{this.play_reverse_button}},\n            position: "{{this.position}}",\n            speedSlider: {{this.speed_slider}},\n            speedStep: {{this.speed_step}},\n            styleNS: "{{this.style_NS}}",\n            timeSlider: {{this.time_slider}},\n            timeSliderDrapUpdate: {{this.time_slider_drap_update}},\n            timeSteps: {{this.index_steps}}\n            })\n            .addTo({{this._parent.get_name()}});\n\n            var {{this.get_name()}} = new TDHeatmap({{this.data}},\n            {heatmapOptions: {\n                    radius: {{this.radius}},\n                    blur: {{this.blur}},\n                    minOpacity: {{this.min_opacity}},\n                    maxOpacity: {{this.max_opacity}},\n                    scaleRadius: {{this.scale_radius}},\n                    useLocalExtrema: {{this.use_local_extrema}},\n                    defaultWeight: 1,\n                    {% if this.gradient %}gradient: {{ this.gradient }}{% endif %}\n                }\n            });\n            {{ this.get_name() }}.addTo({{ this._parent.get_name() }});\n    ')
    assert normalize(tmpl.render(this=hm)) in out
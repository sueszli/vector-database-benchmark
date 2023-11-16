from branca.element import MacroElement
from jinja2 import Template
from folium.elements import JSCSSMixin
from folium.utilities import parse_options

class GroupedLayerControl(JSCSSMixin, MacroElement):
    """
    Create a Layer Control with groups of overlays.

    Parameters
    ----------
    groups : dict
          A dictionary where the keys are group names and the values are lists
          of layer objects.
          e.g. {
              "Group 1": [layer1, layer2],
              "Group 2": [layer3, layer4]
            }
    exclusive_groups: bool, default True
         Whether to use radio buttons (default) or checkboxes.
         If you want to use both, use two separate instances of this class.
    **kwargs
        Additional (possibly inherited) options. See
        https://leafletjs.com/reference.html#control-layers

    """
    default_js = [('leaflet.groupedlayercontrol.min.js', 'https://cdnjs.cloudflare.com/ajax/libs/leaflet-groupedlayercontrol/0.6.1/leaflet.groupedlayercontrol.min.js')]
    default_css = [('leaflet.groupedlayercontrol.min.css', 'https://cdnjs.cloudflare.com/ajax/libs/leaflet-groupedlayercontrol/0.6.1/leaflet.groupedlayercontrol.min.css')]
    _template = Template('\n        {% macro script(this,kwargs) %}\n\n            L.control.groupedLayers(\n                null,\n                {\n                    {%- for group_name, overlays in this.grouped_overlays.items() %}\n                    {{ group_name|tojson }} : {\n                        {%- for overlaykey, val in overlays.items() %}\n                        {{ overlaykey|tojson }} : {{val}},\n                        {%- endfor %}\n                    },\n                    {%- endfor %}\n                },\n                {{ this.options|tojson }},\n            ).addTo({{this._parent.get_name()}});\n\n            {%- for val in this.layers_untoggle %}\n            {{ val }}.remove();\n            {%- endfor %}\n\n        {% endmacro %}\n        ')

    def __init__(self, groups, exclusive_groups=True, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._name = 'GroupedLayerControl'
        self.options = parse_options(**kwargs)
        if exclusive_groups:
            self.options['exclusiveGroups'] = list(groups.keys())
        self.layers_untoggle = set()
        self.grouped_overlays = {}
        for (group_name, sublist) in groups.items():
            self.grouped_overlays[group_name] = {}
            for element in sublist:
                self.grouped_overlays[group_name][element.layer_name] = element.get_name()
                if not element.show:
                    self.layers_untoggle.add(element.get_name())
                element.control = False
            if exclusive_groups:
                for element in sublist[1:]:
                    self.layers_untoggle.add(element.get_name())
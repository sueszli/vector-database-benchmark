from branca.element import Element, Figure, MacroElement
from jinja2 import Template
from folium.elements import JSCSSMixin

class Draw(JSCSSMixin, MacroElement):
    """
    Vector drawing and editing plugin for Leaflet.

    Parameters
    ----------
    export : bool, default False
        Add a small button that exports the drawn shapes as a geojson file.
    filename : string, default 'data.geojson'
        Name of geojson file
    position : {'topleft', 'toprigth', 'bottomleft', 'bottomright'}
        Position of control.
        See https://leafletjs.com/reference.html#control
    show_geometry_on_click : bool, default True
        When True, opens an alert with the geometry description on click.
    draw_options : dict, optional
        The options used to configure the draw toolbar. See
        http://leaflet.github.io/Leaflet.draw/docs/leaflet-draw-latest.html#drawoptions
    edit_options : dict, optional
        The options used to configure the edit toolbar. See
        https://leaflet.github.io/Leaflet.draw/docs/leaflet-draw-latest.html#editpolyoptions

    Examples
    --------
    >>> m = folium.Map()
    >>> Draw(
    ...     export=True,
    ...     filename="my_data.geojson",
    ...     position="topleft",
    ...     draw_options={"polyline": {"allowIntersection": False}},
    ...     edit_options={"poly": {"allowIntersection": False}},
    ... ).add_to(m)

    For more info please check
    https://leaflet.github.io/Leaflet.draw/docs/leaflet-draw-latest.html

    """
    _template = Template("\n        {% macro script(this, kwargs) %}\n            var options = {\n              position: {{ this.position|tojson }},\n              draw: {{ this.draw_options|tojson }},\n              edit: {{ this.edit_options|tojson }},\n            }\n            // FeatureGroup is to store editable layers.\n            var drawnItems_{{ this.get_name() }} = new L.featureGroup().addTo(\n                {{ this._parent.get_name() }}\n            );\n            options.edit.featureGroup = drawnItems_{{ this.get_name() }};\n            var {{ this.get_name() }} = new L.Control.Draw(\n                options\n            ).addTo( {{this._parent.get_name()}} );\n            {{ this._parent.get_name() }}.on(L.Draw.Event.CREATED, function(e) {\n                var layer = e.layer,\n                    type = e.layerType;\n                var coords = JSON.stringify(layer.toGeoJSON());\n                {%- if this.show_geometry_on_click %}\n                layer.on('click', function() {\n                    alert(coords);\n                    console.log(coords);\n                });\n                {%- endif %}\n                drawnItems_{{ this.get_name() }}.addLayer(layer);\n             });\n            {{ this._parent.get_name() }}.on('draw:created', function(e) {\n                drawnItems_{{ this.get_name() }}.addLayer(e.layer);\n            });\n            {% if this.export %}\n            document.getElementById('export').onclick = function(e) {\n                var data = drawnItems_{{ this.get_name() }}.toGeoJSON();\n                var convertedData = 'text/json;charset=utf-8,'\n                    + encodeURIComponent(JSON.stringify(data));\n                document.getElementById('export').setAttribute(\n                    'href', 'data:' + convertedData\n                );\n                document.getElementById('export').setAttribute(\n                    'download', {{ this.filename|tojson }}\n                );\n            }\n            {% endif %}\n        {% endmacro %}\n        ")
    default_js = [('leaflet_draw_js', 'https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.2/leaflet.draw.js')]
    default_css = [('leaflet_draw_css', 'https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.2/leaflet.draw.css')]

    def __init__(self, export=False, filename='data.geojson', position='topleft', show_geometry_on_click=True, draw_options=None, edit_options=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self._name = 'DrawControl'
        self.export = export
        self.filename = filename
        self.position = position
        self.show_geometry_on_click = show_geometry_on_click
        self.draw_options = draw_options or {}
        self.edit_options = edit_options or {}

    def render(self, **kwargs):
        if False:
            while True:
                i = 10
        super().render(**kwargs)
        figure = self.get_root()
        assert isinstance(figure, Figure), 'You cannot render this Element if it is not in a Figure.'
        export_style = "\n            <style>\n                #export {\n                    position: absolute;\n                    top: 5px;\n                    right: 10px;\n                    z-index: 999;\n                    background: white;\n                    color: black;\n                    padding: 6px;\n                    border-radius: 4px;\n                    font-family: 'Helvetica Neue';\n                    cursor: pointer;\n                    font-size: 12px;\n                    text-decoration: none;\n                    top: 90px;\n                }\n            </style>\n        "
        export_button = "<a href='#' id='export'>Export</a>"
        if self.export:
            figure.header.add_child(Element(export_style), name='export')
            figure.html.add_child(Element(export_button), name='export_button')
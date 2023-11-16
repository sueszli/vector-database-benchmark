from branca.element import MacroElement
from jinja2 import Template

class FloatImage(MacroElement):
    """Adds a floating image in HTML canvas on top of the map.

    Parameters
    ----------
    image: str
        Url to image location. Can also be an inline image using a data URI
        or a local file using `file://`.
    bottom: int, default 75
        Vertical position from the bottom, as a percentage of screen height.
    left: int, default 75
        Horizontal position from the left, as a percentage of screen width.
    **kwargs
        Additional keyword arguments are applied as CSS properties.
        For example: `width='300px'`.

    """
    _template = Template('\n            {% macro header(this,kwargs) %}\n                <style>\n                    #{{this.get_name()}} {\n                        position: absolute;\n                        bottom: {{this.bottom}}%;\n                        left: {{this.left}}%;\n                        {%- for property, value in this.css.items() %}\n                          {{ property }}: {{ value }};\n                        {%- endfor %}\n                        }\n                </style>\n            {% endmacro %}\n\n            {% macro html(this,kwargs) %}\n            <img id="{{this.get_name()}}" alt="float_image"\n                 src="{{ this.image }}"\n                 style="z-index: 999999">\n            </img>\n            {% endmacro %}\n            ')

    def __init__(self, image, bottom=75, left=75, **kwargs):
        if False:
            return 10
        super().__init__()
        self._name = 'FloatImage'
        self.image = image
        self.bottom = bottom
        self.left = left
        self.css = kwargs
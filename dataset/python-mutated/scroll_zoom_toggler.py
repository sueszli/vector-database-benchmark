from branca.element import MacroElement
from jinja2 import Template

class ScrollZoomToggler(MacroElement):
    """Creates a button for enabling/disabling scroll on the Map."""
    _template = Template('\n        {% macro header(this,kwargs) %}\n            <style>\n                #{{ this.get_name() }} {\n                    position:absolute;\n                    width:35px;\n                    bottom:10px;\n                    height:35px;\n                    left:10px;\n                    background-color:#fff;\n                    text-align:center;\n                    line-height:35px;\n                    vertical-align: middle;\n                    }\n            </style>\n        {% endmacro %}\n\n        {% macro html(this,kwargs) %}\n            <img id="{{ this.get_name() }}"\n                 alt="scroll"\n                 src="https://cdnjs.cloudflare.com/ajax/libs/ionicons/2.0.1/png/512/arrow-move.png"\n                 style="z-index: 999999"\n                 onclick="{{ this._parent.get_name() }}.toggleScroll()">\n            </img>\n        {% endmacro %}\n\n        {% macro script(this,kwargs) %}\n            {{ this._parent.get_name() }}.scrollEnabled = true;\n\n            {{ this._parent.get_name() }}.toggleScroll = function() {\n                if (this.scrollEnabled) {\n                    this.scrollEnabled = false;\n                    this.scrollWheelZoom.disable();\n                } else {\n                    this.scrollEnabled = true;\n                    this.scrollWheelZoom.enable();\n                }\n            };\n            {{ this._parent.get_name() }}.toggleScroll();\n        {% endmacro %}\n        ')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._name = 'ScrollZoomToggler'
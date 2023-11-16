from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Symbol(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.mapbox.layer'
    _path_str = 'layout.mapbox.layer.symbol'
    _valid_props = {'icon', 'iconsize', 'placement', 'text', 'textfont', 'textposition'}

    @property
    def icon(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the symbol icon image (mapbox.layer.layout.icon-image).\n        Full list: https://www.mapbox.com/maki-icons/\n\n        The 'icon' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['icon']

    @icon.setter
    def icon(self, val):
        if False:
            return 10
        self['icon'] = val

    @property
    def iconsize(self):
        if False:
            return 10
        '\n        Sets the symbol icon size (mapbox.layer.layout.icon-size). Has\n        an effect only when `type` is set to "symbol".\n\n        The \'iconsize\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['iconsize']

    @iconsize.setter
    def iconsize(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['iconsize'] = val

    @property
    def placement(self):
        if False:
            print('Hello World!')
        '\n        Sets the symbol and/or text placement\n        (mapbox.layer.layout.symbol-placement). If `placement` is\n        "point", the label is placed where the geometry is located If\n        `placement` is "line", the label is placed along the line of\n        the geometry If `placement` is "line-center", the label is\n        placed on the center of the geometry\n\n        The \'placement\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'point\', \'line\', \'line-center\']\n\n        Returns\n        -------\n        Any\n        '
        return self['placement']

    @placement.setter
    def placement(self, val):
        if False:
            print('Hello World!')
        self['placement'] = val

    @property
    def text(self):
        if False:
            return 10
        "\n        Sets the symbol text (mapbox.layer.layout.text-field).\n\n        The 'text' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['text']

    @text.setter
    def text(self, val):
        if False:
            print('Hello World!')
        self['text'] = val

    @property
    def textfont(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the icon text font (color=mapbox.layer.paint.text-color,\n        size=mapbox.layer.layout.text-size). Has an effect only when\n        `type` is set to "symbol".\n\n        The \'textfont\' property is an instance of Textfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.mapbox.layer.symbol.Textfont`\n          - A dict of string/value properties that will be passed\n            to the Textfont constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.mapbox.layer.symbol.Textfont\n        '
        return self['textfont']

    @textfont.setter
    def textfont(self, val):
        if False:
            i = 10
            return i + 15
        self['textfont'] = val

    @property
    def textposition(self):
        if False:
            print('Hello World!')
        "\n        Sets the positions of the `text` elements with respects to the\n        (x,y) coordinates.\n\n        The 'textposition' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['top left', 'top center', 'top right', 'middle left',\n                'middle center', 'middle right', 'bottom left', 'bottom\n                center', 'bottom right']\n\n        Returns\n        -------\n        Any\n        "
        return self['textposition']

    @textposition.setter
    def textposition(self, val):
        if False:
            while True:
                i = 10
        self['textposition'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        icon\n            Sets the symbol icon image (mapbox.layer.layout.icon-\n            image). Full list: https://www.mapbox.com/maki-icons/\n        iconsize\n            Sets the symbol icon size (mapbox.layer.layout.icon-\n            size). Has an effect only when `type` is set to\n            "symbol".\n        placement\n            Sets the symbol and/or text placement\n            (mapbox.layer.layout.symbol-placement). If `placement`\n            is "point", the label is placed where the geometry is\n            located If `placement` is "line", the label is placed\n            along the line of the geometry If `placement` is "line-\n            center", the label is placed on the center of the\n            geometry\n        text\n            Sets the symbol text (mapbox.layer.layout.text-field).\n        textfont\n            Sets the icon text font (color=mapbox.layer.paint.text-\n            color, size=mapbox.layer.layout.text-size). Has an\n            effect only when `type` is set to "symbol".\n        textposition\n            Sets the positions of the `text` elements with respects\n            to the (x,y) coordinates.\n        '

    def __init__(self, arg=None, icon=None, iconsize=None, placement=None, text=None, textfont=None, textposition=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Symbol object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.mapbox.layer.Symbol`\n        icon\n            Sets the symbol icon image (mapbox.layer.layout.icon-\n            image). Full list: https://www.mapbox.com/maki-icons/\n        iconsize\n            Sets the symbol icon size (mapbox.layer.layout.icon-\n            size). Has an effect only when `type` is set to\n            "symbol".\n        placement\n            Sets the symbol and/or text placement\n            (mapbox.layer.layout.symbol-placement). If `placement`\n            is "point", the label is placed where the geometry is\n            located If `placement` is "line", the label is placed\n            along the line of the geometry If `placement` is "line-\n            center", the label is placed on the center of the\n            geometry\n        text\n            Sets the symbol text (mapbox.layer.layout.text-field).\n        textfont\n            Sets the icon text font (color=mapbox.layer.paint.text-\n            color, size=mapbox.layer.layout.text-size). Has an\n            effect only when `type` is set to "symbol".\n        textposition\n            Sets the positions of the `text` elements with respects\n            to the (x,y) coordinates.\n\n        Returns\n        -------\n        Symbol\n        '
        super(Symbol, self).__init__('symbol')
        if '_parent' in kwargs:
            self._parent = kwargs['_parent']
            return
        if arg is None:
            arg = {}
        elif isinstance(arg, self.__class__):
            arg = arg.to_plotly_json()
        elif isinstance(arg, dict):
            arg = _copy.copy(arg)
        else:
            raise ValueError('The first argument to the plotly.graph_objs.layout.mapbox.layer.Symbol\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.mapbox.layer.Symbol`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('icon', None)
        _v = icon if icon is not None else _v
        if _v is not None:
            self['icon'] = _v
        _v = arg.pop('iconsize', None)
        _v = iconsize if iconsize is not None else _v
        if _v is not None:
            self['iconsize'] = _v
        _v = arg.pop('placement', None)
        _v = placement if placement is not None else _v
        if _v is not None:
            self['placement'] = _v
        _v = arg.pop('text', None)
        _v = text if text is not None else _v
        if _v is not None:
            self['text'] = _v
        _v = arg.pop('textfont', None)
        _v = textfont if textfont is not None else _v
        if _v is not None:
            self['textfont'] = _v
        _v = arg.pop('textposition', None)
        _v = textposition if textposition is not None else _v
        if _v is not None:
            self['textposition'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
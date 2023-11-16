from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Outsidetextfont(_BaseTraceHierarchyType):
    _parent_path_str = 'bar'
    _path_str = 'bar.outsidetextfont'
    _valid_props = {'color', 'colorsrc', 'family', 'familysrc', 'size', 'sizesrc'}

    @property
    def color(self):
        if False:
            while True:
                i = 10
        "\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n          - A list or array of any of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            return 10
        self['color'] = val

    @property
    def colorsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `color`.\n\n        The 'colorsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['colorsrc']

    @colorsrc.setter
    def colorsrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['colorsrc'] = val

    @property
    def family(self):
        if False:
            while True:
                i = 10
        '\n        HTML font family - the typeface that will be applied by the web\n        browser. The web browser will only be able to apply a font if\n        it is available on the system which it operates. Provide\n        multiple font families, separated by commas, to indicate the\n        preference in which to apply fonts if they aren\'t available on\n        the system. The Chart Studio Cloud (at https://chart-\n        studio.plotly.com or on-premise) generates images on a server,\n        where only a select number of fonts are installed and\n        supported. These include "Arial", "Balto", "Courier New",\n        "Droid Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas\n        One", "Old Standard TT", "Open Sans", "Overpass", "PT Sans\n        Narrow", "Raleway", "Times New Roman".\n\n        The \'family\' property is a string and must be specified as:\n          - A non-empty string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['family']

    @family.setter
    def family(self, val):
        if False:
            return 10
        self['family'] = val

    @property
    def familysrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `family`.\n\n        The 'familysrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['familysrc']

    @familysrc.setter
    def familysrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['familysrc'] = val

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'size' property is a number and may be specified as:\n          - An int or float in the interval [1, inf]\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        int|float|numpy.ndarray\n        "
        return self['size']

    @size.setter
    def size(self, val):
        if False:
            while True:
                i = 10
        self['size'] = val

    @property
    def sizesrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `size`.\n\n        The 'sizesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['sizesrc']

    @sizesrc.setter
    def sizesrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['sizesrc'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        color\n\n        colorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `color`.\n        family\n            HTML font family - the typeface that will be applied by\n            the web browser. The web browser will only be able to\n            apply a font if it is available on the system which it\n            operates. Provide multiple font families, separated by\n            commas, to indicate the preference in which to apply\n            fonts if they aren\'t available on the system. The Chart\n            Studio Cloud (at https://chart-studio.plotly.com or on-\n            premise) generates images on a server, where only a\n            select number of fonts are installed and supported.\n            These include "Arial", "Balto", "Courier New", "Droid\n            Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas\n            One", "Old Standard TT", "Open Sans", "Overpass", "PT\n            Sans Narrow", "Raleway", "Times New Roman".\n        familysrc\n            Sets the source reference on Chart Studio Cloud for\n            `family`.\n        size\n\n        sizesrc\n            Sets the source reference on Chart Studio Cloud for\n            `size`.\n        '

    def __init__(self, arg=None, color=None, colorsrc=None, family=None, familysrc=None, size=None, sizesrc=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Outsidetextfont object\n\n        Sets the font used for `text` lying outside the bar.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.bar.Outsidetextfont`\n        color\n\n        colorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `color`.\n        family\n            HTML font family - the typeface that will be applied by\n            the web browser. The web browser will only be able to\n            apply a font if it is available on the system which it\n            operates. Provide multiple font families, separated by\n            commas, to indicate the preference in which to apply\n            fonts if they aren\'t available on the system. The Chart\n            Studio Cloud (at https://chart-studio.plotly.com or on-\n            premise) generates images on a server, where only a\n            select number of fonts are installed and supported.\n            These include "Arial", "Balto", "Courier New", "Droid\n            Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas\n            One", "Old Standard TT", "Open Sans", "Overpass", "PT\n            Sans Narrow", "Raleway", "Times New Roman".\n        familysrc\n            Sets the source reference on Chart Studio Cloud for\n            `family`.\n        size\n\n        sizesrc\n            Sets the source reference on Chart Studio Cloud for\n            `size`.\n\n        Returns\n        -------\n        Outsidetextfont\n        '
        super(Outsidetextfont, self).__init__('outsidetextfont')
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
            raise ValueError('The first argument to the plotly.graph_objs.bar.Outsidetextfont\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.bar.Outsidetextfont`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('colorsrc', None)
        _v = colorsrc if colorsrc is not None else _v
        if _v is not None:
            self['colorsrc'] = _v
        _v = arg.pop('family', None)
        _v = family if family is not None else _v
        if _v is not None:
            self['family'] = _v
        _v = arg.pop('familysrc', None)
        _v = familysrc if familysrc is not None else _v
        if _v is not None:
            self['familysrc'] = _v
        _v = arg.pop('size', None)
        _v = size if size is not None else _v
        if _v is not None:
            self['size'] = _v
        _v = arg.pop('sizesrc', None)
        _v = sizesrc if sizesrc is not None else _v
        if _v is not None:
            self['sizesrc'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
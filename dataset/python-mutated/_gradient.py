from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Gradient(_BaseTraceHierarchyType):
    _parent_path_str = 'scatterpolar.marker'
    _path_str = 'scatterpolar.marker.gradient'
    _valid_props = {'color', 'colorsrc', 'type', 'typesrc'}

    @property
    def color(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the final color of the gradient fill: the center color for\n        radial, the right for horizontal, or the bottom for vertical.\n\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n          - A list or array of any of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            return 10
        self['color'] = val

    @property
    def colorsrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `color`.\n\n        The 'colorsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['colorsrc']

    @colorsrc.setter
    def colorsrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['colorsrc'] = val

    @property
    def type(self):
        if False:
            print('Hello World!')
        "\n        Sets the type of gradient used to fill the markers\n\n        The 'type' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['radial', 'horizontal', 'vertical', 'none']\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['type']

    @type.setter
    def type(self, val):
        if False:
            i = 10
            return i + 15
        self['type'] = val

    @property
    def typesrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `type`.\n\n        The 'typesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['typesrc']

    @typesrc.setter
    def typesrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['typesrc'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        color\n            Sets the final color of the gradient fill: the center\n            color for radial, the right for horizontal, or the\n            bottom for vertical.\n        colorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `color`.\n        type\n            Sets the type of gradient used to fill the markers\n        typesrc\n            Sets the source reference on Chart Studio Cloud for\n            `type`.\n        '

    def __init__(self, arg=None, color=None, colorsrc=None, type=None, typesrc=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Gradient object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.scatterpolar.marker.Gradient`\n        color\n            Sets the final color of the gradient fill: the center\n            color for radial, the right for horizontal, or the\n            bottom for vertical.\n        colorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `color`.\n        type\n            Sets the type of gradient used to fill the markers\n        typesrc\n            Sets the source reference on Chart Studio Cloud for\n            `type`.\n\n        Returns\n        -------\n        Gradient\n        '
        super(Gradient, self).__init__('gradient')
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
            raise ValueError('The first argument to the plotly.graph_objs.scatterpolar.marker.Gradient\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.scatterpolar.marker.Gradient`')
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
        _v = arg.pop('type', None)
        _v = type if type is not None else _v
        if _v is not None:
            self['type'] = _v
        _v = arg.pop('typesrc', None)
        _v = typesrc if typesrc is not None else _v
        if _v is not None:
            self['typesrc'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
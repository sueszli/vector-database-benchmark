from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Lataxis(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.geo'
    _path_str = 'layout.geo.lataxis'
    _valid_props = {'dtick', 'gridcolor', 'griddash', 'gridwidth', 'range', 'showgrid', 'tick0'}

    @property
    def dtick(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the graticule's longitude/latitude tick step.\n\n        The 'dtick' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['dtick']

    @dtick.setter
    def dtick(self, val):
        if False:
            i = 10
            return i + 15
        self['dtick'] = val

    @property
    def gridcolor(self):
        if False:
            print('Hello World!')
        "\n        Sets the graticule's stroke color.\n\n        The 'gridcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['gridcolor']

    @gridcolor.setter
    def gridcolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['gridcolor'] = val

    @property
    def griddash(self):
        if False:
            while True:
                i = 10
        '\n        Sets the dash style of lines. Set to a dash type string\n        ("solid", "dot", "dash", "longdash", "dashdot", or\n        "longdashdot") or a dash length list in px (eg\n        "5px,10px,2px,2px").\n\n        The \'griddash\' property is an enumeration that may be specified as:\n          - One of the following dash styles:\n                [\'solid\', \'dot\', \'dash\', \'longdash\', \'dashdot\', \'longdashdot\']\n          - A string containing a dash length list in pixels or percentages\n                (e.g. \'5px 10px 2px 2px\', \'5, 10, 2, 2\', \'10% 20% 40%\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['griddash']

    @griddash.setter
    def griddash(self, val):
        if False:
            return 10
        self['griddash'] = val

    @property
    def gridwidth(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the graticule's stroke width (in px).\n\n        The 'gridwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['gridwidth']

    @gridwidth.setter
    def gridwidth(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['gridwidth'] = val

    @property
    def range(self):
        if False:
            i = 10
            return i + 15
        "\n            Sets the range of this axis (in degrees), sets the map's\n            clipped coordinates.\n\n            The 'range' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The 'range[0]' property is a number and may be specified as:\n              - An int or float\n        (1) The 'range[1]' property is a number and may be specified as:\n              - An int or float\n\n            Returns\n            -------\n            list\n        "
        return self['range']

    @range.setter
    def range(self, val):
        if False:
            while True:
                i = 10
        self['range'] = val

    @property
    def showgrid(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets whether or not graticule are shown on the map.\n\n        The 'showgrid' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showgrid']

    @showgrid.setter
    def showgrid(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showgrid'] = val

    @property
    def tick0(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the graticule's starting tick longitude/latitude.\n\n        The 'tick0' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['tick0']

    @tick0.setter
    def tick0(self, val):
        if False:
            i = 10
            return i + 15
        self['tick0'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        dtick\n            Sets the graticule\'s longitude/latitude tick step.\n        gridcolor\n            Sets the graticule\'s stroke color.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the graticule\'s stroke width (in px).\n        range\n            Sets the range of this axis (in degrees), sets the\n            map\'s clipped coordinates.\n        showgrid\n            Sets whether or not graticule are shown on the map.\n        tick0\n            Sets the graticule\'s starting tick longitude/latitude.\n        '

    def __init__(self, arg=None, dtick=None, gridcolor=None, griddash=None, gridwidth=None, range=None, showgrid=None, tick0=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Lataxis object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.geo.Lataxis`\n        dtick\n            Sets the graticule\'s longitude/latitude tick step.\n        gridcolor\n            Sets the graticule\'s stroke color.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the graticule\'s stroke width (in px).\n        range\n            Sets the range of this axis (in degrees), sets the\n            map\'s clipped coordinates.\n        showgrid\n            Sets whether or not graticule are shown on the map.\n        tick0\n            Sets the graticule\'s starting tick longitude/latitude.\n\n        Returns\n        -------\n        Lataxis\n        '
        super(Lataxis, self).__init__('lataxis')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.geo.Lataxis\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.geo.Lataxis`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('dtick', None)
        _v = dtick if dtick is not None else _v
        if _v is not None:
            self['dtick'] = _v
        _v = arg.pop('gridcolor', None)
        _v = gridcolor if gridcolor is not None else _v
        if _v is not None:
            self['gridcolor'] = _v
        _v = arg.pop('griddash', None)
        _v = griddash if griddash is not None else _v
        if _v is not None:
            self['griddash'] = _v
        _v = arg.pop('gridwidth', None)
        _v = gridwidth if gridwidth is not None else _v
        if _v is not None:
            self['gridwidth'] = _v
        _v = arg.pop('range', None)
        _v = range if range is not None else _v
        if _v is not None:
            self['range'] = _v
        _v = arg.pop('showgrid', None)
        _v = showgrid if showgrid is not None else _v
        if _v is not None:
            self['showgrid'] = _v
        _v = arg.pop('tick0', None)
        _v = tick0 if tick0 is not None else _v
        if _v is not None:
            self['tick0'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
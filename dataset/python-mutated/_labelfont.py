from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Labelfont(_BaseTraceHierarchyType):
    _parent_path_str = 'histogram2dcontour.contours'
    _path_str = 'histogram2dcontour.contours.labelfont'
    _valid_props = {'color', 'family', 'size'}

    @property
    def color(self):
        if False:
            print('Hello World!')
        "\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            i = 10
            return i + 15
        self['color'] = val

    @property
    def family(self):
        if False:
            i = 10
            return i + 15
        '\n        HTML font family - the typeface that will be applied by the web\n        browser. The web browser will only be able to apply a font if\n        it is available on the system which it operates. Provide\n        multiple font families, separated by commas, to indicate the\n        preference in which to apply fonts if they aren\'t available on\n        the system. The Chart Studio Cloud (at https://chart-\n        studio.plotly.com or on-premise) generates images on a server,\n        where only a select number of fonts are installed and\n        supported. These include "Arial", "Balto", "Courier New",\n        "Droid Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas\n        One", "Old Standard TT", "Open Sans", "Overpass", "PT Sans\n        Narrow", "Raleway", "Times New Roman".\n\n        The \'family\' property is a string and must be specified as:\n          - A non-empty string\n\n        Returns\n        -------\n        str\n        '
        return self['family']

    @family.setter
    def family(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['family'] = val

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'size' property is a number and may be specified as:\n          - An int or float in the interval [1, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['size']

    @size.setter
    def size(self, val):
        if False:
            print('Hello World!')
        self['size'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        color\n\n        family\n            HTML font family - the typeface that will be applied by\n            the web browser. The web browser will only be able to\n            apply a font if it is available on the system which it\n            operates. Provide multiple font families, separated by\n            commas, to indicate the preference in which to apply\n            fonts if they aren\'t available on the system. The Chart\n            Studio Cloud (at https://chart-studio.plotly.com or on-\n            premise) generates images on a server, where only a\n            select number of fonts are installed and supported.\n            These include "Arial", "Balto", "Courier New", "Droid\n            Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas\n            One", "Old Standard TT", "Open Sans", "Overpass", "PT\n            Sans Narrow", "Raleway", "Times New Roman".\n        size\n\n        '

    def __init__(self, arg=None, color=None, family=None, size=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Labelfont object\n\n        Sets the font used for labeling the contour levels. The default\n        color comes from the lines, if shown. The default family and\n        size come from `layout.font`.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.histogram2dcon\n            tour.contours.Labelfont`\n        color\n\n        family\n            HTML font family - the typeface that will be applied by\n            the web browser. The web browser will only be able to\n            apply a font if it is available on the system which it\n            operates. Provide multiple font families, separated by\n            commas, to indicate the preference in which to apply\n            fonts if they aren\'t available on the system. The Chart\n            Studio Cloud (at https://chart-studio.plotly.com or on-\n            premise) generates images on a server, where only a\n            select number of fonts are installed and supported.\n            These include "Arial", "Balto", "Courier New", "Droid\n            Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas\n            One", "Old Standard TT", "Open Sans", "Overpass", "PT\n            Sans Narrow", "Raleway", "Times New Roman".\n        size\n\n\n        Returns\n        -------\n        Labelfont\n        '
        super(Labelfont, self).__init__('labelfont')
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
            raise ValueError('The first argument to the plotly.graph_objs.histogram2dcontour.contours.Labelfont\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.histogram2dcontour.contours.Labelfont`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('family', None)
        _v = family if family is not None else _v
        if _v is not None:
            self['family'] = _v
        _v = arg.pop('size', None)
        _v = size if size is not None else _v
        if _v is not None:
            self['size'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
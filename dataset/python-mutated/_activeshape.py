from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Activeshape(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.activeshape'
    _valid_props = {'fillcolor', 'opacity'}

    @property
    def fillcolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the color filling the active shape' interior.\n\n        The 'fillcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['fillcolor']

    @fillcolor.setter
    def fillcolor(self, val):
        if False:
            while True:
                i = 10
        self['fillcolor'] = val

    @property
    def opacity(self):
        if False:
            print('Hello World!')
        "\n        Sets the opacity of the active shape.\n\n        The 'opacity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            print('Hello World!')
        self['opacity'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return "        fillcolor\n            Sets the color filling the active shape' interior.\n        opacity\n            Sets the opacity of the active shape.\n        "

    def __init__(self, arg=None, fillcolor=None, opacity=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a new Activeshape object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Activeshape`\n        fillcolor\n            Sets the color filling the active shape' interior.\n        opacity\n            Sets the opacity of the active shape.\n\n        Returns\n        -------\n        Activeshape\n        "
        super(Activeshape, self).__init__('activeshape')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Activeshape\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Activeshape`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('fillcolor', None)
        _v = fillcolor if fillcolor is not None else _v
        if _v is not None:
            self['fillcolor'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
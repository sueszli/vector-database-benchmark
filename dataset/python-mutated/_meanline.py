from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Meanline(_BaseTraceHierarchyType):
    _parent_path_str = 'violin'
    _path_str = 'violin.meanline'
    _valid_props = {'color', 'visible', 'width'}

    @property
    def color(self):
        if False:
            while True:
                i = 10
        "\n        Sets the mean line color.\n\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['color'] = val

    @property
    def visible(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines if a line corresponding to the sample's mean is\n        shown inside the violins. If `box.visible` is turned on, the\n        mean line is drawn inside the inner box. Otherwise, the mean\n        line is drawn from one side of the violin to other.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            while True:
                i = 10
        self['visible'] = val

    @property
    def width(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the mean line width.\n\n        The 'width' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['width']

    @width.setter
    def width(self, val):
        if False:
            while True:
                i = 10
        self['width'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return "        color\n            Sets the mean line color.\n        visible\n            Determines if a line corresponding to the sample's mean\n            is shown inside the violins. If `box.visible` is turned\n            on, the mean line is drawn inside the inner box.\n            Otherwise, the mean line is drawn from one side of the\n            violin to other.\n        width\n            Sets the mean line width.\n        "

    def __init__(self, arg=None, color=None, visible=None, width=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Construct a new Meanline object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.violin.Meanline`\n        color\n            Sets the mean line color.\n        visible\n            Determines if a line corresponding to the sample's mean\n            is shown inside the violins. If `box.visible` is turned\n            on, the mean line is drawn inside the inner box.\n            Otherwise, the mean line is drawn from one side of the\n            violin to other.\n        width\n            Sets the mean line width.\n\n        Returns\n        -------\n        Meanline\n        "
        super(Meanline, self).__init__('meanline')
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
            raise ValueError('The first argument to the plotly.graph_objs.violin.Meanline\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.violin.Meanline`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        _v = arg.pop('width', None)
        _v = width if width is not None else _v
        if _v is not None:
            self['width'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
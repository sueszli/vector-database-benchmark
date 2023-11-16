from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Fill(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.mapbox.layer'
    _path_str = 'layout.mapbox.layer.fill'
    _valid_props = {'outlinecolor'}

    @property
    def outlinecolor(self):
        if False:
            return 10
        '\n        Sets the fill outline color (mapbox.layer.paint.fill-outline-\n        color). Has an effect only when `type` is set to "fill".\n\n        The \'outlinecolor\' property is a color and may be specified as:\n          - A hex string (e.g. \'#ff0000\')\n          - An rgb/rgba string (e.g. \'rgb(255,0,0)\')\n          - An hsl/hsla string (e.g. \'hsl(0,100%,50%)\')\n          - An hsv/hsva string (e.g. \'hsv(0,100%,100%)\')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        '
        return self['outlinecolor']

    @outlinecolor.setter
    def outlinecolor(self, val):
        if False:
            print('Hello World!')
        self['outlinecolor'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        outlinecolor\n            Sets the fill outline color (mapbox.layer.paint.fill-\n            outline-color). Has an effect only when `type` is set\n            to "fill".\n        '

    def __init__(self, arg=None, outlinecolor=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Fill object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.mapbox.layer.Fill`\n        outlinecolor\n            Sets the fill outline color (mapbox.layer.paint.fill-\n            outline-color). Has an effect only when `type` is set\n            to "fill".\n\n        Returns\n        -------\n        Fill\n        '
        super(Fill, self).__init__('fill')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.mapbox.layer.Fill\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.mapbox.layer.Fill`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('outlinecolor', None)
        _v = outlinecolor if outlinecolor is not None else _v
        if _v is not None:
            self['outlinecolor'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
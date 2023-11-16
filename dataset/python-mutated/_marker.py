from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Marker(_BaseTraceHierarchyType):
    _parent_path_str = 'pointcloud'
    _path_str = 'pointcloud.marker'
    _valid_props = {'blend', 'border', 'color', 'opacity', 'sizemax', 'sizemin'}

    @property
    def blend(self):
        if False:
            return 10
        "\n        Determines if colors are blended together for a translucency\n        effect in case `opacity` is specified as a value less then `1`.\n        Setting `blend` to `true` reduces zoom/pan speed if used with\n        large numbers of points.\n\n        The 'blend' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['blend']

    @blend.setter
    def blend(self, val):
        if False:
            return 10
        self['blend'] = val

    @property
    def border(self):
        if False:
            while True:
                i = 10
        "\n        The 'border' property is an instance of Border\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.pointcloud.marker.Border`\n          - A dict of string/value properties that will be passed\n            to the Border constructor\n\n            Supported dict properties:\n\n                arearatio\n                    Specifies what fraction of the marker area is\n                    covered with the border.\n                color\n                    Sets the stroke color. It accepts a specific\n                    color. If the color is not fully opaque and\n                    there are hundreds of thousands of points, it\n                    may cause slower zooming and panning.\n\n        Returns\n        -------\n        plotly.graph_objs.pointcloud.marker.Border\n        "
        return self['border']

    @border.setter
    def border(self, val):
        if False:
            print('Hello World!')
        self['border'] = val

    @property
    def color(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the marker fill color. It accepts a specific color. If the\n        color is not fully opaque and there are hundreds of thousands\n        of points, it may cause slower zooming and panning.\n\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            while True:
                i = 10
        self['color'] = val

    @property
    def opacity(self):
        if False:
            print('Hello World!')
        "\n        Sets the marker opacity. The default value is `1` (fully\n        opaque). If the markers are not fully opaque and there are\n        hundreds of thousands of points, it may cause slower zooming\n        and panning. Opacity fades the color even if `blend` is left on\n        `false` even if there is no translucency effect in that case.\n\n        The 'opacity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            return 10
        self['opacity'] = val

    @property
    def sizemax(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the maximum size (in px) of the rendered marker points.\n        Effective when the `pointcloud` shows only few points.\n\n        The 'sizemax' property is a number and may be specified as:\n          - An int or float in the interval [0.1, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['sizemax']

    @sizemax.setter
    def sizemax(self, val):
        if False:
            return 10
        self['sizemax'] = val

    @property
    def sizemin(self):
        if False:
            while True:
                i = 10
        "\n        Sets the minimum size (in px) of the rendered marker points,\n        effective when the `pointcloud` shows a million or more points.\n\n        The 'sizemin' property is a number and may be specified as:\n          - An int or float in the interval [0.1, 2]\n\n        Returns\n        -------\n        int|float\n        "
        return self['sizemin']

    @sizemin.setter
    def sizemin(self, val):
        if False:
            return 10
        self['sizemin'] = val

    @property
    def _prop_descriptions(self):
        if False:
            i = 10
            return i + 15
        return '        blend\n            Determines if colors are blended together for a\n            translucency effect in case `opacity` is specified as a\n            value less then `1`. Setting `blend` to `true` reduces\n            zoom/pan speed if used with large numbers of points.\n        border\n            :class:`plotly.graph_objects.pointcloud.marker.Border`\n            instance or dict with compatible properties\n        color\n            Sets the marker fill color. It accepts a specific\n            color. If the color is not fully opaque and there are\n            hundreds of thousands of points, it may cause slower\n            zooming and panning.\n        opacity\n            Sets the marker opacity. The default value is `1`\n            (fully opaque). If the markers are not fully opaque and\n            there are hundreds of thousands of points, it may cause\n            slower zooming and panning. Opacity fades the color\n            even if `blend` is left on `false` even if there is no\n            translucency effect in that case.\n        sizemax\n            Sets the maximum size (in px) of the rendered marker\n            points. Effective when the `pointcloud` shows only few\n            points.\n        sizemin\n            Sets the minimum size (in px) of the rendered marker\n            points, effective when the `pointcloud` shows a million\n            or more points.\n        '

    def __init__(self, arg=None, blend=None, border=None, color=None, opacity=None, sizemax=None, sizemin=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Marker object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.pointcloud.Marker`\n        blend\n            Determines if colors are blended together for a\n            translucency effect in case `opacity` is specified as a\n            value less then `1`. Setting `blend` to `true` reduces\n            zoom/pan speed if used with large numbers of points.\n        border\n            :class:`plotly.graph_objects.pointcloud.marker.Border`\n            instance or dict with compatible properties\n        color\n            Sets the marker fill color. It accepts a specific\n            color. If the color is not fully opaque and there are\n            hundreds of thousands of points, it may cause slower\n            zooming and panning.\n        opacity\n            Sets the marker opacity. The default value is `1`\n            (fully opaque). If the markers are not fully opaque and\n            there are hundreds of thousands of points, it may cause\n            slower zooming and panning. Opacity fades the color\n            even if `blend` is left on `false` even if there is no\n            translucency effect in that case.\n        sizemax\n            Sets the maximum size (in px) of the rendered marker\n            points. Effective when the `pointcloud` shows only few\n            points.\n        sizemin\n            Sets the minimum size (in px) of the rendered marker\n            points, effective when the `pointcloud` shows a million\n            or more points.\n\n        Returns\n        -------\n        Marker\n        '
        super(Marker, self).__init__('marker')
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
            raise ValueError('The first argument to the plotly.graph_objs.pointcloud.Marker\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.pointcloud.Marker`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('blend', None)
        _v = blend if blend is not None else _v
        if _v is not None:
            self['blend'] = _v
        _v = arg.pop('border', None)
        _v = border if border is not None else _v
        if _v is not None:
            self['border'] = _v
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        _v = arg.pop('sizemax', None)
        _v = sizemax if sizemax is not None else _v
        if _v is not None:
            self['sizemax'] = _v
        _v = arg.pop('sizemin', None)
        _v = sizemin if sizemin is not None else _v
        if _v is not None:
            self['sizemin'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
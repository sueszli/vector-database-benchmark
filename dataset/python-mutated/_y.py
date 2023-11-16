from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Y(_BaseTraceHierarchyType):
    _parent_path_str = 'surface.contours'
    _path_str = 'surface.contours.y'
    _valid_props = {'color', 'end', 'highlight', 'highlightcolor', 'highlightwidth', 'project', 'show', 'size', 'start', 'usecolormap', 'width'}

    @property
    def color(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the color of the contour lines.\n\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            print('Hello World!')
        self['color'] = val

    @property
    def end(self):
        if False:
            return 10
        "\n        Sets the end contour level value. Must be more than\n        `contours.start`\n\n        The 'end' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['end']

    @end.setter
    def end(self, val):
        if False:
            while True:
                i = 10
        self['end'] = val

    @property
    def highlight(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not contour lines about the y dimension\n        are highlighted on hover.\n\n        The 'highlight' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['highlight']

    @highlight.setter
    def highlight(self, val):
        if False:
            return 10
        self['highlight'] = val

    @property
    def highlightcolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the color of the highlighted contour lines.\n\n        The 'highlightcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['highlightcolor']

    @highlightcolor.setter
    def highlightcolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['highlightcolor'] = val

    @property
    def highlightwidth(self):
        if False:
            return 10
        "\n        Sets the width of the highlighted contour lines.\n\n        The 'highlightwidth' property is a number and may be specified as:\n          - An int or float in the interval [1, 16]\n\n        Returns\n        -------\n        int|float\n        "
        return self['highlightwidth']

    @highlightwidth.setter
    def highlightwidth(self, val):
        if False:
            i = 10
            return i + 15
        self['highlightwidth'] = val

    @property
    def project(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'project' property is an instance of Project\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.surface.contours.y.Project`\n          - A dict of string/value properties that will be passed\n            to the Project constructor\n\n            Supported dict properties:\n\n                x\n                    Determines whether or not these contour lines\n                    are projected on the x plane. If `highlight` is\n                    set to True (the default), the projected lines\n                    are shown on hover. If `show` is set to True,\n                    the projected lines are shown in permanence.\n                y\n                    Determines whether or not these contour lines\n                    are projected on the y plane. If `highlight` is\n                    set to True (the default), the projected lines\n                    are shown on hover. If `show` is set to True,\n                    the projected lines are shown in permanence.\n                z\n                    Determines whether or not these contour lines\n                    are projected on the z plane. If `highlight` is\n                    set to True (the default), the projected lines\n                    are shown on hover. If `show` is set to True,\n                    the projected lines are shown in permanence.\n\n        Returns\n        -------\n        plotly.graph_objs.surface.contours.y.Project\n        "
        return self['project']

    @project.setter
    def project(self, val):
        if False:
            while True:
                i = 10
        self['project'] = val

    @property
    def show(self):
        if False:
            return 10
        "\n        Determines whether or not contour lines about the y dimension\n        are drawn.\n\n        The 'show' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['show']

    @show.setter
    def show(self, val):
        if False:
            i = 10
            return i + 15
        self['show'] = val

    @property
    def size(self):
        if False:
            while True:
                i = 10
        "\n        Sets the step between each contour level. Must be positive.\n\n        The 'size' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['size']

    @size.setter
    def size(self, val):
        if False:
            while True:
                i = 10
        self['size'] = val

    @property
    def start(self):
        if False:
            return 10
        "\n        Sets the starting contour level value. Must be less than\n        `contours.end`\n\n        The 'start' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['start']

    @start.setter
    def start(self, val):
        if False:
            return 10
        self['start'] = val

    @property
    def usecolormap(self):
        if False:
            print('Hello World!')
        '\n        An alternate to "color". Determines whether or not the contour\n        lines are colored using the trace "colorscale".\n\n        The \'usecolormap\' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        '
        return self['usecolormap']

    @usecolormap.setter
    def usecolormap(self, val):
        if False:
            print('Hello World!')
        self['usecolormap'] = val

    @property
    def width(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the width of the contour lines.\n\n        The 'width' property is a number and may be specified as:\n          - An int or float in the interval [1, 16]\n\n        Returns\n        -------\n        int|float\n        "
        return self['width']

    @width.setter
    def width(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['width'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        color\n            Sets the color of the contour lines.\n        end\n            Sets the end contour level value. Must be more than\n            `contours.start`\n        highlight\n            Determines whether or not contour lines about the y\n            dimension are highlighted on hover.\n        highlightcolor\n            Sets the color of the highlighted contour lines.\n        highlightwidth\n            Sets the width of the highlighted contour lines.\n        project\n            :class:`plotly.graph_objects.surface.contours.y.Project\n            ` instance or dict with compatible properties\n        show\n            Determines whether or not contour lines about the y\n            dimension are drawn.\n        size\n            Sets the step between each contour level. Must be\n            positive.\n        start\n            Sets the starting contour level value. Must be less\n            than `contours.end`\n        usecolormap\n            An alternate to "color". Determines whether or not the\n            contour lines are colored using the trace "colorscale".\n        width\n            Sets the width of the contour lines.\n        '

    def __init__(self, arg=None, color=None, end=None, highlight=None, highlightcolor=None, highlightwidth=None, project=None, show=None, size=None, start=None, usecolormap=None, width=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Y object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.surface.contours.Y`\n        color\n            Sets the color of the contour lines.\n        end\n            Sets the end contour level value. Must be more than\n            `contours.start`\n        highlight\n            Determines whether or not contour lines about the y\n            dimension are highlighted on hover.\n        highlightcolor\n            Sets the color of the highlighted contour lines.\n        highlightwidth\n            Sets the width of the highlighted contour lines.\n        project\n            :class:`plotly.graph_objects.surface.contours.y.Project\n            ` instance or dict with compatible properties\n        show\n            Determines whether or not contour lines about the y\n            dimension are drawn.\n        size\n            Sets the step between each contour level. Must be\n            positive.\n        start\n            Sets the starting contour level value. Must be less\n            than `contours.end`\n        usecolormap\n            An alternate to "color". Determines whether or not the\n            contour lines are colored using the trace "colorscale".\n        width\n            Sets the width of the contour lines.\n\n        Returns\n        -------\n        Y\n        '
        super(Y, self).__init__('y')
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
            raise ValueError('The first argument to the plotly.graph_objs.surface.contours.Y\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.surface.contours.Y`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('end', None)
        _v = end if end is not None else _v
        if _v is not None:
            self['end'] = _v
        _v = arg.pop('highlight', None)
        _v = highlight if highlight is not None else _v
        if _v is not None:
            self['highlight'] = _v
        _v = arg.pop('highlightcolor', None)
        _v = highlightcolor if highlightcolor is not None else _v
        if _v is not None:
            self['highlightcolor'] = _v
        _v = arg.pop('highlightwidth', None)
        _v = highlightwidth if highlightwidth is not None else _v
        if _v is not None:
            self['highlightwidth'] = _v
        _v = arg.pop('project', None)
        _v = project if project is not None else _v
        if _v is not None:
            self['project'] = _v
        _v = arg.pop('show', None)
        _v = show if show is not None else _v
        if _v is not None:
            self['show'] = _v
        _v = arg.pop('size', None)
        _v = size if size is not None else _v
        if _v is not None:
            self['size'] = _v
        _v = arg.pop('start', None)
        _v = start if start is not None else _v
        if _v is not None:
            self['start'] = _v
        _v = arg.pop('usecolormap', None)
        _v = usecolormap if usecolormap is not None else _v
        if _v is not None:
            self['usecolormap'] = _v
        _v = arg.pop('width', None)
        _v = width if width is not None else _v
        if _v is not None:
            self['width'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
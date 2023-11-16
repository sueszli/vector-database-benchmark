from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Rangeslider(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.xaxis'
    _path_str = 'layout.xaxis.rangeslider'
    _valid_props = {'autorange', 'bgcolor', 'bordercolor', 'borderwidth', 'range', 'thickness', 'visible', 'yaxis'}

    @property
    def autorange(self):
        if False:
            return 10
        "\n        Determines whether or not the range slider range is computed in\n        relation to the input data. If `range` is provided, then\n        `autorange` is set to False.\n\n        The 'autorange' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['autorange']

    @autorange.setter
    def autorange(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['autorange'] = val

    @property
    def bgcolor(self):
        if False:
            while True:
                i = 10
        "\n        Sets the background color of the range slider.\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            print('Hello World!')
        self['bgcolor'] = val

    @property
    def bordercolor(self):
        if False:
            print('Hello World!')
        "\n        Sets the border color of the range slider.\n\n        The 'bordercolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bordercolor']

    @bordercolor.setter
    def bordercolor(self, val):
        if False:
            i = 10
            return i + 15
        self['bordercolor'] = val

    @property
    def borderwidth(self):
        if False:
            return 10
        "\n        Sets the border width of the range slider.\n\n        The 'borderwidth' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['borderwidth']

    @borderwidth.setter
    def borderwidth(self, val):
        if False:
            return 10
        self['borderwidth'] = val

    @property
    def range(self):
        if False:
            return 10
        '\n            Sets the range of the range slider. If not set, defaults to the\n            full xaxis range. If the axis `type` is "log", then you must\n            take the log of your desired range. If the axis `type` is\n            "date", it should be date strings, like date data, though Date\n            objects and unix milliseconds will be accepted and converted to\n            strings. If the axis `type` is "category", it should be\n            numbers, using the scale where each category is assigned a\n            serial number from zero in the order it appears.\n\n            The \'range\' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The \'range[0]\' property accepts values of any type\n        (1) The \'range[1]\' property accepts values of any type\n\n            Returns\n            -------\n            list\n        '
        return self['range']

    @range.setter
    def range(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['range'] = val

    @property
    def thickness(self):
        if False:
            return 10
        "\n        The height of the range slider as a fraction of the total plot\n        area height.\n\n        The 'thickness' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['thickness']

    @thickness.setter
    def thickness(self, val):
        if False:
            return 10
        self['thickness'] = val

    @property
    def visible(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not the range slider will be visible. If\n        visible, perpendicular axes will be set to `fixedrange`\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            print('Hello World!')
        self['visible'] = val

    @property
    def yaxis(self):
        if False:
            return 10
        '\n        The \'yaxis\' property is an instance of YAxis\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.xaxis.rangeslider.YAxis`\n          - A dict of string/value properties that will be passed\n            to the YAxis constructor\n\n            Supported dict properties:\n\n                range\n                    Sets the range of this axis for the\n                    rangeslider.\n                rangemode\n                    Determines whether or not the range of this\n                    axis in the rangeslider use the same value than\n                    in the main plot when zooming in/out. If\n                    "auto", the autorange will be used. If "fixed",\n                    the `range` is used. If "match", the current\n                    range of the corresponding y-axis on the main\n                    subplot is used.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.xaxis.rangeslider.YAxis\n        '
        return self['yaxis']

    @yaxis.setter
    def yaxis(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['yaxis'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        autorange\n            Determines whether or not the range slider range is\n            computed in relation to the input data. If `range` is\n            provided, then `autorange` is set to False.\n        bgcolor\n            Sets the background color of the range slider.\n        bordercolor\n            Sets the border color of the range slider.\n        borderwidth\n            Sets the border width of the range slider.\n        range\n            Sets the range of the range slider. If not set,\n            defaults to the full xaxis range. If the axis `type` is\n            "log", then you must take the log of your desired\n            range. If the axis `type` is "date", it should be date\n            strings, like date data, though Date objects and unix\n            milliseconds will be accepted and converted to strings.\n            If the axis `type` is "category", it should be numbers,\n            using the scale where each category is assigned a\n            serial number from zero in the order it appears.\n        thickness\n            The height of the range slider as a fraction of the\n            total plot area height.\n        visible\n            Determines whether or not the range slider will be\n            visible. If visible, perpendicular axes will be set to\n            `fixedrange`\n        yaxis\n            :class:`plotly.graph_objects.layout.xaxis.rangeslider.Y\n            Axis` instance or dict with compatible properties\n        '

    def __init__(self, arg=None, autorange=None, bgcolor=None, bordercolor=None, borderwidth=None, range=None, thickness=None, visible=None, yaxis=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Rangeslider object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.xaxis.Rangeslider`\n        autorange\n            Determines whether or not the range slider range is\n            computed in relation to the input data. If `range` is\n            provided, then `autorange` is set to False.\n        bgcolor\n            Sets the background color of the range slider.\n        bordercolor\n            Sets the border color of the range slider.\n        borderwidth\n            Sets the border width of the range slider.\n        range\n            Sets the range of the range slider. If not set,\n            defaults to the full xaxis range. If the axis `type` is\n            "log", then you must take the log of your desired\n            range. If the axis `type` is "date", it should be date\n            strings, like date data, though Date objects and unix\n            milliseconds will be accepted and converted to strings.\n            If the axis `type` is "category", it should be numbers,\n            using the scale where each category is assigned a\n            serial number from zero in the order it appears.\n        thickness\n            The height of the range slider as a fraction of the\n            total plot area height.\n        visible\n            Determines whether or not the range slider will be\n            visible. If visible, perpendicular axes will be set to\n            `fixedrange`\n        yaxis\n            :class:`plotly.graph_objects.layout.xaxis.rangeslider.Y\n            Axis` instance or dict with compatible properties\n\n        Returns\n        -------\n        Rangeslider\n        '
        super(Rangeslider, self).__init__('rangeslider')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.xaxis.Rangeslider\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.xaxis.Rangeslider`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('autorange', None)
        _v = autorange if autorange is not None else _v
        if _v is not None:
            self['autorange'] = _v
        _v = arg.pop('bgcolor', None)
        _v = bgcolor if bgcolor is not None else _v
        if _v is not None:
            self['bgcolor'] = _v
        _v = arg.pop('bordercolor', None)
        _v = bordercolor if bordercolor is not None else _v
        if _v is not None:
            self['bordercolor'] = _v
        _v = arg.pop('borderwidth', None)
        _v = borderwidth if borderwidth is not None else _v
        if _v is not None:
            self['borderwidth'] = _v
        _v = arg.pop('range', None)
        _v = range if range is not None else _v
        if _v is not None:
            self['range'] = _v
        _v = arg.pop('thickness', None)
        _v = thickness if thickness is not None else _v
        if _v is not None:
            self['thickness'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        _v = arg.pop('yaxis', None)
        _v = yaxis if yaxis is not None else _v
        if _v is not None:
            self['yaxis'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
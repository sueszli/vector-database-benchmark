from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Minor(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.xaxis'
    _path_str = 'layout.xaxis.minor'
    _valid_props = {'dtick', 'gridcolor', 'griddash', 'gridwidth', 'nticks', 'showgrid', 'tick0', 'tickcolor', 'ticklen', 'tickmode', 'ticks', 'tickvals', 'tickvalssrc', 'tickwidth'}

    @property
    def dtick(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the step in-between ticks on this axis. Use with `tick0`.\n        Must be a positive number, or special strings available to\n        "log" and "date" axes. If the axis `type` is "log", then ticks\n        are set every 10^(n*dtick) where n is the tick number. For\n        example, to set a tick mark at 1, 10, 100, 1000, ... set dtick\n        to 1. To set tick marks at 1, 100, 10000, ... set dtick to 2.\n        To set tick marks at 1, 5, 25, 125, 625, 3125, ... set dtick to\n        log_10(5), or 0.69897000433. "log" has several special values;\n        "L<f>", where `f` is a positive number, gives ticks linearly\n        spaced in value (but not position). For example `tick0` = 0.1,\n        `dtick` = "L0.5" will put ticks at 0.1, 0.6, 1.1, 1.6 etc. To\n        show powers of 10 plus small digits between, use "D1" (all\n        digits) or "D2" (only 2 and 5). `tick0` is ignored for "D1" and\n        "D2". If the axis `type` is "date", then you must convert the\n        time to milliseconds. For example, to set the interval between\n        ticks to one day, set `dtick` to 86400000.0. "date" also has\n        special values "M<n>" gives ticks spaced by a number of months.\n        `n` must be a positive integer. To set ticks on the 15th of\n        every third month, set `tick0` to "2000-01-15" and `dtick` to\n        "M3". To set ticks every 4 years, set `dtick` to "M48"\n\n        The \'dtick\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['dtick']

    @dtick.setter
    def dtick(self, val):
        if False:
            return 10
        self['dtick'] = val

    @property
    def gridcolor(self):
        if False:
            while True:
                i = 10
        "\n        Sets the color of the grid lines.\n\n        The 'gridcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['gridcolor']

    @gridcolor.setter
    def gridcolor(self, val):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
        "\n        Sets the width (in px) of the grid lines.\n\n        The 'gridwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['gridwidth']

    @gridwidth.setter
    def gridwidth(self, val):
        if False:
            i = 10
            return i + 15
        self['gridwidth'] = val

    @property
    def nticks(self):
        if False:
            while True:
                i = 10
        '\n        Specifies the maximum number of ticks for the particular axis.\n        The actual number of ticks will be chosen automatically to be\n        less than or equal to `nticks`. Has an effect only if\n        `tickmode` is set to "auto".\n\n        The \'nticks\' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        '
        return self['nticks']

    @nticks.setter
    def nticks(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['nticks'] = val

    @property
    def showgrid(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether or not grid lines are drawn. If True, the\n        grid lines are drawn at every tick mark.\n\n        The 'showgrid' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showgrid']

    @showgrid.setter
    def showgrid(self, val):
        if False:
            print('Hello World!')
        self['showgrid'] = val

    @property
    def tick0(self):
        if False:
            print('Hello World!')
        '\n        Sets the placement of the first tick on this axis. Use with\n        `dtick`. If the axis `type` is "log", then you must take the\n        log of your starting tick (e.g. to set the starting tick to\n        100, set the `tick0` to 2) except when `dtick`=*L<f>* (see\n        `dtick` for more info). If the axis `type` is "date", it should\n        be a date string, like date data. If the axis `type` is\n        "category", it should be a number, using the scale where each\n        category is assigned a serial number from zero in the order it\n        appears.\n\n        The \'tick0\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['tick0']

    @tick0.setter
    def tick0(self, val):
        if False:
            while True:
                i = 10
        self['tick0'] = val

    @property
    def tickcolor(self):
        if False:
            return 10
        "\n        Sets the tick color.\n\n        The 'tickcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['tickcolor']

    @tickcolor.setter
    def tickcolor(self, val):
        if False:
            print('Hello World!')
        self['tickcolor'] = val

    @property
    def ticklen(self):
        if False:
            return 10
        "\n        Sets the tick length (in px).\n\n        The 'ticklen' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['ticklen']

    @ticklen.setter
    def ticklen(self, val):
        if False:
            return 10
        self['ticklen'] = val

    @property
    def tickmode(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the tick mode for this axis. If "auto", the number of\n        ticks is set via `nticks`. If "linear", the placement of the\n        ticks is determined by a starting position `tick0` and a tick\n        step `dtick` ("linear" is the default value if `tick0` and\n        `dtick` are provided). If "array", the placement of the ticks\n        is set via `tickvals` and the tick text is `ticktext`. ("array"\n        is the default value if `tickvals` is provided).\n\n        The \'tickmode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'linear\', \'array\']\n\n        Returns\n        -------\n        Any\n        '
        return self['tickmode']

    @tickmode.setter
    def tickmode(self, val):
        if False:
            return 10
        self['tickmode'] = val

    @property
    def ticks(self):
        if False:
            print('Hello World!')
        '\n        Determines whether ticks are drawn or not. If "", this axis\'\n        ticks are not drawn. If "outside" ("inside"), this axis\' are\n        drawn outside (inside) the axis lines.\n\n        The \'ticks\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'outside\', \'inside\', \'\']\n\n        Returns\n        -------\n        Any\n        '
        return self['ticks']

    @ticks.setter
    def ticks(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['ticks'] = val

    @property
    def tickvals(self):
        if False:
            return 10
        '\n        Sets the values at which ticks on this axis appear. Only has an\n        effect if `tickmode` is set to "array". Used with `ticktext`.\n\n        The \'tickvals\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['tickvals']

    @tickvals.setter
    def tickvals(self, val):
        if False:
            while True:
                i = 10
        self['tickvals'] = val

    @property
    def tickvalssrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `tickvals`.\n\n        The 'tickvalssrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['tickvalssrc']

    @tickvalssrc.setter
    def tickvalssrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickvalssrc'] = val

    @property
    def tickwidth(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the tick width (in px).\n\n        The 'tickwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['tickwidth']

    @tickwidth.setter
    def tickwidth(self, val):
        if False:
            i = 10
            return i + 15
        self['tickwidth'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        dtick\n            Sets the step in-between ticks on this axis. Use with\n            `tick0`. Must be a positive number, or special strings\n            available to "log" and "date" axes. If the axis `type`\n            is "log", then ticks are set every 10^(n*dtick) where n\n            is the tick number. For example, to set a tick mark at\n            1, 10, 100, 1000, ... set dtick to 1. To set tick marks\n            at 1, 100, 10000, ... set dtick to 2. To set tick marks\n            at 1, 5, 25, 125, 625, 3125, ... set dtick to\n            log_10(5), or 0.69897000433. "log" has several special\n            values; "L<f>", where `f` is a positive number, gives\n            ticks linearly spaced in value (but not position). For\n            example `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n            at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus\n            small digits between, use "D1" (all digits) or "D2"\n            (only 2 and 5). `tick0` is ignored for "D1" and "D2".\n            If the axis `type` is "date", then you must convert the\n            time to milliseconds. For example, to set the interval\n            between ticks to one day, set `dtick` to 86400000.0.\n            "date" also has special values "M<n>" gives ticks\n            spaced by a number of months. `n` must be a positive\n            integer. To set ticks on the 15th of every third month,\n            set `tick0` to "2000-01-15" and `dtick` to "M3". To set\n            ticks every 4 years, set `dtick` to "M48"\n        gridcolor\n            Sets the color of the grid lines.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the grid lines.\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        tick0\n            Sets the placement of the first tick on this axis. Use\n            with `dtick`. If the axis `type` is "log", then you\n            must take the log of your starting tick (e.g. to set\n            the starting tick to 100, set the `tick0` to 2) except\n            when `dtick`=*L<f>* (see `dtick` for more info). If the\n            axis `type` is "date", it should be a date string, like\n            date data. If the axis `type` is "category", it should\n            be a number, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        tickcolor\n            Sets the tick color.\n        ticklen\n            Sets the tick length (in px).\n        tickmode\n            Sets the tick mode for this axis. If "auto", the number\n            of ticks is set via `nticks`. If "linear", the\n            placement of the ticks is determined by a starting\n            position `tick0` and a tick step `dtick` ("linear" is\n            the default value if `tick0` and `dtick` are provided).\n            If "array", the placement of the ticks is set via\n            `tickvals` and the tick text is `ticktext`. ("array" is\n            the default value if `tickvals` is provided).\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n        '

    def __init__(self, arg=None, dtick=None, gridcolor=None, griddash=None, gridwidth=None, nticks=None, showgrid=None, tick0=None, tickcolor=None, ticklen=None, tickmode=None, ticks=None, tickvals=None, tickvalssrc=None, tickwidth=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Minor object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.xaxis.Minor`\n        dtick\n            Sets the step in-between ticks on this axis. Use with\n            `tick0`. Must be a positive number, or special strings\n            available to "log" and "date" axes. If the axis `type`\n            is "log", then ticks are set every 10^(n*dtick) where n\n            is the tick number. For example, to set a tick mark at\n            1, 10, 100, 1000, ... set dtick to 1. To set tick marks\n            at 1, 100, 10000, ... set dtick to 2. To set tick marks\n            at 1, 5, 25, 125, 625, 3125, ... set dtick to\n            log_10(5), or 0.69897000433. "log" has several special\n            values; "L<f>", where `f` is a positive number, gives\n            ticks linearly spaced in value (but not position). For\n            example `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n            at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus\n            small digits between, use "D1" (all digits) or "D2"\n            (only 2 and 5). `tick0` is ignored for "D1" and "D2".\n            If the axis `type` is "date", then you must convert the\n            time to milliseconds. For example, to set the interval\n            between ticks to one day, set `dtick` to 86400000.0.\n            "date" also has special values "M<n>" gives ticks\n            spaced by a number of months. `n` must be a positive\n            integer. To set ticks on the 15th of every third month,\n            set `tick0` to "2000-01-15" and `dtick` to "M3". To set\n            ticks every 4 years, set `dtick` to "M48"\n        gridcolor\n            Sets the color of the grid lines.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the grid lines.\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        tick0\n            Sets the placement of the first tick on this axis. Use\n            with `dtick`. If the axis `type` is "log", then you\n            must take the log of your starting tick (e.g. to set\n            the starting tick to 100, set the `tick0` to 2) except\n            when `dtick`=*L<f>* (see `dtick` for more info). If the\n            axis `type` is "date", it should be a date string, like\n            date data. If the axis `type` is "category", it should\n            be a number, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        tickcolor\n            Sets the tick color.\n        ticklen\n            Sets the tick length (in px).\n        tickmode\n            Sets the tick mode for this axis. If "auto", the number\n            of ticks is set via `nticks`. If "linear", the\n            placement of the ticks is determined by a starting\n            position `tick0` and a tick step `dtick` ("linear" is\n            the default value if `tick0` and `dtick` are provided).\n            If "array", the placement of the ticks is set via\n            `tickvals` and the tick text is `ticktext`. ("array" is\n            the default value if `tickvals` is provided).\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n\n        Returns\n        -------\n        Minor\n        '
        super(Minor, self).__init__('minor')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.xaxis.Minor\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.xaxis.Minor`')
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
        _v = arg.pop('nticks', None)
        _v = nticks if nticks is not None else _v
        if _v is not None:
            self['nticks'] = _v
        _v = arg.pop('showgrid', None)
        _v = showgrid if showgrid is not None else _v
        if _v is not None:
            self['showgrid'] = _v
        _v = arg.pop('tick0', None)
        _v = tick0 if tick0 is not None else _v
        if _v is not None:
            self['tick0'] = _v
        _v = arg.pop('tickcolor', None)
        _v = tickcolor if tickcolor is not None else _v
        if _v is not None:
            self['tickcolor'] = _v
        _v = arg.pop('ticklen', None)
        _v = ticklen if ticklen is not None else _v
        if _v is not None:
            self['ticklen'] = _v
        _v = arg.pop('tickmode', None)
        _v = tickmode if tickmode is not None else _v
        if _v is not None:
            self['tickmode'] = _v
        _v = arg.pop('ticks', None)
        _v = ticks if ticks is not None else _v
        if _v is not None:
            self['ticks'] = _v
        _v = arg.pop('tickvals', None)
        _v = tickvals if tickvals is not None else _v
        if _v is not None:
            self['tickvals'] = _v
        _v = arg.pop('tickvalssrc', None)
        _v = tickvalssrc if tickvalssrc is not None else _v
        if _v is not None:
            self['tickvalssrc'] = _v
        _v = arg.pop('tickwidth', None)
        _v = tickwidth if tickwidth is not None else _v
        if _v is not None:
            self['tickwidth'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
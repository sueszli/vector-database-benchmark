from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class RadialAxis(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.polar'
    _path_str = 'layout.polar.radialaxis'
    _valid_props = {'angle', 'autorange', 'autorangeoptions', 'autotypenumbers', 'calendar', 'categoryarray', 'categoryarraysrc', 'categoryorder', 'color', 'dtick', 'exponentformat', 'gridcolor', 'griddash', 'gridwidth', 'hoverformat', 'labelalias', 'layer', 'linecolor', 'linewidth', 'maxallowed', 'minallowed', 'minexponent', 'nticks', 'range', 'rangemode', 'separatethousands', 'showexponent', 'showgrid', 'showline', 'showticklabels', 'showtickprefix', 'showticksuffix', 'side', 'tick0', 'tickangle', 'tickcolor', 'tickfont', 'tickformat', 'tickformatstopdefaults', 'tickformatstops', 'ticklabelstep', 'ticklen', 'tickmode', 'tickprefix', 'ticks', 'ticksuffix', 'ticktext', 'ticktextsrc', 'tickvals', 'tickvalssrc', 'tickwidth', 'title', 'titlefont', 'type', 'uirevision', 'visible'}

    @property
    def angle(self):
        if False:
            print('Hello World!')
        "\n        Sets the angle (in degrees) from which the radial axis is\n        drawn. Note that by default, radial axis line on the theta=0\n        line corresponds to a line pointing right (like what\n        mathematicians prefer). Defaults to the first `polar.sector`\n        angle.\n\n        The 'angle' property is a angle (in degrees) that may be\n        specified as a number between -180 and 180.\n        Numeric values outside this range are converted to the equivalent value\n        (e.g. 270 is converted to -90).\n\n        Returns\n        -------\n        int|float\n        "
        return self['angle']

    @angle.setter
    def angle(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['angle'] = val

    @property
    def autorange(self):
        if False:
            return 10
        '\n        Determines whether or not the range of this axis is computed in\n        relation to the input data. See `rangemode` for more info. If\n        `range` is provided and it has a value for both the lower and\n        upper bound, `autorange` is set to False. Using "min" applies\n        autorange only to set the minimum. Using "max" applies\n        autorange only to set the maximum. Using *min reversed* applies\n        autorange only to set the minimum on a reversed axis. Using\n        *max reversed* applies autorange only to set the maximum on a\n        reversed axis. Using "reversed" applies autorange on both ends\n        and reverses the axis direction.\n\n        The \'autorange\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, \'reversed\', \'min reversed\', \'max reversed\',\n                \'min\', \'max\']\n\n        Returns\n        -------\n        Any\n        '
        return self['autorange']

    @autorange.setter
    def autorange(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['autorange'] = val

    @property
    def autorangeoptions(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'autorangeoptions' property is an instance of Autorangeoptions\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.polar.radialaxis.Autorangeoptions`\n          - A dict of string/value properties that will be passed\n            to the Autorangeoptions constructor\n\n            Supported dict properties:\n\n                clipmax\n                    Clip autorange maximum if it goes beyond this\n                    value. Has no effect when\n                    `autorangeoptions.maxallowed` is provided.\n                clipmin\n                    Clip autorange minimum if it goes beyond this\n                    value. Has no effect when\n                    `autorangeoptions.minallowed` is provided.\n                include\n                    Ensure this value is included in autorange.\n                includesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `include`.\n                maxallowed\n                    Use this value exactly as autorange maximum.\n                minallowed\n                    Use this value exactly as autorange minimum.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.polar.radialaxis.Autorangeoptions\n        "
        return self['autorangeoptions']

    @autorangeoptions.setter
    def autorangeoptions(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['autorangeoptions'] = val

    @property
    def autotypenumbers(self):
        if False:
            print('Hello World!')
        '\n        Using "strict" a numeric string in trace data is not converted\n        to a number. Using *convert types* a numeric string in trace\n        data may be treated as a number during automatic axis `type`\n        detection. Defaults to layout.autotypenumbers.\n\n        The \'autotypenumbers\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'convert types\', \'strict\']\n\n        Returns\n        -------\n        Any\n        '
        return self['autotypenumbers']

    @autotypenumbers.setter
    def autotypenumbers(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['autotypenumbers'] = val

    @property
    def calendar(self):
        if False:
            while True:
                i = 10
        "\n        Sets the calendar system to use for `range` and `tick0` if this\n        is a date axis. This does not set the calendar for interpreting\n        data on this axis, that's specified in the trace or via the\n        global `layout.calendar`\n\n        The 'calendar' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['chinese', 'coptic', 'discworld', 'ethiopian',\n                'gregorian', 'hebrew', 'islamic', 'jalali', 'julian',\n                'mayan', 'nanakshahi', 'nepali', 'persian', 'taiwan',\n                'thai', 'ummalqura']\n\n        Returns\n        -------\n        Any\n        "
        return self['calendar']

    @calendar.setter
    def calendar(self, val):
        if False:
            while True:
                i = 10
        self['calendar'] = val

    @property
    def categoryarray(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the order in which categories on this axis appear. Only\n        has an effect if `categoryorder` is set to "array". Used with\n        `categoryorder`.\n\n        The \'categoryarray\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['categoryarray']

    @categoryarray.setter
    def categoryarray(self, val):
        if False:
            print('Hello World!')
        self['categoryarray'] = val

    @property
    def categoryarraysrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for\n        `categoryarray`.\n\n        The 'categoryarraysrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['categoryarraysrc']

    @categoryarraysrc.setter
    def categoryarraysrc(self, val):
        if False:
            return 10
        self['categoryarraysrc'] = val

    @property
    def categoryorder(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Specifies the ordering logic for the case of categorical\n        variables. By default, plotly uses "trace", which specifies the\n        order that is present in the data supplied. Set `categoryorder`\n        to *category ascending* or *category descending* if order\n        should be determined by the alphanumerical order of the\n        category names. Set `categoryorder` to "array" to derive the\n        ordering from the attribute `categoryarray`. If a category is\n        not found in the `categoryarray` array, the sorting behavior\n        for that attribute will be identical to the "trace" mode. The\n        unspecified categories will follow the categories in\n        `categoryarray`. Set `categoryorder` to *total ascending* or\n        *total descending* if order should be determined by the\n        numerical order of the values. Similarly, the order can be\n        determined by the min, max, sum, mean or median of all the\n        values.\n\n        The \'categoryorder\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'trace\', \'category ascending\', \'category descending\',\n                \'array\', \'total ascending\', \'total descending\', \'min\n                ascending\', \'min descending\', \'max ascending\', \'max\n                descending\', \'sum ascending\', \'sum descending\', \'mean\n                ascending\', \'mean descending\', \'median ascending\', \'median\n                descending\']\n\n        Returns\n        -------\n        Any\n        '
        return self['categoryorder']

    @categoryorder.setter
    def categoryorder(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['categoryorder'] = val

    @property
    def color(self):
        if False:
            print('Hello World!')
        "\n        Sets default for all colors associated with this axis all at\n        once: line, font, tick, and grid colors. Grid color is\n        lightened by blending this with the plot background Individual\n        pieces can override this.\n\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            print('Hello World!')
        self['color'] = val

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
            print('Hello World!')
        self['dtick'] = val

    @property
    def exponentformat(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines a formatting rule for the tick exponents. For\n        example, consider the number 1,000,000,000. If "none", it\n        appears as 1,000,000,000. If "e", 1e+9. If "E", 1E+9. If\n        "power", 1x10^9 (with 9 in a super script). If "SI", 1G. If\n        "B", 1B.\n\n        The \'exponentformat\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'none\', \'e\', \'E\', \'power\', \'SI\', \'B\']\n\n        Returns\n        -------\n        Any\n        '
        return self['exponentformat']

    @exponentformat.setter
    def exponentformat(self, val):
        if False:
            while True:
                i = 10
        self['exponentformat'] = val

    @property
    def gridcolor(self):
        if False:
            print('Hello World!')
        "\n        Sets the color of the grid lines.\n\n        The 'gridcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['gridcolor']

    @gridcolor.setter
    def gridcolor(self, val):
        if False:
            while True:
                i = 10
        self['gridcolor'] = val

    @property
    def griddash(self):
        if False:
            print('Hello World!')
        '\n        Sets the dash style of lines. Set to a dash type string\n        ("solid", "dot", "dash", "longdash", "dashdot", or\n        "longdashdot") or a dash length list in px (eg\n        "5px,10px,2px,2px").\n\n        The \'griddash\' property is an enumeration that may be specified as:\n          - One of the following dash styles:\n                [\'solid\', \'dot\', \'dash\', \'longdash\', \'dashdot\', \'longdashdot\']\n          - A string containing a dash length list in pixels or percentages\n                (e.g. \'5px 10px 2px 2px\', \'5, 10, 2, 2\', \'10% 20% 40%\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['griddash']

    @griddash.setter
    def griddash(self, val):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
        self['gridwidth'] = val

    @property
    def hoverformat(self):
        if False:
            return 10
        '\n        Sets the hover text formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display "09~15~23.46"\n\n        The \'hoverformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['hoverformat']

    @hoverformat.setter
    def hoverformat(self, val):
        if False:
            while True:
                i = 10
        self['hoverformat'] = val

    @property
    def labelalias(self):
        if False:
            print('Hello World!')
        "\n        Replacement text for specific tick or hover labels. For example\n        using {US: 'USA', CA: 'Canada'} changes US to USA and CA to\n        Canada. The labels we would have shown must match the keys\n        exactly, after adding any tickprefix or ticksuffix. For\n        negative numbers the minus sign symbol used (U+2212) is wider\n        than the regular ascii dash. That means you need to use −1\n        instead of -1. labelalias can be used with any axis type, and\n        both keys (if needed) and values (if desired) can include html-\n        like tags or MathJax.\n\n        The 'labelalias' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['labelalias']

    @labelalias.setter
    def labelalias(self, val):
        if False:
            return 10
        self['labelalias'] = val

    @property
    def layer(self):
        if False:
            print('Hello World!')
        "\n        Sets the layer on which this axis is displayed. If *above\n        traces*, this axis is displayed above all the subplot's traces\n        If *below traces*, this axis is displayed below all the\n        subplot's traces, but above the grid lines. Useful when used\n        together with scatter-like traces with `cliponaxis` set to\n        False to show markers and/or text nodes above this axis.\n\n        The 'layer' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['above traces', 'below traces']\n\n        Returns\n        -------\n        Any\n        "
        return self['layer']

    @layer.setter
    def layer(self, val):
        if False:
            return 10
        self['layer'] = val

    @property
    def linecolor(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the axis line color.\n\n        The 'linecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['linecolor']

    @linecolor.setter
    def linecolor(self, val):
        if False:
            return 10
        self['linecolor'] = val

    @property
    def linewidth(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the width (in px) of the axis line.\n\n        The 'linewidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['linewidth']

    @linewidth.setter
    def linewidth(self, val):
        if False:
            i = 10
            return i + 15
        self['linewidth'] = val

    @property
    def maxallowed(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines the maximum range of this axis.\n\n        The 'maxallowed' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['maxallowed']

    @maxallowed.setter
    def maxallowed(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['maxallowed'] = val

    @property
    def minallowed(self):
        if False:
            return 10
        "\n        Determines the minimum range of this axis.\n\n        The 'minallowed' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['minallowed']

    @minallowed.setter
    def minallowed(self, val):
        if False:
            i = 10
            return i + 15
        self['minallowed'] = val

    @property
    def minexponent(self):
        if False:
            i = 10
            return i + 15
        '\n        Hide SI prefix for 10^n if |n| is below this number. This only\n        has an effect when `tickformat` is "SI" or "B".\n\n        The \'minexponent\' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        '
        return self['minexponent']

    @minexponent.setter
    def minexponent(self, val):
        if False:
            return 10
        self['minexponent'] = val

    @property
    def nticks(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Specifies the maximum number of ticks for the particular axis.\n        The actual number of ticks will be chosen automatically to be\n        less than or equal to `nticks`. Has an effect only if\n        `tickmode` is set to "auto".\n\n        The \'nticks\' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        '
        return self['nticks']

    @nticks.setter
    def nticks(self, val):
        if False:
            while True:
                i = 10
        self['nticks'] = val

    @property
    def range(self):
        if False:
            for i in range(10):
                print('nop')
        '\n            Sets the range of this axis. If the axis `type` is "log", then\n            you must take the log of your desired range (e.g. to set the\n            range from 1 to 100, set the range from 0 to 2). If the axis\n            `type` is "date", it should be date strings, like date data,\n            though Date objects and unix milliseconds will be accepted and\n            converted to strings. If the axis `type` is "category", it\n            should be numbers, using the scale where each category is\n            assigned a serial number from zero in the order it appears.\n            Leaving either or both elements `null` impacts the default\n            `autorange`.\n\n            The \'range\' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The \'range[0]\' property accepts values of any type\n        (1) The \'range[1]\' property accepts values of any type\n\n            Returns\n            -------\n            list\n        '
        return self['range']

    @range.setter
    def range(self, val):
        if False:
            print('Hello World!')
        self['range'] = val

    @property
    def rangemode(self):
        if False:
            return 10
        '\n        If *tozero*`, the range extends to 0, regardless of the input\n        data If "nonnegative", the range is non-negative, regardless of\n        the input data. If "normal", the range is computed in relation\n        to the extrema of the input data (same behavior as for\n        cartesian axes).\n\n        The \'rangemode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'tozero\', \'nonnegative\', \'normal\']\n\n        Returns\n        -------\n        Any\n        '
        return self['rangemode']

    @rangemode.setter
    def rangemode(self, val):
        if False:
            while True:
                i = 10
        self['rangemode'] = val

    @property
    def separatethousands(self):
        if False:
            print('Hello World!')
        '\n        If "true", even 4-digit integers are separated\n\n        The \'separatethousands\' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        '
        return self['separatethousands']

    @separatethousands.setter
    def separatethousands(self, val):
        if False:
            while True:
                i = 10
        self['separatethousands'] = val

    @property
    def showexponent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If "all", all exponents are shown besides their significands.\n        If "first", only the exponent of the first tick is shown. If\n        "last", only the exponent of the last tick is shown. If "none",\n        no exponents appear.\n\n        The \'showexponent\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'all\', \'first\', \'last\', \'none\']\n\n        Returns\n        -------\n        Any\n        '
        return self['showexponent']

    @showexponent.setter
    def showexponent(self, val):
        if False:
            i = 10
            return i + 15
        self['showexponent'] = val

    @property
    def showgrid(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not grid lines are drawn. If True, the\n        grid lines are drawn at every tick mark.\n\n        The 'showgrid' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showgrid']

    @showgrid.setter
    def showgrid(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showgrid'] = val

    @property
    def showline(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not a line bounding this axis is drawn.\n\n        The 'showline' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showline']

    @showline.setter
    def showline(self, val):
        if False:
            while True:
                i = 10
        self['showline'] = val

    @property
    def showticklabels(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not the tick labels are drawn.\n\n        The 'showticklabels' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showticklabels']

    @showticklabels.setter
    def showticklabels(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showticklabels'] = val

    @property
    def showtickprefix(self):
        if False:
            i = 10
            return i + 15
        '\n        If "all", all tick labels are displayed with a prefix. If\n        "first", only the first tick is displayed with a prefix. If\n        "last", only the last tick is displayed with a suffix. If\n        "none", tick prefixes are hidden.\n\n        The \'showtickprefix\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'all\', \'first\', \'last\', \'none\']\n\n        Returns\n        -------\n        Any\n        '
        return self['showtickprefix']

    @showtickprefix.setter
    def showtickprefix(self, val):
        if False:
            while True:
                i = 10
        self['showtickprefix'] = val

    @property
    def showticksuffix(self):
        if False:
            return 10
        "\n        Same as `showtickprefix` but for tick suffixes.\n\n        The 'showticksuffix' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['all', 'first', 'last', 'none']\n\n        Returns\n        -------\n        Any\n        "
        return self['showticksuffix']

    @showticksuffix.setter
    def showticksuffix(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showticksuffix'] = val

    @property
    def side(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines on which side of radial axis line the tick and tick\n        labels appear.\n\n        The 'side' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['clockwise', 'counterclockwise']\n\n        Returns\n        -------\n        Any\n        "
        return self['side']

    @side.setter
    def side(self, val):
        if False:
            while True:
                i = 10
        self['side'] = val

    @property
    def tick0(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the placement of the first tick on this axis. Use with\n        `dtick`. If the axis `type` is "log", then you must take the\n        log of your starting tick (e.g. to set the starting tick to\n        100, set the `tick0` to 2) except when `dtick`=*L<f>* (see\n        `dtick` for more info). If the axis `type` is "date", it should\n        be a date string, like date data. If the axis `type` is\n        "category", it should be a number, using the scale where each\n        category is assigned a serial number from zero in the order it\n        appears.\n\n        The \'tick0\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['tick0']

    @tick0.setter
    def tick0(self, val):
        if False:
            i = 10
            return i + 15
        self['tick0'] = val

    @property
    def tickangle(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the angle of the tick labels with respect to the\n        horizontal. For example, a `tickangle` of -90 draws the tick\n        labels vertically.\n\n        The 'tickangle' property is a angle (in degrees) that may be\n        specified as a number between -180 and 180.\n        Numeric values outside this range are converted to the equivalent value\n        (e.g. 270 is converted to -90).\n\n        Returns\n        -------\n        int|float\n        "
        return self['tickangle']

    @tickangle.setter
    def tickangle(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickangle'] = val

    @property
    def tickcolor(self):
        if False:
            print('Hello World!')
        "\n        Sets the tick color.\n\n        The 'tickcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['tickcolor']

    @tickcolor.setter
    def tickcolor(self, val):
        if False:
            i = 10
            return i + 15
        self['tickcolor'] = val

    @property
    def tickfont(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the tick font.\n\n        The \'tickfont\' property is an instance of Tickfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.polar.radialaxis.Tickfont`\n          - A dict of string/value properties that will be passed\n            to the Tickfont constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.polar.radialaxis.Tickfont\n        '
        return self['tickfont']

    @tickfont.setter
    def tickfont(self, val):
        if False:
            print('Hello World!')
        self['tickfont'] = val

    @property
    def tickformat(self):
        if False:
            print('Hello World!')
        '\n        Sets the tick label formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display "09~15~23.46"\n\n        The \'tickformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['tickformat']

    @tickformat.setter
    def tickformat(self, val):
        if False:
            return 10
        self['tickformat'] = val

    @property
    def tickformatstops(self):
        if False:
            print('Hello World!')
        '\n        The \'tickformatstops\' property is a tuple of instances of\n        Tickformatstop that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.layout.polar.radialaxis.Tickformatstop\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Tickformatstop constructor\n\n            Supported dict properties:\n\n                dtickrange\n                    range [*min*, *max*], where "min", "max" -\n                    dtick values which describe some zoom level, it\n                    is possible to omit "min" or "max" value by\n                    passing "null"\n                enabled\n                    Determines whether or not this stop is used. If\n                    `false`, this stop is ignored even within its\n                    `dtickrange`.\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n                value\n                    string - dtickformat for described zoom level,\n                    the same as "tickformat"\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.layout.polar.radialaxis.Tickformatstop]\n        '
        return self['tickformatstops']

    @tickformatstops.setter
    def tickformatstops(self, val):
        if False:
            print('Hello World!')
        self['tickformatstops'] = val

    @property
    def tickformatstopdefaults(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        When used in a template (as layout.template.layout.polar.radial\n        axis.tickformatstopdefaults), sets the default property values\n        to use for elements of layout.polar.radialaxis.tickformatstops\n\n        The 'tickformatstopdefaults' property is an instance of Tickformatstop\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.polar.radialaxis.Tickformatstop`\n          - A dict of string/value properties that will be passed\n            to the Tickformatstop constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.layout.polar.radialaxis.Tickformatstop\n        "
        return self['tickformatstopdefaults']

    @tickformatstopdefaults.setter
    def tickformatstopdefaults(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickformatstopdefaults'] = val

    @property
    def ticklabelstep(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the spacing between tick labels as compared to the spacing\n        between ticks. A value of 1 (default) means each tick gets a\n        label. A value of 2 means shows every 2nd label. A larger value\n        n means only every nth tick is labeled. `tick0` determines\n        which labels are shown. Not implemented for axes with `type`\n        "log" or "multicategory", or when `tickmode` is "array".\n\n        The \'ticklabelstep\' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [1, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        '
        return self['ticklabelstep']

    @ticklabelstep.setter
    def ticklabelstep(self, val):
        if False:
            return 10
        self['ticklabelstep'] = val

    @property
    def ticklen(self):
        if False:
            i = 10
            return i + 15
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
            return 10
        '\n        Sets the tick mode for this axis. If "auto", the number of\n        ticks is set via `nticks`. If "linear", the placement of the\n        ticks is determined by a starting position `tick0` and a tick\n        step `dtick` ("linear" is the default value if `tick0` and\n        `dtick` are provided). If "array", the placement of the ticks\n        is set via `tickvals` and the tick text is `ticktext`. ("array"\n        is the default value if `tickvals` is provided).\n\n        The \'tickmode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'linear\', \'array\']\n\n        Returns\n        -------\n        Any\n        '
        return self['tickmode']

    @tickmode.setter
    def tickmode(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickmode'] = val

    @property
    def tickprefix(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets a tick label prefix.\n\n        The 'tickprefix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['tickprefix']

    @tickprefix.setter
    def tickprefix(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickprefix'] = val

    @property
    def ticks(self):
        if False:
            print('Hello World!')
        '\n        Determines whether ticks are drawn or not. If "", this axis\'\n        ticks are not drawn. If "outside" ("inside"), this axis\' are\n        drawn outside (inside) the axis lines.\n\n        The \'ticks\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'outside\', \'inside\', \'\']\n\n        Returns\n        -------\n        Any\n        '
        return self['ticks']

    @ticks.setter
    def ticks(self, val):
        if False:
            while True:
                i = 10
        self['ticks'] = val

    @property
    def ticksuffix(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets a tick label suffix.\n\n        The 'ticksuffix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['ticksuffix']

    @ticksuffix.setter
    def ticksuffix(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['ticksuffix'] = val

    @property
    def ticktext(self):
        if False:
            print('Hello World!')
        '\n        Sets the text displayed at the ticks position via `tickvals`.\n        Only has an effect if `tickmode` is set to "array". Used with\n        `tickvals`.\n\n        The \'ticktext\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['ticktext']

    @ticktext.setter
    def ticktext(self, val):
        if False:
            print('Hello World!')
        self['ticktext'] = val

    @property
    def ticktextsrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `ticktext`.\n\n        The 'ticktextsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['ticktextsrc']

    @ticktextsrc.setter
    def ticktextsrc(self, val):
        if False:
            while True:
                i = 10
        self['ticktextsrc'] = val

    @property
    def tickvals(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the values at which ticks on this axis appear. Only has an\n        effect if `tickmode` is set to "array". Used with `ticktext`.\n\n        The \'tickvals\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['tickvals']

    @tickvals.setter
    def tickvals(self, val):
        if False:
            return 10
        self['tickvals'] = val

    @property
    def tickvalssrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `tickvals`.\n\n        The 'tickvalssrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['tickvalssrc']

    @tickvalssrc.setter
    def tickvalssrc(self, val):
        if False:
            while True:
                i = 10
        self['tickvalssrc'] = val

    @property
    def tickwidth(self):
        if False:
            return 10
        "\n        Sets the tick width (in px).\n\n        The 'tickwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['tickwidth']

    @tickwidth.setter
    def tickwidth(self, val):
        if False:
            while True:
                i = 10
        self['tickwidth'] = val

    @property
    def title(self):
        if False:
            while True:
                i = 10
        "\n        The 'title' property is an instance of Title\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.polar.radialaxis.Title`\n          - A dict of string/value properties that will be passed\n            to the Title constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this axis' title font. Note that the\n                    title's font used to be customized by the now\n                    deprecated `titlefont` attribute.\n                text\n                    Sets the title of this axis. Note that before\n                    the existence of `title.text`, the title's\n                    contents used to be defined as the `title`\n                    attribute itself. This behavior has been\n                    deprecated.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.polar.radialaxis.Title\n        "
        return self['title']

    @title.setter
    def title(self, val):
        if False:
            print('Hello World!')
        self['title'] = val

    @property
    def titlefont(self):
        if False:
            i = 10
            return i + 15
        '\n        Deprecated: Please use layout.polar.radialaxis.title.font\n        instead. Sets this axis\' title font. Note that the title\'s font\n        used to be customized by the now deprecated `titlefont`\n        attribute.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.polar.radialaxis.title.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n\n        '
        return self['titlefont']

    @titlefont.setter
    def titlefont(self, val):
        if False:
            return 10
        self['titlefont'] = val

    @property
    def type(self):
        if False:
            while True:
                i = 10
        "\n        Sets the axis type. By default, plotly attempts to determined\n        the axis type by looking into the data of the traces that\n        referenced the axis in question.\n\n        The 'type' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['-', 'linear', 'log', 'date', 'category']\n\n        Returns\n        -------\n        Any\n        "
        return self['type']

    @type.setter
    def type(self, val):
        if False:
            i = 10
            return i + 15
        self['type'] = val

    @property
    def uirevision(self):
        if False:
            i = 10
            return i + 15
        "\n        Controls persistence of user-driven changes in axis `range`,\n        `autorange`, `angle`, and `title` if in `editable: true`\n        configuration. Defaults to `polar<N>.uirevision`.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            return 10
        self['uirevision'] = val

    @property
    def visible(self):
        if False:
            print('Hello World!')
        "\n        A single toggle to hide the axis while preserving interaction\n        like dragging. Default is true when a cheater plot is present\n        on the axis, otherwise false\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            while True:
                i = 10
        self['visible'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        angle\n            Sets the angle (in degrees) from which the radial axis\n            is drawn. Note that by default, radial axis line on the\n            theta=0 line corresponds to a line pointing right (like\n            what mathematicians prefer). Defaults to the first\n            `polar.sector` angle.\n        autorange\n            Determines whether or not the range of this axis is\n            computed in relation to the input data. See `rangemode`\n            for more info. If `range` is provided and it has a\n            value for both the lower and upper bound, `autorange`\n            is set to False. Using "min" applies autorange only to\n            set the minimum. Using "max" applies autorange only to\n            set the maximum. Using *min reversed* applies autorange\n            only to set the minimum on a reversed axis. Using *max\n            reversed* applies autorange only to set the maximum on\n            a reversed axis. Using "reversed" applies autorange on\n            both ends and reverses the axis direction.\n        autorangeoptions\n            :class:`plotly.graph_objects.layout.polar.radialaxis.Au\n            torangeoptions` instance or dict with compatible\n            properties\n        autotypenumbers\n            Using "strict" a numeric string in trace data is not\n            converted to a number. Using *convert types* a numeric\n            string in trace data may be treated as a number during\n            automatic axis `type` detection. Defaults to\n            layout.autotypenumbers.\n        calendar\n            Sets the calendar system to use for `range` and `tick0`\n            if this is a date axis. This does not set the calendar\n            for interpreting data on this axis, that\'s specified in\n            the trace or via the global `layout.calendar`\n        categoryarray\n            Sets the order in which categories on this axis appear.\n            Only has an effect if `categoryorder` is set to\n            "array". Used with `categoryorder`.\n        categoryarraysrc\n            Sets the source reference on Chart Studio Cloud for\n            `categoryarray`.\n        categoryorder\n            Specifies the ordering logic for the case of\n            categorical variables. By default, plotly uses "trace",\n            which specifies the order that is present in the data\n            supplied. Set `categoryorder` to *category ascending*\n            or *category descending* if order should be determined\n            by the alphanumerical order of the category names. Set\n            `categoryorder` to "array" to derive the ordering from\n            the attribute `categoryarray`. If a category is not\n            found in the `categoryarray` array, the sorting\n            behavior for that attribute will be identical to the\n            "trace" mode. The unspecified categories will follow\n            the categories in `categoryarray`. Set `categoryorder`\n            to *total ascending* or *total descending* if order\n            should be determined by the numerical order of the\n            values. Similarly, the order can be determined by the\n            min, max, sum, mean or median of all the values.\n        color\n            Sets default for all colors associated with this axis\n            all at once: line, font, tick, and grid colors. Grid\n            color is lightened by blending this with the plot\n            background Individual pieces can override this.\n        dtick\n            Sets the step in-between ticks on this axis. Use with\n            `tick0`. Must be a positive number, or special strings\n            available to "log" and "date" axes. If the axis `type`\n            is "log", then ticks are set every 10^(n*dtick) where n\n            is the tick number. For example, to set a tick mark at\n            1, 10, 100, 1000, ... set dtick to 1. To set tick marks\n            at 1, 100, 10000, ... set dtick to 2. To set tick marks\n            at 1, 5, 25, 125, 625, 3125, ... set dtick to\n            log_10(5), or 0.69897000433. "log" has several special\n            values; "L<f>", where `f` is a positive number, gives\n            ticks linearly spaced in value (but not position). For\n            example `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n            at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus\n            small digits between, use "D1" (all digits) or "D2"\n            (only 2 and 5). `tick0` is ignored for "D1" and "D2".\n            If the axis `type` is "date", then you must convert the\n            time to milliseconds. For example, to set the interval\n            between ticks to one day, set `dtick` to 86400000.0.\n            "date" also has special values "M<n>" gives ticks\n            spaced by a number of months. `n` must be a positive\n            integer. To set ticks on the 15th of every third month,\n            set `tick0` to "2000-01-15" and `dtick` to "M3". To set\n            ticks every 4 years, set `dtick` to "M48"\n        exponentformat\n            Determines a formatting rule for the tick exponents.\n            For example, consider the number 1,000,000,000. If\n            "none", it appears as 1,000,000,000. If "e", 1e+9. If\n            "E", 1E+9. If "power", 1x10^9 (with 9 in a super\n            script). If "SI", 1G. If "B", 1B.\n        gridcolor\n            Sets the color of the grid lines.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the grid lines.\n        hoverformat\n            Sets the hover text formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        layer\n            Sets the layer on which this axis is displayed. If\n            *above traces*, this axis is displayed above all the\n            subplot\'s traces If *below traces*, this axis is\n            displayed below all the subplot\'s traces, but above the\n            grid lines. Useful when used together with scatter-like\n            traces with `cliponaxis` set to False to show markers\n            and/or text nodes above this axis.\n        linecolor\n            Sets the axis line color.\n        linewidth\n            Sets the width (in px) of the axis line.\n        maxallowed\n            Determines the maximum range of this axis.\n        minallowed\n            Determines the minimum range of this axis.\n        minexponent\n            Hide SI prefix for 10^n if |n| is below this number.\n            This only has an effect when `tickformat` is "SI" or\n            "B".\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        range\n            Sets the range of this axis. If the axis `type` is\n            "log", then you must take the log of your desired range\n            (e.g. to set the range from 1 to 100, set the range\n            from 0 to 2). If the axis `type` is "date", it should\n            be date strings, like date data, though Date objects\n            and unix milliseconds will be accepted and converted to\n            strings. If the axis `type` is "category", it should be\n            numbers, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears. Leaving either or both elements `null` impacts\n            the default `autorange`.\n        rangemode\n            If *tozero*`, the range extends to 0, regardless of the\n            input data If "nonnegative", the range is non-negative,\n            regardless of the input data. If "normal", the range is\n            computed in relation to the extrema of the input data\n            (same behavior as for cartesian axes).\n        separatethousands\n            If "true", even 4-digit integers are separated\n        showexponent\n            If "all", all exponents are shown besides their\n            significands. If "first", only the exponent of the\n            first tick is shown. If "last", only the exponent of\n            the last tick is shown. If "none", no exponents appear.\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        showline\n            Determines whether or not a line bounding this axis is\n            drawn.\n        showticklabels\n            Determines whether or not the tick labels are drawn.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        side\n            Determines on which side of radial axis line the tick\n            and tick labels appear.\n        tick0\n            Sets the placement of the first tick on this axis. Use\n            with `dtick`. If the axis `type` is "log", then you\n            must take the log of your starting tick (e.g. to set\n            the starting tick to 100, set the `tick0` to 2) except\n            when `dtick`=*L<f>* (see `dtick` for more info). If the\n            axis `type` is "date", it should be a date string, like\n            date data. If the axis `type` is "category", it should\n            be a number, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        tickangle\n            Sets the angle of the tick labels with respect to the\n            horizontal. For example, a `tickangle` of -90 draws the\n            tick labels vertically.\n        tickcolor\n            Sets the tick color.\n        tickfont\n            Sets the tick font.\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        tickformatstops\n            A tuple of :class:`plotly.graph_objects.layout.polar.ra\n            dialaxis.Tickformatstop` instances or dicts with\n            compatible properties\n        tickformatstopdefaults\n            When used in a template (as layout.template.layout.pola\n            r.radialaxis.tickformatstopdefaults), sets the default\n            property values to use for elements of\n            layout.polar.radialaxis.tickformatstops\n        ticklabelstep\n            Sets the spacing between tick labels as compared to the\n            spacing between ticks. A value of 1 (default) means\n            each tick gets a label. A value of 2 means shows every\n            2nd label. A larger value n means only every nth tick\n            is labeled. `tick0` determines which labels are shown.\n            Not implemented for axes with `type` "log" or\n            "multicategory", or when `tickmode` is "array".\n        ticklen\n            Sets the tick length (in px).\n        tickmode\n            Sets the tick mode for this axis. If "auto", the number\n            of ticks is set via `nticks`. If "linear", the\n            placement of the ticks is determined by a starting\n            position `tick0` and a tick step `dtick` ("linear" is\n            the default value if `tick0` and `dtick` are provided).\n            If "array", the placement of the ticks is set via\n            `tickvals` and the tick text is `ticktext`. ("array" is\n            the default value if `tickvals` is provided).\n        tickprefix\n            Sets a tick label prefix.\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        ticksuffix\n            Sets a tick label suffix.\n        ticktext\n            Sets the text displayed at the ticks position via\n            `tickvals`. Only has an effect if `tickmode` is set to\n            "array". Used with `tickvals`.\n        ticktextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `ticktext`.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n        title\n            :class:`plotly.graph_objects.layout.polar.radialaxis.Ti\n            tle` instance or dict with compatible properties\n        titlefont\n            Deprecated: Please use\n            layout.polar.radialaxis.title.font instead. Sets this\n            axis\' title font. Note that the title\'s font used to be\n            customized by the now deprecated `titlefont` attribute.\n        type\n            Sets the axis type. By default, plotly attempts to\n            determined the axis type by looking into the data of\n            the traces that referenced the axis in question.\n        uirevision\n            Controls persistence of user-driven changes in axis\n            `range`, `autorange`, `angle`, and `title` if in\n            `editable: true` configuration. Defaults to\n            `polar<N>.uirevision`.\n        visible\n            A single toggle to hide the axis while preserving\n            interaction like dragging. Default is true when a\n            cheater plot is present on the axis, otherwise false\n        '
    _mapped_properties = {'titlefont': ('title', 'font')}

    def __init__(self, arg=None, angle=None, autorange=None, autorangeoptions=None, autotypenumbers=None, calendar=None, categoryarray=None, categoryarraysrc=None, categoryorder=None, color=None, dtick=None, exponentformat=None, gridcolor=None, griddash=None, gridwidth=None, hoverformat=None, labelalias=None, layer=None, linecolor=None, linewidth=None, maxallowed=None, minallowed=None, minexponent=None, nticks=None, range=None, rangemode=None, separatethousands=None, showexponent=None, showgrid=None, showline=None, showticklabels=None, showtickprefix=None, showticksuffix=None, side=None, tick0=None, tickangle=None, tickcolor=None, tickfont=None, tickformat=None, tickformatstops=None, tickformatstopdefaults=None, ticklabelstep=None, ticklen=None, tickmode=None, tickprefix=None, ticks=None, ticksuffix=None, ticktext=None, ticktextsrc=None, tickvals=None, tickvalssrc=None, tickwidth=None, title=None, titlefont=None, type=None, uirevision=None, visible=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new RadialAxis object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.polar.RadialAxis`\n        angle\n            Sets the angle (in degrees) from which the radial axis\n            is drawn. Note that by default, radial axis line on the\n            theta=0 line corresponds to a line pointing right (like\n            what mathematicians prefer). Defaults to the first\n            `polar.sector` angle.\n        autorange\n            Determines whether or not the range of this axis is\n            computed in relation to the input data. See `rangemode`\n            for more info. If `range` is provided and it has a\n            value for both the lower and upper bound, `autorange`\n            is set to False. Using "min" applies autorange only to\n            set the minimum. Using "max" applies autorange only to\n            set the maximum. Using *min reversed* applies autorange\n            only to set the minimum on a reversed axis. Using *max\n            reversed* applies autorange only to set the maximum on\n            a reversed axis. Using "reversed" applies autorange on\n            both ends and reverses the axis direction.\n        autorangeoptions\n            :class:`plotly.graph_objects.layout.polar.radialaxis.Au\n            torangeoptions` instance or dict with compatible\n            properties\n        autotypenumbers\n            Using "strict" a numeric string in trace data is not\n            converted to a number. Using *convert types* a numeric\n            string in trace data may be treated as a number during\n            automatic axis `type` detection. Defaults to\n            layout.autotypenumbers.\n        calendar\n            Sets the calendar system to use for `range` and `tick0`\n            if this is a date axis. This does not set the calendar\n            for interpreting data on this axis, that\'s specified in\n            the trace or via the global `layout.calendar`\n        categoryarray\n            Sets the order in which categories on this axis appear.\n            Only has an effect if `categoryorder` is set to\n            "array". Used with `categoryorder`.\n        categoryarraysrc\n            Sets the source reference on Chart Studio Cloud for\n            `categoryarray`.\n        categoryorder\n            Specifies the ordering logic for the case of\n            categorical variables. By default, plotly uses "trace",\n            which specifies the order that is present in the data\n            supplied. Set `categoryorder` to *category ascending*\n            or *category descending* if order should be determined\n            by the alphanumerical order of the category names. Set\n            `categoryorder` to "array" to derive the ordering from\n            the attribute `categoryarray`. If a category is not\n            found in the `categoryarray` array, the sorting\n            behavior for that attribute will be identical to the\n            "trace" mode. The unspecified categories will follow\n            the categories in `categoryarray`. Set `categoryorder`\n            to *total ascending* or *total descending* if order\n            should be determined by the numerical order of the\n            values. Similarly, the order can be determined by the\n            min, max, sum, mean or median of all the values.\n        color\n            Sets default for all colors associated with this axis\n            all at once: line, font, tick, and grid colors. Grid\n            color is lightened by blending this with the plot\n            background Individual pieces can override this.\n        dtick\n            Sets the step in-between ticks on this axis. Use with\n            `tick0`. Must be a positive number, or special strings\n            available to "log" and "date" axes. If the axis `type`\n            is "log", then ticks are set every 10^(n*dtick) where n\n            is the tick number. For example, to set a tick mark at\n            1, 10, 100, 1000, ... set dtick to 1. To set tick marks\n            at 1, 100, 10000, ... set dtick to 2. To set tick marks\n            at 1, 5, 25, 125, 625, 3125, ... set dtick to\n            log_10(5), or 0.69897000433. "log" has several special\n            values; "L<f>", where `f` is a positive number, gives\n            ticks linearly spaced in value (but not position). For\n            example `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n            at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus\n            small digits between, use "D1" (all digits) or "D2"\n            (only 2 and 5). `tick0` is ignored for "D1" and "D2".\n            If the axis `type` is "date", then you must convert the\n            time to milliseconds. For example, to set the interval\n            between ticks to one day, set `dtick` to 86400000.0.\n            "date" also has special values "M<n>" gives ticks\n            spaced by a number of months. `n` must be a positive\n            integer. To set ticks on the 15th of every third month,\n            set `tick0` to "2000-01-15" and `dtick` to "M3". To set\n            ticks every 4 years, set `dtick` to "M48"\n        exponentformat\n            Determines a formatting rule for the tick exponents.\n            For example, consider the number 1,000,000,000. If\n            "none", it appears as 1,000,000,000. If "e", 1e+9. If\n            "E", 1E+9. If "power", 1x10^9 (with 9 in a super\n            script). If "SI", 1G. If "B", 1B.\n        gridcolor\n            Sets the color of the grid lines.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the grid lines.\n        hoverformat\n            Sets the hover text formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        layer\n            Sets the layer on which this axis is displayed. If\n            *above traces*, this axis is displayed above all the\n            subplot\'s traces If *below traces*, this axis is\n            displayed below all the subplot\'s traces, but above the\n            grid lines. Useful when used together with scatter-like\n            traces with `cliponaxis` set to False to show markers\n            and/or text nodes above this axis.\n        linecolor\n            Sets the axis line color.\n        linewidth\n            Sets the width (in px) of the axis line.\n        maxallowed\n            Determines the maximum range of this axis.\n        minallowed\n            Determines the minimum range of this axis.\n        minexponent\n            Hide SI prefix for 10^n if |n| is below this number.\n            This only has an effect when `tickformat` is "SI" or\n            "B".\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        range\n            Sets the range of this axis. If the axis `type` is\n            "log", then you must take the log of your desired range\n            (e.g. to set the range from 1 to 100, set the range\n            from 0 to 2). If the axis `type` is "date", it should\n            be date strings, like date data, though Date objects\n            and unix milliseconds will be accepted and converted to\n            strings. If the axis `type` is "category", it should be\n            numbers, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears. Leaving either or both elements `null` impacts\n            the default `autorange`.\n        rangemode\n            If *tozero*`, the range extends to 0, regardless of the\n            input data If "nonnegative", the range is non-negative,\n            regardless of the input data. If "normal", the range is\n            computed in relation to the extrema of the input data\n            (same behavior as for cartesian axes).\n        separatethousands\n            If "true", even 4-digit integers are separated\n        showexponent\n            If "all", all exponents are shown besides their\n            significands. If "first", only the exponent of the\n            first tick is shown. If "last", only the exponent of\n            the last tick is shown. If "none", no exponents appear.\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        showline\n            Determines whether or not a line bounding this axis is\n            drawn.\n        showticklabels\n            Determines whether or not the tick labels are drawn.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        side\n            Determines on which side of radial axis line the tick\n            and tick labels appear.\n        tick0\n            Sets the placement of the first tick on this axis. Use\n            with `dtick`. If the axis `type` is "log", then you\n            must take the log of your starting tick (e.g. to set\n            the starting tick to 100, set the `tick0` to 2) except\n            when `dtick`=*L<f>* (see `dtick` for more info). If the\n            axis `type` is "date", it should be a date string, like\n            date data. If the axis `type` is "category", it should\n            be a number, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        tickangle\n            Sets the angle of the tick labels with respect to the\n            horizontal. For example, a `tickangle` of -90 draws the\n            tick labels vertically.\n        tickcolor\n            Sets the tick color.\n        tickfont\n            Sets the tick font.\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        tickformatstops\n            A tuple of :class:`plotly.graph_objects.layout.polar.ra\n            dialaxis.Tickformatstop` instances or dicts with\n            compatible properties\n        tickformatstopdefaults\n            When used in a template (as layout.template.layout.pola\n            r.radialaxis.tickformatstopdefaults), sets the default\n            property values to use for elements of\n            layout.polar.radialaxis.tickformatstops\n        ticklabelstep\n            Sets the spacing between tick labels as compared to the\n            spacing between ticks. A value of 1 (default) means\n            each tick gets a label. A value of 2 means shows every\n            2nd label. A larger value n means only every nth tick\n            is labeled. `tick0` determines which labels are shown.\n            Not implemented for axes with `type` "log" or\n            "multicategory", or when `tickmode` is "array".\n        ticklen\n            Sets the tick length (in px).\n        tickmode\n            Sets the tick mode for this axis. If "auto", the number\n            of ticks is set via `nticks`. If "linear", the\n            placement of the ticks is determined by a starting\n            position `tick0` and a tick step `dtick` ("linear" is\n            the default value if `tick0` and `dtick` are provided).\n            If "array", the placement of the ticks is set via\n            `tickvals` and the tick text is `ticktext`. ("array" is\n            the default value if `tickvals` is provided).\n        tickprefix\n            Sets a tick label prefix.\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        ticksuffix\n            Sets a tick label suffix.\n        ticktext\n            Sets the text displayed at the ticks position via\n            `tickvals`. Only has an effect if `tickmode` is set to\n            "array". Used with `tickvals`.\n        ticktextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `ticktext`.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n        title\n            :class:`plotly.graph_objects.layout.polar.radialaxis.Ti\n            tle` instance or dict with compatible properties\n        titlefont\n            Deprecated: Please use\n            layout.polar.radialaxis.title.font instead. Sets this\n            axis\' title font. Note that the title\'s font used to be\n            customized by the now deprecated `titlefont` attribute.\n        type\n            Sets the axis type. By default, plotly attempts to\n            determined the axis type by looking into the data of\n            the traces that referenced the axis in question.\n        uirevision\n            Controls persistence of user-driven changes in axis\n            `range`, `autorange`, `angle`, and `title` if in\n            `editable: true` configuration. Defaults to\n            `polar<N>.uirevision`.\n        visible\n            A single toggle to hide the axis while preserving\n            interaction like dragging. Default is true when a\n            cheater plot is present on the axis, otherwise false\n\n        Returns\n        -------\n        RadialAxis\n        '
        super(RadialAxis, self).__init__('radialaxis')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.polar.RadialAxis\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.polar.RadialAxis`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('angle', None)
        _v = angle if angle is not None else _v
        if _v is not None:
            self['angle'] = _v
        _v = arg.pop('autorange', None)
        _v = autorange if autorange is not None else _v
        if _v is not None:
            self['autorange'] = _v
        _v = arg.pop('autorangeoptions', None)
        _v = autorangeoptions if autorangeoptions is not None else _v
        if _v is not None:
            self['autorangeoptions'] = _v
        _v = arg.pop('autotypenumbers', None)
        _v = autotypenumbers if autotypenumbers is not None else _v
        if _v is not None:
            self['autotypenumbers'] = _v
        _v = arg.pop('calendar', None)
        _v = calendar if calendar is not None else _v
        if _v is not None:
            self['calendar'] = _v
        _v = arg.pop('categoryarray', None)
        _v = categoryarray if categoryarray is not None else _v
        if _v is not None:
            self['categoryarray'] = _v
        _v = arg.pop('categoryarraysrc', None)
        _v = categoryarraysrc if categoryarraysrc is not None else _v
        if _v is not None:
            self['categoryarraysrc'] = _v
        _v = arg.pop('categoryorder', None)
        _v = categoryorder if categoryorder is not None else _v
        if _v is not None:
            self['categoryorder'] = _v
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('dtick', None)
        _v = dtick if dtick is not None else _v
        if _v is not None:
            self['dtick'] = _v
        _v = arg.pop('exponentformat', None)
        _v = exponentformat if exponentformat is not None else _v
        if _v is not None:
            self['exponentformat'] = _v
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
        _v = arg.pop('hoverformat', None)
        _v = hoverformat if hoverformat is not None else _v
        if _v is not None:
            self['hoverformat'] = _v
        _v = arg.pop('labelalias', None)
        _v = labelalias if labelalias is not None else _v
        if _v is not None:
            self['labelalias'] = _v
        _v = arg.pop('layer', None)
        _v = layer if layer is not None else _v
        if _v is not None:
            self['layer'] = _v
        _v = arg.pop('linecolor', None)
        _v = linecolor if linecolor is not None else _v
        if _v is not None:
            self['linecolor'] = _v
        _v = arg.pop('linewidth', None)
        _v = linewidth if linewidth is not None else _v
        if _v is not None:
            self['linewidth'] = _v
        _v = arg.pop('maxallowed', None)
        _v = maxallowed if maxallowed is not None else _v
        if _v is not None:
            self['maxallowed'] = _v
        _v = arg.pop('minallowed', None)
        _v = minallowed if minallowed is not None else _v
        if _v is not None:
            self['minallowed'] = _v
        _v = arg.pop('minexponent', None)
        _v = minexponent if minexponent is not None else _v
        if _v is not None:
            self['minexponent'] = _v
        _v = arg.pop('nticks', None)
        _v = nticks if nticks is not None else _v
        if _v is not None:
            self['nticks'] = _v
        _v = arg.pop('range', None)
        _v = range if range is not None else _v
        if _v is not None:
            self['range'] = _v
        _v = arg.pop('rangemode', None)
        _v = rangemode if rangemode is not None else _v
        if _v is not None:
            self['rangemode'] = _v
        _v = arg.pop('separatethousands', None)
        _v = separatethousands if separatethousands is not None else _v
        if _v is not None:
            self['separatethousands'] = _v
        _v = arg.pop('showexponent', None)
        _v = showexponent if showexponent is not None else _v
        if _v is not None:
            self['showexponent'] = _v
        _v = arg.pop('showgrid', None)
        _v = showgrid if showgrid is not None else _v
        if _v is not None:
            self['showgrid'] = _v
        _v = arg.pop('showline', None)
        _v = showline if showline is not None else _v
        if _v is not None:
            self['showline'] = _v
        _v = arg.pop('showticklabels', None)
        _v = showticklabels if showticklabels is not None else _v
        if _v is not None:
            self['showticklabels'] = _v
        _v = arg.pop('showtickprefix', None)
        _v = showtickprefix if showtickprefix is not None else _v
        if _v is not None:
            self['showtickprefix'] = _v
        _v = arg.pop('showticksuffix', None)
        _v = showticksuffix if showticksuffix is not None else _v
        if _v is not None:
            self['showticksuffix'] = _v
        _v = arg.pop('side', None)
        _v = side if side is not None else _v
        if _v is not None:
            self['side'] = _v
        _v = arg.pop('tick0', None)
        _v = tick0 if tick0 is not None else _v
        if _v is not None:
            self['tick0'] = _v
        _v = arg.pop('tickangle', None)
        _v = tickangle if tickangle is not None else _v
        if _v is not None:
            self['tickangle'] = _v
        _v = arg.pop('tickcolor', None)
        _v = tickcolor if tickcolor is not None else _v
        if _v is not None:
            self['tickcolor'] = _v
        _v = arg.pop('tickfont', None)
        _v = tickfont if tickfont is not None else _v
        if _v is not None:
            self['tickfont'] = _v
        _v = arg.pop('tickformat', None)
        _v = tickformat if tickformat is not None else _v
        if _v is not None:
            self['tickformat'] = _v
        _v = arg.pop('tickformatstops', None)
        _v = tickformatstops if tickformatstops is not None else _v
        if _v is not None:
            self['tickformatstops'] = _v
        _v = arg.pop('tickformatstopdefaults', None)
        _v = tickformatstopdefaults if tickformatstopdefaults is not None else _v
        if _v is not None:
            self['tickformatstopdefaults'] = _v
        _v = arg.pop('ticklabelstep', None)
        _v = ticklabelstep if ticklabelstep is not None else _v
        if _v is not None:
            self['ticklabelstep'] = _v
        _v = arg.pop('ticklen', None)
        _v = ticklen if ticklen is not None else _v
        if _v is not None:
            self['ticklen'] = _v
        _v = arg.pop('tickmode', None)
        _v = tickmode if tickmode is not None else _v
        if _v is not None:
            self['tickmode'] = _v
        _v = arg.pop('tickprefix', None)
        _v = tickprefix if tickprefix is not None else _v
        if _v is not None:
            self['tickprefix'] = _v
        _v = arg.pop('ticks', None)
        _v = ticks if ticks is not None else _v
        if _v is not None:
            self['ticks'] = _v
        _v = arg.pop('ticksuffix', None)
        _v = ticksuffix if ticksuffix is not None else _v
        if _v is not None:
            self['ticksuffix'] = _v
        _v = arg.pop('ticktext', None)
        _v = ticktext if ticktext is not None else _v
        if _v is not None:
            self['ticktext'] = _v
        _v = arg.pop('ticktextsrc', None)
        _v = ticktextsrc if ticktextsrc is not None else _v
        if _v is not None:
            self['ticktextsrc'] = _v
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
        _v = arg.pop('title', None)
        _v = title if title is not None else _v
        if _v is not None:
            self['title'] = _v
        _v = arg.pop('titlefont', None)
        _v = titlefont if titlefont is not None else _v
        if _v is not None:
            self['titlefont'] = _v
        _v = arg.pop('type', None)
        _v = type if type is not None else _v
        if _v is not None:
            self['type'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class ColorBar(_BaseTraceHierarchyType):
    _parent_path_str = 'surface'
    _path_str = 'surface.colorbar'
    _valid_props = {'bgcolor', 'bordercolor', 'borderwidth', 'dtick', 'exponentformat', 'labelalias', 'len', 'lenmode', 'minexponent', 'nticks', 'orientation', 'outlinecolor', 'outlinewidth', 'separatethousands', 'showexponent', 'showticklabels', 'showtickprefix', 'showticksuffix', 'thickness', 'thicknessmode', 'tick0', 'tickangle', 'tickcolor', 'tickfont', 'tickformat', 'tickformatstopdefaults', 'tickformatstops', 'ticklabeloverflow', 'ticklabelposition', 'ticklabelstep', 'ticklen', 'tickmode', 'tickprefix', 'ticks', 'ticksuffix', 'ticktext', 'ticktextsrc', 'tickvals', 'tickvalssrc', 'tickwidth', 'title', 'titlefont', 'titleside', 'x', 'xanchor', 'xpad', 'xref', 'y', 'yanchor', 'ypad', 'yref'}

    @property
    def bgcolor(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the color of padded area.\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['bgcolor'] = val

    @property
    def bordercolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the axis line color.\n\n        The 'bordercolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bordercolor']

    @bordercolor.setter
    def bordercolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['bordercolor'] = val

    @property
    def borderwidth(self):
        if False:
            return 10
        "\n        Sets the width (in px) or the border enclosing this color bar.\n\n        The 'borderwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['borderwidth']

    @borderwidth.setter
    def borderwidth(self, val):
        if False:
            print('Hello World!')
        self['borderwidth'] = val

    @property
    def dtick(self):
        if False:
            return 10
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
            print('Hello World!')
        '\n        Determines a formatting rule for the tick exponents. For\n        example, consider the number 1,000,000,000. If "none", it\n        appears as 1,000,000,000. If "e", 1e+9. If "E", 1E+9. If\n        "power", 1x10^9 (with 9 in a super script). If "SI", 1G. If\n        "B", 1B.\n\n        The \'exponentformat\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'none\', \'e\', \'E\', \'power\', \'SI\', \'B\']\n\n        Returns\n        -------\n        Any\n        '
        return self['exponentformat']

    @exponentformat.setter
    def exponentformat(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['exponentformat'] = val

    @property
    def labelalias(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Replacement text for specific tick or hover labels. For example\n        using {US: 'USA', CA: 'Canada'} changes US to USA and CA to\n        Canada. The labels we would have shown must match the keys\n        exactly, after adding any tickprefix or ticksuffix. For\n        negative numbers the minus sign symbol used (U+2212) is wider\n        than the regular ascii dash. That means you need to use −1\n        instead of -1. labelalias can be used with any axis type, and\n        both keys (if needed) and values (if desired) can include html-\n        like tags or MathJax.\n\n        The 'labelalias' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['labelalias']

    @labelalias.setter
    def labelalias(self, val):
        if False:
            while True:
                i = 10
        self['labelalias'] = val

    @property
    def len(self):
        if False:
            print('Hello World!')
        "\n        Sets the length of the color bar This measure excludes the\n        padding of both ends. That is, the color bar length is this\n        length minus the padding on both ends.\n\n        The 'len' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['len']

    @len.setter
    def len(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['len'] = val

    @property
    def lenmode(self):
        if False:
            print('Hello World!')
        '\n        Determines whether this color bar\'s length (i.e. the measure in\n        the color variation direction) is set in units of plot\n        "fraction" or in *pixels. Use `len` to set the value.\n\n        The \'lenmode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'fraction\', \'pixels\']\n\n        Returns\n        -------\n        Any\n        '
        return self['lenmode']

    @lenmode.setter
    def lenmode(self, val):
        if False:
            return 10
        self['lenmode'] = val

    @property
    def minexponent(self):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        '\n        Specifies the maximum number of ticks for the particular axis.\n        The actual number of ticks will be chosen automatically to be\n        less than or equal to `nticks`. Has an effect only if\n        `tickmode` is set to "auto".\n\n        The \'nticks\' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        '
        return self['nticks']

    @nticks.setter
    def nticks(self, val):
        if False:
            return 10
        self['nticks'] = val

    @property
    def orientation(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the orientation of the colorbar.\n\n        The 'orientation' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['h', 'v']\n\n        Returns\n        -------\n        Any\n        "
        return self['orientation']

    @orientation.setter
    def orientation(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['orientation'] = val

    @property
    def outlinecolor(self):
        if False:
            while True:
                i = 10
        "\n        Sets the axis line color.\n\n        The 'outlinecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['outlinecolor']

    @outlinecolor.setter
    def outlinecolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['outlinecolor'] = val

    @property
    def outlinewidth(self):
        if False:
            return 10
        "\n        Sets the width (in px) of the axis line.\n\n        The 'outlinewidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['outlinewidth']

    @outlinewidth.setter
    def outlinewidth(self, val):
        if False:
            i = 10
            return i + 15
        self['outlinewidth'] = val

    @property
    def separatethousands(self):
        if False:
            while True:
                i = 10
        '\n        If "true", even 4-digit integers are separated\n\n        The \'separatethousands\' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        '
        return self['separatethousands']

    @separatethousands.setter
    def separatethousands(self, val):
        if False:
            print('Hello World!')
        self['separatethousands'] = val

    @property
    def showexponent(self):
        if False:
            while True:
                i = 10
        '\n        If "all", all exponents are shown besides their significands.\n        If "first", only the exponent of the first tick is shown. If\n        "last", only the exponent of the last tick is shown. If "none",\n        no exponents appear.\n\n        The \'showexponent\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'all\', \'first\', \'last\', \'none\']\n\n        Returns\n        -------\n        Any\n        '
        return self['showexponent']

    @showexponent.setter
    def showexponent(self, val):
        if False:
            i = 10
            return i + 15
        self['showexponent'] = val

    @property
    def showticklabels(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not the tick labels are drawn.\n\n        The 'showticklabels' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showticklabels']

    @showticklabels.setter
    def showticklabels(self, val):
        if False:
            print('Hello World!')
        self['showticklabels'] = val

    @property
    def showtickprefix(self):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
        self['showticksuffix'] = val

    @property
    def thickness(self):
        if False:
            print('Hello World!')
        "\n        Sets the thickness of the color bar This measure excludes the\n        size of the padding, ticks and labels.\n\n        The 'thickness' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['thickness']

    @thickness.setter
    def thickness(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['thickness'] = val

    @property
    def thicknessmode(self):
        if False:
            print('Hello World!')
        '\n        Determines whether this color bar\'s thickness (i.e. the measure\n        in the constant color direction) is set in units of plot\n        "fraction" or in "pixels". Use `thickness` to set the value.\n\n        The \'thicknessmode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'fraction\', \'pixels\']\n\n        Returns\n        -------\n        Any\n        '
        return self['thicknessmode']

    @thicknessmode.setter
    def thicknessmode(self, val):
        if False:
            print('Hello World!')
        self['thicknessmode'] = val

    @property
    def tick0(self):
        if False:
            while True:
                i = 10
        '\n        Sets the placement of the first tick on this axis. Use with\n        `dtick`. If the axis `type` is "log", then you must take the\n        log of your starting tick (e.g. to set the starting tick to\n        100, set the `tick0` to 2) except when `dtick`=*L<f>* (see\n        `dtick` for more info). If the axis `type` is "date", it should\n        be a date string, like date data. If the axis `type` is\n        "category", it should be a number, using the scale where each\n        category is assigned a serial number from zero in the order it\n        appears.\n\n        The \'tick0\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['tick0']

    @tick0.setter
    def tick0(self, val):
        if False:
            return 10
        self['tick0'] = val

    @property
    def tickangle(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the angle of the tick labels with respect to the\n        horizontal. For example, a `tickangle` of -90 draws the tick\n        labels vertically.\n\n        The 'tickangle' property is a angle (in degrees) that may be\n        specified as a number between -180 and 180.\n        Numeric values outside this range are converted to the equivalent value\n        (e.g. 270 is converted to -90).\n\n        Returns\n        -------\n        int|float\n        "
        return self['tickangle']

    @tickangle.setter
    def tickangle(self, val):
        if False:
            while True:
                i = 10
        self['tickangle'] = val

    @property
    def tickcolor(self):
        if False:
            return 10
        "\n        Sets the tick color.\n\n        The 'tickcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['tickcolor']

    @tickcolor.setter
    def tickcolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickcolor'] = val

    @property
    def tickfont(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the color bar\'s tick label font\n\n        The \'tickfont\' property is an instance of Tickfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.surface.colorbar.Tickfont`\n          - A dict of string/value properties that will be passed\n            to the Tickfont constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.surface.colorbar.Tickfont\n        '
        return self['tickfont']

    @tickfont.setter
    def tickfont(self, val):
        if False:
            return 10
        self['tickfont'] = val

    @property
    def tickformat(self):
        if False:
            while True:
                i = 10
        '\n        Sets the tick label formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display "09~15~23.46"\n\n        The \'tickformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['tickformat']

    @tickformat.setter
    def tickformat(self, val):
        if False:
            print('Hello World!')
        self['tickformat'] = val

    @property
    def tickformatstops(self):
        if False:
            print('Hello World!')
        '\n        The \'tickformatstops\' property is a tuple of instances of\n        Tickformatstop that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.surface.colorbar.Tickformatstop\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Tickformatstop constructor\n\n            Supported dict properties:\n\n                dtickrange\n                    range [*min*, *max*], where "min", "max" -\n                    dtick values which describe some zoom level, it\n                    is possible to omit "min" or "max" value by\n                    passing "null"\n                enabled\n                    Determines whether or not this stop is used. If\n                    `false`, this stop is ignored even within its\n                    `dtickrange`.\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n                value\n                    string - dtickformat for described zoom level,\n                    the same as "tickformat"\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.surface.colorbar.Tickformatstop]\n        '
        return self['tickformatstops']

    @tickformatstops.setter
    def tickformatstops(self, val):
        if False:
            return 10
        self['tickformatstops'] = val

    @property
    def tickformatstopdefaults(self):
        if False:
            while True:
                i = 10
        "\n        When used in a template (as\n        layout.template.data.surface.colorbar.tickformatstopdefaults),\n        sets the default property values to use for elements of\n        surface.colorbar.tickformatstops\n\n        The 'tickformatstopdefaults' property is an instance of Tickformatstop\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.surface.colorbar.Tickformatstop`\n          - A dict of string/value properties that will be passed\n            to the Tickformatstop constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.surface.colorbar.Tickformatstop\n        "
        return self['tickformatstopdefaults']

    @tickformatstopdefaults.setter
    def tickformatstopdefaults(self, val):
        if False:
            return 10
        self['tickformatstopdefaults'] = val

    @property
    def ticklabeloverflow(self):
        if False:
            print('Hello World!')
        "\n        Determines how we handle tick labels that would overflow either\n        the graph div or the domain of the axis. The default value for\n        inside tick labels is *hide past domain*. In other cases the\n        default is *hide past div*.\n\n        The 'ticklabeloverflow' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['allow', 'hide past div', 'hide past domain']\n\n        Returns\n        -------\n        Any\n        "
        return self['ticklabeloverflow']

    @ticklabeloverflow.setter
    def ticklabeloverflow(self, val):
        if False:
            print('Hello World!')
        self['ticklabeloverflow'] = val

    @property
    def ticklabelposition(self):
        if False:
            return 10
        '\n        Determines where tick labels are drawn relative to the ticks.\n        Left and right options are used when `orientation` is "h", top\n        and bottom when `orientation` is "v".\n\n        The \'ticklabelposition\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'outside\', \'inside\', \'outside top\', \'inside top\',\n                \'outside left\', \'inside left\', \'outside right\', \'inside\n                right\', \'outside bottom\', \'inside bottom\']\n\n        Returns\n        -------\n        Any\n        '
        return self['ticklabelposition']

    @ticklabelposition.setter
    def ticklabelposition(self, val):
        if False:
            while True:
                i = 10
        self['ticklabelposition'] = val

    @property
    def ticklabelstep(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the spacing between tick labels as compared to the spacing\n        between ticks. A value of 1 (default) means each tick gets a\n        label. A value of 2 means shows every 2nd label. A larger value\n        n means only every nth tick is labeled. `tick0` determines\n        which labels are shown. Not implemented for axes with `type`\n        "log" or "multicategory", or when `tickmode` is "array".\n\n        The \'ticklabelstep\' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [1, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        '
        return self['ticklabelstep']

    @ticklabelstep.setter
    def ticklabelstep(self, val):
        if False:
            while True:
                i = 10
        self['ticklabelstep'] = val

    @property
    def ticklen(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the tick length (in px).\n\n        The 'ticklen' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['ticklen']

    @ticklen.setter
    def ticklen(self, val):
        if False:
            print('Hello World!')
        self['ticklen'] = val

    @property
    def tickmode(self):
        if False:
            print('Hello World!')
        '\n        Sets the tick mode for this axis. If "auto", the number of\n        ticks is set via `nticks`. If "linear", the placement of the\n        ticks is determined by a starting position `tick0` and a tick\n        step `dtick` ("linear" is the default value if `tick0` and\n        `dtick` are provided). If "array", the placement of the ticks\n        is set via `tickvals` and the tick text is `ticktext`. ("array"\n        is the default value if `tickvals` is provided).\n\n        The \'tickmode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'linear\', \'array\']\n\n        Returns\n        -------\n        Any\n        '
        return self['tickmode']

    @tickmode.setter
    def tickmode(self, val):
        if False:
            while True:
                i = 10
        self['tickmode'] = val

    @property
    def tickprefix(self):
        if False:
            while True:
                i = 10
        "\n        Sets a tick label prefix.\n\n        The 'tickprefix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['tickprefix']

    @tickprefix.setter
    def tickprefix(self, val):
        if False:
            return 10
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
            return 10
        self['ticks'] = val

    @property
    def ticksuffix(self):
        if False:
            while True:
                i = 10
        "\n        Sets a tick label suffix.\n\n        The 'ticksuffix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['ticksuffix']

    @ticksuffix.setter
    def ticksuffix(self, val):
        if False:
            return 10
        self['ticksuffix'] = val

    @property
    def ticktext(self):
        if False:
            while True:
                i = 10
        '\n        Sets the text displayed at the ticks position via `tickvals`.\n        Only has an effect if `tickmode` is set to "array". Used with\n        `tickvals`.\n\n        The \'ticktext\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['ticktext']

    @ticktext.setter
    def ticktext(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['ticktext'] = val

    @property
    def ticktextsrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `ticktext`.\n\n        The 'ticktextsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['ticktextsrc']

    @ticktextsrc.setter
    def ticktextsrc(self, val):
        if False:
            return 10
        self['ticktextsrc'] = val

    @property
    def tickvals(self):
        if False:
            print('Hello World!')
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
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `tickvals`.\n\n        The 'tickvalssrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['tickvalssrc']

    @tickvalssrc.setter
    def tickvalssrc(self, val):
        if False:
            print('Hello World!')
        self['tickvalssrc'] = val

    @property
    def tickwidth(self):
        if False:
            while True:
                i = 10
        "\n        Sets the tick width (in px).\n\n        The 'tickwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['tickwidth']

    @tickwidth.setter
    def tickwidth(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickwidth'] = val

    @property
    def title(self):
        if False:
            i = 10
            return i + 15
        '\n        The \'title\' property is an instance of Title\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.surface.colorbar.Title`\n          - A dict of string/value properties that will be passed\n            to the Title constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this color bar\'s title font. Note that the\n                    title\'s font used to be set by the now\n                    deprecated `titlefont` attribute.\n                side\n                    Determines the location of color bar\'s title\n                    with respect to the color bar. Defaults to\n                    "top" when `orientation` if "v" and  defaults\n                    to "right" when `orientation` if "h". Note that\n                    the title\'s location used to be set by the now\n                    deprecated `titleside` attribute.\n                text\n                    Sets the title of the color bar. Note that\n                    before the existence of `title.text`, the\n                    title\'s contents used to be defined as the\n                    `title` attribute itself. This behavior has\n                    been deprecated.\n\n        Returns\n        -------\n        plotly.graph_objs.surface.colorbar.Title\n        '
        return self['title']

    @title.setter
    def title(self, val):
        if False:
            return 10
        self['title'] = val

    @property
    def titlefont(self):
        if False:
            return 10
        '\n        Deprecated: Please use surface.colorbar.title.font instead.\n        Sets this color bar\'s title font. Note that the title\'s font\n        used to be set by the now deprecated `titlefont` attribute.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.surface.colorbar.title.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n\n        '
        return self['titlefont']

    @titlefont.setter
    def titlefont(self, val):
        if False:
            while True:
                i = 10
        self['titlefont'] = val

    @property
    def titleside(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deprecated: Please use surface.colorbar.title.side instead.\n        Determines the location of color bar\'s title with respect to\n        the color bar. Defaults to "top" when `orientation` if "v" and\n        defaults to "right" when `orientation` if "h". Note that the\n        title\'s location used to be set by the now deprecated\n        `titleside` attribute.\n\n        The \'side\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'right\', \'top\', \'bottom\']\n\n        Returns\n        -------\n\n        '
        return self['titleside']

    @titleside.setter
    def titleside(self, val):
        if False:
            i = 10
            return i + 15
        self['titleside'] = val

    @property
    def x(self):
        if False:
            return 10
        '\n        Sets the x position with respect to `xref` of the color bar (in\n        plot fraction). When `xref` is "paper", defaults to 1.02 when\n        `orientation` is "v" and 0.5 when `orientation` is "h". When\n        `xref` is "container", defaults to 1 when `orientation` is "v"\n        and 0.5 when `orientation` is "h". Must be between 0 and 1 if\n        `xref` is "container" and between "-2" and 3 if `xref` is\n        "paper".\n\n        The \'x\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            return 10
        self['x'] = val

    @property
    def xanchor(self):
        if False:
            print('Hello World!')
        '\n        Sets this color bar\'s horizontal position anchor. This anchor\n        binds the `x` position to the "left", "center" or "right" of\n        the color bar. Defaults to "left" when `orientation` is "v" and\n        "center" when `orientation` is "h".\n\n        The \'xanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'left\', \'center\', \'right\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xanchor']

    @xanchor.setter
    def xanchor(self, val):
        if False:
            print('Hello World!')
        self['xanchor'] = val

    @property
    def xpad(self):
        if False:
            return 10
        "\n        Sets the amount of padding (in px) along the x direction.\n\n        The 'xpad' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['xpad']

    @xpad.setter
    def xpad(self, val):
        if False:
            print('Hello World!')
        self['xpad'] = val

    @property
    def xref(self):
        if False:
            while True:
                i = 10
        '\n        Sets the container `x` refers to. "container" spans the entire\n        `width` of the plot. "paper" refers to the width of the\n        plotting area only.\n\n        The \'xref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'container\', \'paper\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xref']

    @xref.setter
    def xref(self, val):
        if False:
            return 10
        self['xref'] = val

    @property
    def y(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the y position with respect to `yref` of the color bar (in\n        plot fraction). When `yref` is "paper", defaults to 0.5 when\n        `orientation` is "v" and 1.02 when `orientation` is "h". When\n        `yref` is "container", defaults to 0.5 when `orientation` is\n        "v" and 1 when `orientation` is "h". Must be between 0 and 1 if\n        `yref` is "container" and between "-2" and 3 if `yref` is\n        "paper".\n\n        The \'y\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            i = 10
            return i + 15
        self['y'] = val

    @property
    def yanchor(self):
        if False:
            print('Hello World!')
        '\n        Sets this color bar\'s vertical position anchor This anchor\n        binds the `y` position to the "top", "middle" or "bottom" of\n        the color bar. Defaults to "middle" when `orientation` is "v"\n        and "bottom" when `orientation` is "h".\n\n        The \'yanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'top\', \'middle\', \'bottom\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yanchor']

    @yanchor.setter
    def yanchor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['yanchor'] = val

    @property
    def ypad(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the amount of padding (in px) along the y direction.\n\n        The 'ypad' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['ypad']

    @ypad.setter
    def ypad(self, val):
        if False:
            return 10
        self['ypad'] = val

    @property
    def yref(self):
        if False:
            return 10
        '\n        Sets the container `y` refers to. "container" spans the entire\n        `height` of the plot. "paper" refers to the height of the\n        plotting area only.\n\n        The \'yref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'container\', \'paper\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yref']

    @yref.setter
    def yref(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['yref'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        bgcolor\n            Sets the color of padded area.\n        bordercolor\n            Sets the axis line color.\n        borderwidth\n            Sets the width (in px) or the border enclosing this\n            color bar.\n        dtick\n            Sets the step in-between ticks on this axis. Use with\n            `tick0`. Must be a positive number, or special strings\n            available to "log" and "date" axes. If the axis `type`\n            is "log", then ticks are set every 10^(n*dtick) where n\n            is the tick number. For example, to set a tick mark at\n            1, 10, 100, 1000, ... set dtick to 1. To set tick marks\n            at 1, 100, 10000, ... set dtick to 2. To set tick marks\n            at 1, 5, 25, 125, 625, 3125, ... set dtick to\n            log_10(5), or 0.69897000433. "log" has several special\n            values; "L<f>", where `f` is a positive number, gives\n            ticks linearly spaced in value (but not position). For\n            example `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n            at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus\n            small digits between, use "D1" (all digits) or "D2"\n            (only 2 and 5). `tick0` is ignored for "D1" and "D2".\n            If the axis `type` is "date", then you must convert the\n            time to milliseconds. For example, to set the interval\n            between ticks to one day, set `dtick` to 86400000.0.\n            "date" also has special values "M<n>" gives ticks\n            spaced by a number of months. `n` must be a positive\n            integer. To set ticks on the 15th of every third month,\n            set `tick0` to "2000-01-15" and `dtick` to "M3". To set\n            ticks every 4 years, set `dtick` to "M48"\n        exponentformat\n            Determines a formatting rule for the tick exponents.\n            For example, consider the number 1,000,000,000. If\n            "none", it appears as 1,000,000,000. If "e", 1e+9. If\n            "E", 1E+9. If "power", 1x10^9 (with 9 in a super\n            script). If "SI", 1G. If "B", 1B.\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        len\n            Sets the length of the color bar This measure excludes\n            the padding of both ends. That is, the color bar length\n            is this length minus the padding on both ends.\n        lenmode\n            Determines whether this color bar\'s length (i.e. the\n            measure in the color variation direction) is set in\n            units of plot "fraction" or in *pixels. Use `len` to\n            set the value.\n        minexponent\n            Hide SI prefix for 10^n if |n| is below this number.\n            This only has an effect when `tickformat` is "SI" or\n            "B".\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        orientation\n            Sets the orientation of the colorbar.\n        outlinecolor\n            Sets the axis line color.\n        outlinewidth\n            Sets the width (in px) of the axis line.\n        separatethousands\n            If "true", even 4-digit integers are separated\n        showexponent\n            If "all", all exponents are shown besides their\n            significands. If "first", only the exponent of the\n            first tick is shown. If "last", only the exponent of\n            the last tick is shown. If "none", no exponents appear.\n        showticklabels\n            Determines whether or not the tick labels are drawn.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        thickness\n            Sets the thickness of the color bar This measure\n            excludes the size of the padding, ticks and labels.\n        thicknessmode\n            Determines whether this color bar\'s thickness (i.e. the\n            measure in the constant color direction) is set in\n            units of plot "fraction" or in "pixels". Use\n            `thickness` to set the value.\n        tick0\n            Sets the placement of the first tick on this axis. Use\n            with `dtick`. If the axis `type` is "log", then you\n            must take the log of your starting tick (e.g. to set\n            the starting tick to 100, set the `tick0` to 2) except\n            when `dtick`=*L<f>* (see `dtick` for more info). If the\n            axis `type` is "date", it should be a date string, like\n            date data. If the axis `type` is "category", it should\n            be a number, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        tickangle\n            Sets the angle of the tick labels with respect to the\n            horizontal. For example, a `tickangle` of -90 draws the\n            tick labels vertically.\n        tickcolor\n            Sets the tick color.\n        tickfont\n            Sets the color bar\'s tick label font\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        tickformatstops\n            A tuple of :class:`plotly.graph_objects.surface.colorba\n            r.Tickformatstop` instances or dicts with compatible\n            properties\n        tickformatstopdefaults\n            When used in a template (as layout.template.data.surfac\n            e.colorbar.tickformatstopdefaults), sets the default\n            property values to use for elements of\n            surface.colorbar.tickformatstops\n        ticklabeloverflow\n            Determines how we handle tick labels that would\n            overflow either the graph div or the domain of the\n            axis. The default value for inside tick labels is *hide\n            past domain*. In other cases the default is *hide past\n            div*.\n        ticklabelposition\n            Determines where tick labels are drawn relative to the\n            ticks. Left and right options are used when\n            `orientation` is "h", top and bottom when `orientation`\n            is "v".\n        ticklabelstep\n            Sets the spacing between tick labels as compared to the\n            spacing between ticks. A value of 1 (default) means\n            each tick gets a label. A value of 2 means shows every\n            2nd label. A larger value n means only every nth tick\n            is labeled. `tick0` determines which labels are shown.\n            Not implemented for axes with `type` "log" or\n            "multicategory", or when `tickmode` is "array".\n        ticklen\n            Sets the tick length (in px).\n        tickmode\n            Sets the tick mode for this axis. If "auto", the number\n            of ticks is set via `nticks`. If "linear", the\n            placement of the ticks is determined by a starting\n            position `tick0` and a tick step `dtick` ("linear" is\n            the default value if `tick0` and `dtick` are provided).\n            If "array", the placement of the ticks is set via\n            `tickvals` and the tick text is `ticktext`. ("array" is\n            the default value if `tickvals` is provided).\n        tickprefix\n            Sets a tick label prefix.\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        ticksuffix\n            Sets a tick label suffix.\n        ticktext\n            Sets the text displayed at the ticks position via\n            `tickvals`. Only has an effect if `tickmode` is set to\n            "array". Used with `tickvals`.\n        ticktextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `ticktext`.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n        title\n            :class:`plotly.graph_objects.surface.colorbar.Title`\n            instance or dict with compatible properties\n        titlefont\n            Deprecated: Please use surface.colorbar.title.font\n            instead. Sets this color bar\'s title font. Note that\n            the title\'s font used to be set by the now deprecated\n            `titlefont` attribute.\n        titleside\n            Deprecated: Please use surface.colorbar.title.side\n            instead. Determines the location of color bar\'s title\n            with respect to the color bar. Defaults to "top" when\n            `orientation` if "v" and  defaults to "right" when\n            `orientation` if "h". Note that the title\'s location\n            used to be set by the now deprecated `titleside`\n            attribute.\n        x\n            Sets the x position with respect to `xref` of the color\n            bar (in plot fraction). When `xref` is "paper",\n            defaults to 1.02 when `orientation` is "v" and 0.5 when\n            `orientation` is "h". When `xref` is "container",\n            defaults to 1 when `orientation` is "v" and 0.5 when\n            `orientation` is "h". Must be between 0 and 1 if `xref`\n            is "container" and between "-2" and 3 if `xref` is\n            "paper".\n        xanchor\n            Sets this color bar\'s horizontal position anchor. This\n            anchor binds the `x` position to the "left", "center"\n            or "right" of the color bar. Defaults to "left" when\n            `orientation` is "v" and "center" when `orientation` is\n            "h".\n        xpad\n            Sets the amount of padding (in px) along the x\n            direction.\n        xref\n            Sets the container `x` refers to. "container" spans the\n            entire `width` of the plot. "paper" refers to the width\n            of the plotting area only.\n        y\n            Sets the y position with respect to `yref` of the color\n            bar (in plot fraction). When `yref` is "paper",\n            defaults to 0.5 when `orientation` is "v" and 1.02 when\n            `orientation` is "h". When `yref` is "container",\n            defaults to 0.5 when `orientation` is "v" and 1 when\n            `orientation` is "h". Must be between 0 and 1 if `yref`\n            is "container" and between "-2" and 3 if `yref` is\n            "paper".\n        yanchor\n            Sets this color bar\'s vertical position anchor This\n            anchor binds the `y` position to the "top", "middle" or\n            "bottom" of the color bar. Defaults to "middle" when\n            `orientation` is "v" and "bottom" when `orientation` is\n            "h".\n        ypad\n            Sets the amount of padding (in px) along the y\n            direction.\n        yref\n            Sets the container `y` refers to. "container" spans the\n            entire `height` of the plot. "paper" refers to the\n            height of the plotting area only.\n        '
    _mapped_properties = {'titlefont': ('title', 'font'), 'titleside': ('title', 'side')}

    def __init__(self, arg=None, bgcolor=None, bordercolor=None, borderwidth=None, dtick=None, exponentformat=None, labelalias=None, len=None, lenmode=None, minexponent=None, nticks=None, orientation=None, outlinecolor=None, outlinewidth=None, separatethousands=None, showexponent=None, showticklabels=None, showtickprefix=None, showticksuffix=None, thickness=None, thicknessmode=None, tick0=None, tickangle=None, tickcolor=None, tickfont=None, tickformat=None, tickformatstops=None, tickformatstopdefaults=None, ticklabeloverflow=None, ticklabelposition=None, ticklabelstep=None, ticklen=None, tickmode=None, tickprefix=None, ticks=None, ticksuffix=None, ticktext=None, ticktextsrc=None, tickvals=None, tickvalssrc=None, tickwidth=None, title=None, titlefont=None, titleside=None, x=None, xanchor=None, xpad=None, xref=None, y=None, yanchor=None, ypad=None, yref=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new ColorBar object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.surface.ColorBar`\n        bgcolor\n            Sets the color of padded area.\n        bordercolor\n            Sets the axis line color.\n        borderwidth\n            Sets the width (in px) or the border enclosing this\n            color bar.\n        dtick\n            Sets the step in-between ticks on this axis. Use with\n            `tick0`. Must be a positive number, or special strings\n            available to "log" and "date" axes. If the axis `type`\n            is "log", then ticks are set every 10^(n*dtick) where n\n            is the tick number. For example, to set a tick mark at\n            1, 10, 100, 1000, ... set dtick to 1. To set tick marks\n            at 1, 100, 10000, ... set dtick to 2. To set tick marks\n            at 1, 5, 25, 125, 625, 3125, ... set dtick to\n            log_10(5), or 0.69897000433. "log" has several special\n            values; "L<f>", where `f` is a positive number, gives\n            ticks linearly spaced in value (but not position). For\n            example `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n            at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus\n            small digits between, use "D1" (all digits) or "D2"\n            (only 2 and 5). `tick0` is ignored for "D1" and "D2".\n            If the axis `type` is "date", then you must convert the\n            time to milliseconds. For example, to set the interval\n            between ticks to one day, set `dtick` to 86400000.0.\n            "date" also has special values "M<n>" gives ticks\n            spaced by a number of months. `n` must be a positive\n            integer. To set ticks on the 15th of every third month,\n            set `tick0` to "2000-01-15" and `dtick` to "M3". To set\n            ticks every 4 years, set `dtick` to "M48"\n        exponentformat\n            Determines a formatting rule for the tick exponents.\n            For example, consider the number 1,000,000,000. If\n            "none", it appears as 1,000,000,000. If "e", 1e+9. If\n            "E", 1E+9. If "power", 1x10^9 (with 9 in a super\n            script). If "SI", 1G. If "B", 1B.\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        len\n            Sets the length of the color bar This measure excludes\n            the padding of both ends. That is, the color bar length\n            is this length minus the padding on both ends.\n        lenmode\n            Determines whether this color bar\'s length (i.e. the\n            measure in the color variation direction) is set in\n            units of plot "fraction" or in *pixels. Use `len` to\n            set the value.\n        minexponent\n            Hide SI prefix for 10^n if |n| is below this number.\n            This only has an effect when `tickformat` is "SI" or\n            "B".\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        orientation\n            Sets the orientation of the colorbar.\n        outlinecolor\n            Sets the axis line color.\n        outlinewidth\n            Sets the width (in px) of the axis line.\n        separatethousands\n            If "true", even 4-digit integers are separated\n        showexponent\n            If "all", all exponents are shown besides their\n            significands. If "first", only the exponent of the\n            first tick is shown. If "last", only the exponent of\n            the last tick is shown. If "none", no exponents appear.\n        showticklabels\n            Determines whether or not the tick labels are drawn.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        thickness\n            Sets the thickness of the color bar This measure\n            excludes the size of the padding, ticks and labels.\n        thicknessmode\n            Determines whether this color bar\'s thickness (i.e. the\n            measure in the constant color direction) is set in\n            units of plot "fraction" or in "pixels". Use\n            `thickness` to set the value.\n        tick0\n            Sets the placement of the first tick on this axis. Use\n            with `dtick`. If the axis `type` is "log", then you\n            must take the log of your starting tick (e.g. to set\n            the starting tick to 100, set the `tick0` to 2) except\n            when `dtick`=*L<f>* (see `dtick` for more info). If the\n            axis `type` is "date", it should be a date string, like\n            date data. If the axis `type` is "category", it should\n            be a number, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        tickangle\n            Sets the angle of the tick labels with respect to the\n            horizontal. For example, a `tickangle` of -90 draws the\n            tick labels vertically.\n        tickcolor\n            Sets the tick color.\n        tickfont\n            Sets the color bar\'s tick label font\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        tickformatstops\n            A tuple of :class:`plotly.graph_objects.surface.colorba\n            r.Tickformatstop` instances or dicts with compatible\n            properties\n        tickformatstopdefaults\n            When used in a template (as layout.template.data.surfac\n            e.colorbar.tickformatstopdefaults), sets the default\n            property values to use for elements of\n            surface.colorbar.tickformatstops\n        ticklabeloverflow\n            Determines how we handle tick labels that would\n            overflow either the graph div or the domain of the\n            axis. The default value for inside tick labels is *hide\n            past domain*. In other cases the default is *hide past\n            div*.\n        ticklabelposition\n            Determines where tick labels are drawn relative to the\n            ticks. Left and right options are used when\n            `orientation` is "h", top and bottom when `orientation`\n            is "v".\n        ticklabelstep\n            Sets the spacing between tick labels as compared to the\n            spacing between ticks. A value of 1 (default) means\n            each tick gets a label. A value of 2 means shows every\n            2nd label. A larger value n means only every nth tick\n            is labeled. `tick0` determines which labels are shown.\n            Not implemented for axes with `type` "log" or\n            "multicategory", or when `tickmode` is "array".\n        ticklen\n            Sets the tick length (in px).\n        tickmode\n            Sets the tick mode for this axis. If "auto", the number\n            of ticks is set via `nticks`. If "linear", the\n            placement of the ticks is determined by a starting\n            position `tick0` and a tick step `dtick` ("linear" is\n            the default value if `tick0` and `dtick` are provided).\n            If "array", the placement of the ticks is set via\n            `tickvals` and the tick text is `ticktext`. ("array" is\n            the default value if `tickvals` is provided).\n        tickprefix\n            Sets a tick label prefix.\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        ticksuffix\n            Sets a tick label suffix.\n        ticktext\n            Sets the text displayed at the ticks position via\n            `tickvals`. Only has an effect if `tickmode` is set to\n            "array". Used with `tickvals`.\n        ticktextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `ticktext`.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n        title\n            :class:`plotly.graph_objects.surface.colorbar.Title`\n            instance or dict with compatible properties\n        titlefont\n            Deprecated: Please use surface.colorbar.title.font\n            instead. Sets this color bar\'s title font. Note that\n            the title\'s font used to be set by the now deprecated\n            `titlefont` attribute.\n        titleside\n            Deprecated: Please use surface.colorbar.title.side\n            instead. Determines the location of color bar\'s title\n            with respect to the color bar. Defaults to "top" when\n            `orientation` if "v" and  defaults to "right" when\n            `orientation` if "h". Note that the title\'s location\n            used to be set by the now deprecated `titleside`\n            attribute.\n        x\n            Sets the x position with respect to `xref` of the color\n            bar (in plot fraction). When `xref` is "paper",\n            defaults to 1.02 when `orientation` is "v" and 0.5 when\n            `orientation` is "h". When `xref` is "container",\n            defaults to 1 when `orientation` is "v" and 0.5 when\n            `orientation` is "h". Must be between 0 and 1 if `xref`\n            is "container" and between "-2" and 3 if `xref` is\n            "paper".\n        xanchor\n            Sets this color bar\'s horizontal position anchor. This\n            anchor binds the `x` position to the "left", "center"\n            or "right" of the color bar. Defaults to "left" when\n            `orientation` is "v" and "center" when `orientation` is\n            "h".\n        xpad\n            Sets the amount of padding (in px) along the x\n            direction.\n        xref\n            Sets the container `x` refers to. "container" spans the\n            entire `width` of the plot. "paper" refers to the width\n            of the plotting area only.\n        y\n            Sets the y position with respect to `yref` of the color\n            bar (in plot fraction). When `yref` is "paper",\n            defaults to 0.5 when `orientation` is "v" and 1.02 when\n            `orientation` is "h". When `yref` is "container",\n            defaults to 0.5 when `orientation` is "v" and 1 when\n            `orientation` is "h". Must be between 0 and 1 if `yref`\n            is "container" and between "-2" and 3 if `yref` is\n            "paper".\n        yanchor\n            Sets this color bar\'s vertical position anchor This\n            anchor binds the `y` position to the "top", "middle" or\n            "bottom" of the color bar. Defaults to "middle" when\n            `orientation` is "v" and "bottom" when `orientation` is\n            "h".\n        ypad\n            Sets the amount of padding (in px) along the y\n            direction.\n        yref\n            Sets the container `y` refers to. "container" spans the\n            entire `height` of the plot. "paper" refers to the\n            height of the plotting area only.\n\n        Returns\n        -------\n        ColorBar\n        '
        super(ColorBar, self).__init__('colorbar')
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
            raise ValueError('The first argument to the plotly.graph_objs.surface.ColorBar\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.surface.ColorBar`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
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
        _v = arg.pop('dtick', None)
        _v = dtick if dtick is not None else _v
        if _v is not None:
            self['dtick'] = _v
        _v = arg.pop('exponentformat', None)
        _v = exponentformat if exponentformat is not None else _v
        if _v is not None:
            self['exponentformat'] = _v
        _v = arg.pop('labelalias', None)
        _v = labelalias if labelalias is not None else _v
        if _v is not None:
            self['labelalias'] = _v
        _v = arg.pop('len', None)
        _v = len if len is not None else _v
        if _v is not None:
            self['len'] = _v
        _v = arg.pop('lenmode', None)
        _v = lenmode if lenmode is not None else _v
        if _v is not None:
            self['lenmode'] = _v
        _v = arg.pop('minexponent', None)
        _v = minexponent if minexponent is not None else _v
        if _v is not None:
            self['minexponent'] = _v
        _v = arg.pop('nticks', None)
        _v = nticks if nticks is not None else _v
        if _v is not None:
            self['nticks'] = _v
        _v = arg.pop('orientation', None)
        _v = orientation if orientation is not None else _v
        if _v is not None:
            self['orientation'] = _v
        _v = arg.pop('outlinecolor', None)
        _v = outlinecolor if outlinecolor is not None else _v
        if _v is not None:
            self['outlinecolor'] = _v
        _v = arg.pop('outlinewidth', None)
        _v = outlinewidth if outlinewidth is not None else _v
        if _v is not None:
            self['outlinewidth'] = _v
        _v = arg.pop('separatethousands', None)
        _v = separatethousands if separatethousands is not None else _v
        if _v is not None:
            self['separatethousands'] = _v
        _v = arg.pop('showexponent', None)
        _v = showexponent if showexponent is not None else _v
        if _v is not None:
            self['showexponent'] = _v
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
        _v = arg.pop('thickness', None)
        _v = thickness if thickness is not None else _v
        if _v is not None:
            self['thickness'] = _v
        _v = arg.pop('thicknessmode', None)
        _v = thicknessmode if thicknessmode is not None else _v
        if _v is not None:
            self['thicknessmode'] = _v
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
        _v = arg.pop('ticklabeloverflow', None)
        _v = ticklabeloverflow if ticklabeloverflow is not None else _v
        if _v is not None:
            self['ticklabeloverflow'] = _v
        _v = arg.pop('ticklabelposition', None)
        _v = ticklabelposition if ticklabelposition is not None else _v
        if _v is not None:
            self['ticklabelposition'] = _v
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
        _v = arg.pop('titleside', None)
        _v = titleside if titleside is not None else _v
        if _v is not None:
            self['titleside'] = _v
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('xanchor', None)
        _v = xanchor if xanchor is not None else _v
        if _v is not None:
            self['xanchor'] = _v
        _v = arg.pop('xpad', None)
        _v = xpad if xpad is not None else _v
        if _v is not None:
            self['xpad'] = _v
        _v = arg.pop('xref', None)
        _v = xref if xref is not None else _v
        if _v is not None:
            self['xref'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('yanchor', None)
        _v = yanchor if yanchor is not None else _v
        if _v is not None:
            self['yanchor'] = _v
        _v = arg.pop('ypad', None)
        _v = ypad if ypad is not None else _v
        if _v is not None:
            self['ypad'] = _v
        _v = arg.pop('yref', None)
        _v = yref if yref is not None else _v
        if _v is not None:
            self['yref'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
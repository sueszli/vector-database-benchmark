from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Baxis(_BaseTraceHierarchyType):
    _parent_path_str = 'carpet'
    _path_str = 'carpet.baxis'
    _valid_props = {'arraydtick', 'arraytick0', 'autorange', 'autotypenumbers', 'categoryarray', 'categoryarraysrc', 'categoryorder', 'cheatertype', 'color', 'dtick', 'endline', 'endlinecolor', 'endlinewidth', 'exponentformat', 'fixedrange', 'gridcolor', 'griddash', 'gridwidth', 'labelalias', 'labelpadding', 'labelprefix', 'labelsuffix', 'linecolor', 'linewidth', 'minexponent', 'minorgridcolor', 'minorgridcount', 'minorgriddash', 'minorgridwidth', 'nticks', 'range', 'rangemode', 'separatethousands', 'showexponent', 'showgrid', 'showline', 'showticklabels', 'showtickprefix', 'showticksuffix', 'smoothing', 'startline', 'startlinecolor', 'startlinewidth', 'tick0', 'tickangle', 'tickfont', 'tickformat', 'tickformatstopdefaults', 'tickformatstops', 'tickmode', 'tickprefix', 'ticksuffix', 'ticktext', 'ticktextsrc', 'tickvals', 'tickvalssrc', 'title', 'titlefont', 'titleoffset', 'type'}

    @property
    def arraydtick(self):
        if False:
            print('Hello World!')
        "\n        The stride between grid lines along the axis\n\n        The 'arraydtick' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [1, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['arraydtick']

    @arraydtick.setter
    def arraydtick(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['arraydtick'] = val

    @property
    def arraytick0(self):
        if False:
            i = 10
            return i + 15
        "\n        The starting index of grid lines along the axis\n\n        The 'arraytick0' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['arraytick0']

    @arraytick0.setter
    def arraytick0(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['arraytick0'] = val

    @property
    def autorange(self):
        if False:
            return 10
        "\n        Determines whether or not the range of this axis is computed in\n        relation to the input data. See `rangemode` for more info. If\n        `range` is provided, then `autorange` is set to False.\n\n        The 'autorange' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, 'reversed']\n\n        Returns\n        -------\n        Any\n        "
        return self['autorange']

    @autorange.setter
    def autorange(self, val):
        if False:
            print('Hello World!')
        self['autorange'] = val

    @property
    def autotypenumbers(self):
        if False:
            print('Hello World!')
        '\n        Using "strict" a numeric string in trace data is not converted\n        to a number. Using *convert types* a numeric string in trace\n        data may be treated as a number during automatic axis `type`\n        detection. Defaults to layout.autotypenumbers.\n\n        The \'autotypenumbers\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'convert types\', \'strict\']\n\n        Returns\n        -------\n        Any\n        '
        return self['autotypenumbers']

    @autotypenumbers.setter
    def autotypenumbers(self, val):
        if False:
            while True:
                i = 10
        self['autotypenumbers'] = val

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
            i = 10
            return i + 15
        self['categoryarray'] = val

    @property
    def categoryarraysrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `categoryarray`.\n\n        The 'categoryarraysrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['categoryarraysrc']

    @categoryarraysrc.setter
    def categoryarraysrc(self, val):
        if False:
            print('Hello World!')
        self['categoryarraysrc'] = val

    @property
    def categoryorder(self):
        if False:
            while True:
                i = 10
        '\n        Specifies the ordering logic for the case of categorical\n        variables. By default, plotly uses "trace", which specifies the\n        order that is present in the data supplied. Set `categoryorder`\n        to *category ascending* or *category descending* if order\n        should be determined by the alphanumerical order of the\n        category names. Set `categoryorder` to "array" to derive the\n        ordering from the attribute `categoryarray`. If a category is\n        not found in the `categoryarray` array, the sorting behavior\n        for that attribute will be identical to the "trace" mode. The\n        unspecified categories will follow the categories in\n        `categoryarray`.\n\n        The \'categoryorder\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'trace\', \'category ascending\', \'category descending\',\n                \'array\']\n\n        Returns\n        -------\n        Any\n        '
        return self['categoryorder']

    @categoryorder.setter
    def categoryorder(self, val):
        if False:
            while True:
                i = 10
        self['categoryorder'] = val

    @property
    def cheatertype(self):
        if False:
            print('Hello World!')
        "\n        The 'cheatertype' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['index', 'value']\n\n        Returns\n        -------\n        Any\n        "
        return self['cheatertype']

    @cheatertype.setter
    def cheatertype(self, val):
        if False:
            while True:
                i = 10
        self['cheatertype'] = val

    @property
    def color(self):
        if False:
            while True:
                i = 10
        "\n        Sets default for all colors associated with this axis all at\n        once: line, font, tick, and grid colors. Grid color is\n        lightened by blending this with the plot background Individual\n        pieces can override this.\n\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            i = 10
            return i + 15
        self['color'] = val

    @property
    def dtick(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The stride between grid lines along the axis\n\n        The 'dtick' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['dtick']

    @dtick.setter
    def dtick(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['dtick'] = val

    @property
    def endline(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines whether or not a line is drawn at along the final\n        value of this axis. If True, the end line is drawn on top of\n        the grid lines.\n\n        The 'endline' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['endline']

    @endline.setter
    def endline(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['endline'] = val

    @property
    def endlinecolor(self):
        if False:
            print('Hello World!')
        "\n        Sets the line color of the end line.\n\n        The 'endlinecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['endlinecolor']

    @endlinecolor.setter
    def endlinecolor(self, val):
        if False:
            print('Hello World!')
        self['endlinecolor'] = val

    @property
    def endlinewidth(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the width (in px) of the end line.\n\n        The 'endlinewidth' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['endlinewidth']

    @endlinewidth.setter
    def endlinewidth(self, val):
        if False:
            i = 10
            return i + 15
        self['endlinewidth'] = val

    @property
    def exponentformat(self):
        if False:
            i = 10
            return i + 15
        '\n        Determines a formatting rule for the tick exponents. For\n        example, consider the number 1,000,000,000. If "none", it\n        appears as 1,000,000,000. If "e", 1e+9. If "E", 1E+9. If\n        "power", 1x10^9 (with 9 in a super script). If "SI", 1G. If\n        "B", 1B.\n\n        The \'exponentformat\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'none\', \'e\', \'E\', \'power\', \'SI\', \'B\']\n\n        Returns\n        -------\n        Any\n        '
        return self['exponentformat']

    @exponentformat.setter
    def exponentformat(self, val):
        if False:
            while True:
                i = 10
        self['exponentformat'] = val

    @property
    def fixedrange(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not this axis is zoom-able. If true, then\n        zoom is disabled.\n\n        The 'fixedrange' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['fixedrange']

    @fixedrange.setter
    def fixedrange(self, val):
        if False:
            print('Hello World!')
        self['fixedrange'] = val

    @property
    def gridcolor(self):
        if False:
            return 10
        "\n        Sets the axis line color.\n\n        The 'gridcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['gridcolor']

    @gridcolor.setter
    def gridcolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['gridcolor'] = val

    @property
    def griddash(self):
        if False:
            i = 10
            return i + 15
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
            return 10
        "\n        Sets the width (in px) of the axis line.\n\n        The 'gridwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['gridwidth']

    @gridwidth.setter
    def gridwidth(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['gridwidth'] = val

    @property
    def labelalias(self):
        if False:
            while True:
                i = 10
        "\n        Replacement text for specific tick or hover labels. For example\n        using {US: 'USA', CA: 'Canada'} changes US to USA and CA to\n        Canada. The labels we would have shown must match the keys\n        exactly, after adding any tickprefix or ticksuffix. For\n        negative numbers the minus sign symbol used (U+2212) is wider\n        than the regular ascii dash. That means you need to use −1\n        instead of -1. labelalias can be used with any axis type, and\n        both keys (if needed) and values (if desired) can include html-\n        like tags or MathJax.\n\n        The 'labelalias' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['labelalias']

    @labelalias.setter
    def labelalias(self, val):
        if False:
            return 10
        self['labelalias'] = val

    @property
    def labelpadding(self):
        if False:
            i = 10
            return i + 15
        "\n        Extra padding between label and the axis\n\n        The 'labelpadding' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n\n        Returns\n        -------\n        int\n        "
        return self['labelpadding']

    @labelpadding.setter
    def labelpadding(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['labelpadding'] = val

    @property
    def labelprefix(self):
        if False:
            print('Hello World!')
        "\n        Sets a axis label prefix.\n\n        The 'labelprefix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['labelprefix']

    @labelprefix.setter
    def labelprefix(self, val):
        if False:
            print('Hello World!')
        self['labelprefix'] = val

    @property
    def labelsuffix(self):
        if False:
            while True:
                i = 10
        "\n        Sets a axis label suffix.\n\n        The 'labelsuffix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['labelsuffix']

    @labelsuffix.setter
    def labelsuffix(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['labelsuffix'] = val

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
            while True:
                i = 10
        "\n        Sets the width (in px) of the axis line.\n\n        The 'linewidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['linewidth']

    @linewidth.setter
    def linewidth(self, val):
        if False:
            while True:
                i = 10
        self['linewidth'] = val

    @property
    def minexponent(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Hide SI prefix for 10^n if |n| is below this number\n\n        The 'minexponent' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['minexponent']

    @minexponent.setter
    def minexponent(self, val):
        if False:
            while True:
                i = 10
        self['minexponent'] = val

    @property
    def minorgridcolor(self):
        if False:
            while True:
                i = 10
        "\n        Sets the color of the grid lines.\n\n        The 'minorgridcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['minorgridcolor']

    @minorgridcolor.setter
    def minorgridcolor(self, val):
        if False:
            return 10
        self['minorgridcolor'] = val

    @property
    def minorgridcount(self):
        if False:
            return 10
        "\n        Sets the number of minor grid ticks per major grid tick\n\n        The 'minorgridcount' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['minorgridcount']

    @minorgridcount.setter
    def minorgridcount(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['minorgridcount'] = val

    @property
    def minorgriddash(self):
        if False:
            print('Hello World!')
        '\n        Sets the dash style of lines. Set to a dash type string\n        ("solid", "dot", "dash", "longdash", "dashdot", or\n        "longdashdot") or a dash length list in px (eg\n        "5px,10px,2px,2px").\n\n        The \'minorgriddash\' property is an enumeration that may be specified as:\n          - One of the following dash styles:\n                [\'solid\', \'dot\', \'dash\', \'longdash\', \'dashdot\', \'longdashdot\']\n          - A string containing a dash length list in pixels or percentages\n                (e.g. \'5px 10px 2px 2px\', \'5, 10, 2, 2\', \'10% 20% 40%\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['minorgriddash']

    @minorgriddash.setter
    def minorgriddash(self, val):
        if False:
            i = 10
            return i + 15
        self['minorgriddash'] = val

    @property
    def minorgridwidth(self):
        if False:
            return 10
        "\n        Sets the width (in px) of the grid lines.\n\n        The 'minorgridwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['minorgridwidth']

    @minorgridwidth.setter
    def minorgridwidth(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['minorgridwidth'] = val

    @property
    def nticks(self):
        if False:
            return 10
        '\n        Specifies the maximum number of ticks for the particular axis.\n        The actual number of ticks will be chosen automatically to be\n        less than or equal to `nticks`. Has an effect only if\n        `tickmode` is set to "auto".\n\n        The \'nticks\' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        '
        return self['nticks']

    @nticks.setter
    def nticks(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['nticks'] = val

    @property
    def range(self):
        if False:
            while True:
                i = 10
        '\n            Sets the range of this axis. If the axis `type` is "log", then\n            you must take the log of your desired range (e.g. to set the\n            range from 1 to 100, set the range from 0 to 2). If the axis\n            `type` is "date", it should be date strings, like date data,\n            though Date objects and unix milliseconds will be accepted and\n            converted to strings. If the axis `type` is "category", it\n            should be numbers, using the scale where each category is\n            assigned a serial number from zero in the order it appears.\n\n            The \'range\' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The \'range[0]\' property accepts values of any type\n        (1) The \'range[1]\' property accepts values of any type\n\n            Returns\n            -------\n            list\n        '
        return self['range']

    @range.setter
    def range(self, val):
        if False:
            while True:
                i = 10
        self['range'] = val

    @property
    def rangemode(self):
        if False:
            print('Hello World!')
        '\n        If "normal", the range is computed in relation to the extrema\n        of the input data. If *tozero*`, the range extends to 0,\n        regardless of the input data If "nonnegative", the range is\n        non-negative, regardless of the input data.\n\n        The \'rangemode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'normal\', \'tozero\', \'nonnegative\']\n\n        Returns\n        -------\n        Any\n        '
        return self['rangemode']

    @rangemode.setter
    def rangemode(self, val):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
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
            while True:
                i = 10
        "\n        Determines whether or not grid lines are drawn. If True, the\n        grid lines are drawn at every tick mark.\n\n        The 'showgrid' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showgrid']

    @showgrid.setter
    def showgrid(self, val):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
        self['showline'] = val

    @property
    def showticklabels(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines whether axis labels are drawn on the low side, the\n        high side, both, or neither side of the axis.\n\n        The 'showticklabels' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['start', 'end', 'both', 'none']\n\n        Returns\n        -------\n        Any\n        "
        return self['showticklabels']

    @showticklabels.setter
    def showticklabels(self, val):
        if False:
            print('Hello World!')
        self['showticklabels'] = val

    @property
    def showtickprefix(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If "all", all tick labels are displayed with a prefix. If\n        "first", only the first tick is displayed with a prefix. If\n        "last", only the last tick is displayed with a suffix. If\n        "none", tick prefixes are hidden.\n\n        The \'showtickprefix\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'all\', \'first\', \'last\', \'none\']\n\n        Returns\n        -------\n        Any\n        '
        return self['showtickprefix']

    @showtickprefix.setter
    def showtickprefix(self, val):
        if False:
            print('Hello World!')
        self['showtickprefix'] = val

    @property
    def showticksuffix(self):
        if False:
            while True:
                i = 10
        "\n        Same as `showtickprefix` but for tick suffixes.\n\n        The 'showticksuffix' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['all', 'first', 'last', 'none']\n\n        Returns\n        -------\n        Any\n        "
        return self['showticksuffix']

    @showticksuffix.setter
    def showticksuffix(self, val):
        if False:
            return 10
        self['showticksuffix'] = val

    @property
    def smoothing(self):
        if False:
            while True:
                i = 10
        "\n        The 'smoothing' property is a number and may be specified as:\n          - An int or float in the interval [0, 1.3]\n\n        Returns\n        -------\n        int|float\n        "
        return self['smoothing']

    @smoothing.setter
    def smoothing(self, val):
        if False:
            while True:
                i = 10
        self['smoothing'] = val

    @property
    def startline(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether or not a line is drawn at along the starting\n        value of this axis. If True, the start line is drawn on top of\n        the grid lines.\n\n        The 'startline' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['startline']

    @startline.setter
    def startline(self, val):
        if False:
            while True:
                i = 10
        self['startline'] = val

    @property
    def startlinecolor(self):
        if False:
            return 10
        "\n        Sets the line color of the start line.\n\n        The 'startlinecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['startlinecolor']

    @startlinecolor.setter
    def startlinecolor(self, val):
        if False:
            return 10
        self['startlinecolor'] = val

    @property
    def startlinewidth(self):
        if False:
            while True:
                i = 10
        "\n        Sets the width (in px) of the start line.\n\n        The 'startlinewidth' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['startlinewidth']

    @startlinewidth.setter
    def startlinewidth(self, val):
        if False:
            print('Hello World!')
        self['startlinewidth'] = val

    @property
    def tick0(self):
        if False:
            while True:
                i = 10
        "\n        The starting index of grid lines along the axis\n\n        The 'tick0' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
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
            return 10
        "\n        Sets the angle of the tick labels with respect to the\n        horizontal. For example, a `tickangle` of -90 draws the tick\n        labels vertically.\n\n        The 'tickangle' property is a angle (in degrees) that may be\n        specified as a number between -180 and 180.\n        Numeric values outside this range are converted to the equivalent value\n        (e.g. 270 is converted to -90).\n\n        Returns\n        -------\n        int|float\n        "
        return self['tickangle']

    @tickangle.setter
    def tickangle(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickangle'] = val

    @property
    def tickfont(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the tick font.\n\n        The \'tickfont\' property is an instance of Tickfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.carpet.baxis.Tickfont`\n          - A dict of string/value properties that will be passed\n            to the Tickfont constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.carpet.baxis.Tickfont\n        '
        return self['tickfont']

    @tickfont.setter
    def tickfont(self, val):
        if False:
            print('Hello World!')
        self['tickfont'] = val

    @property
    def tickformat(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the tick label formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display "09~15~23.46"\n\n        The \'tickformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['tickformat']

    @tickformat.setter
    def tickformat(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickformat'] = val

    @property
    def tickformatstops(self):
        if False:
            while True:
                i = 10
        '\n        The \'tickformatstops\' property is a tuple of instances of\n        Tickformatstop that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.carpet.baxis.Tickformatstop\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Tickformatstop constructor\n\n            Supported dict properties:\n\n                dtickrange\n                    range [*min*, *max*], where "min", "max" -\n                    dtick values which describe some zoom level, it\n                    is possible to omit "min" or "max" value by\n                    passing "null"\n                enabled\n                    Determines whether or not this stop is used. If\n                    `false`, this stop is ignored even within its\n                    `dtickrange`.\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n                value\n                    string - dtickformat for described zoom level,\n                    the same as "tickformat"\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.carpet.baxis.Tickformatstop]\n        '
        return self['tickformatstops']

    @tickformatstops.setter
    def tickformatstops(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickformatstops'] = val

    @property
    def tickformatstopdefaults(self):
        if False:
            return 10
        "\n        When used in a template (as\n        layout.template.data.carpet.baxis.tickformatstopdefaults), sets\n        the default property values to use for elements of\n        carpet.baxis.tickformatstops\n\n        The 'tickformatstopdefaults' property is an instance of Tickformatstop\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.carpet.baxis.Tickformatstop`\n          - A dict of string/value properties that will be passed\n            to the Tickformatstop constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.carpet.baxis.Tickformatstop\n        "
        return self['tickformatstopdefaults']

    @tickformatstopdefaults.setter
    def tickformatstopdefaults(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tickformatstopdefaults'] = val

    @property
    def tickmode(self):
        if False:
            while True:
                i = 10
        "\n        The 'tickmode' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['linear', 'array']\n\n        Returns\n        -------\n        Any\n        "
        return self['tickmode']

    @tickmode.setter
    def tickmode(self, val):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
        self['tickprefix'] = val

    @property
    def ticksuffix(self):
        if False:
            print('Hello World!')
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
            return 10
        '\n        Sets the text displayed at the ticks position via `tickvals`.\n        Only has an effect if `tickmode` is set to "array". Used with\n        `tickvals`.\n\n        The \'ticktext\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['ticktext']

    @ticktext.setter
    def ticktext(self, val):
        if False:
            return 10
        self['ticktext'] = val

    @property
    def ticktextsrc(self):
        if False:
            return 10
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
            i = 10
            return i + 15
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
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `tickvals`.\n\n        The 'tickvalssrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['tickvalssrc']

    @tickvalssrc.setter
    def tickvalssrc(self, val):
        if False:
            i = 10
            return i + 15
        self['tickvalssrc'] = val

    @property
    def title(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'title' property is an instance of Title\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.carpet.baxis.Title`\n          - A dict of string/value properties that will be passed\n            to the Title constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this axis' title font. Note that the\n                    title's font used to be set by the now\n                    deprecated `titlefont` attribute.\n                offset\n                    An additional amount by which to offset the\n                    title from the tick labels, given in pixels.\n                    Note that this used to be set by the now\n                    deprecated `titleoffset` attribute.\n                text\n                    Sets the title of this axis. Note that before\n                    the existence of `title.text`, the title's\n                    contents used to be defined as the `title`\n                    attribute itself. This behavior has been\n                    deprecated.\n\n        Returns\n        -------\n        plotly.graph_objs.carpet.baxis.Title\n        "
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
        '\n        Deprecated: Please use carpet.baxis.title.font instead. Sets\n        this axis\' title font. Note that the title\'s font used to be\n        set by the now deprecated `titlefont` attribute.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.carpet.baxis.title.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n\n        '
        return self['titlefont']

    @titlefont.setter
    def titlefont(self, val):
        if False:
            print('Hello World!')
        self['titlefont'] = val

    @property
    def titleoffset(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Deprecated: Please use carpet.baxis.title.offset instead. An\n        additional amount by which to offset the title from the tick\n        labels, given in pixels. Note that this used to be set by the\n        now deprecated `titleoffset` attribute.\n\n        The 'offset' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n\n        "
        return self['titleoffset']

    @titleoffset.setter
    def titleoffset(self, val):
        if False:
            i = 10
            return i + 15
        self['titleoffset'] = val

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the axis type. By default, plotly attempts to determined\n        the axis type by looking into the data of the traces that\n        referenced the axis in question.\n\n        The 'type' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['-', 'linear', 'date', 'category']\n\n        Returns\n        -------\n        Any\n        "
        return self['type']

    @type.setter
    def type(self, val):
        if False:
            print('Hello World!')
        self['type'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        arraydtick\n            The stride between grid lines along the axis\n        arraytick0\n            The starting index of grid lines along the axis\n        autorange\n            Determines whether or not the range of this axis is\n            computed in relation to the input data. See `rangemode`\n            for more info. If `range` is provided, then `autorange`\n            is set to False.\n        autotypenumbers\n            Using "strict" a numeric string in trace data is not\n            converted to a number. Using *convert types* a numeric\n            string in trace data may be treated as a number during\n            automatic axis `type` detection. Defaults to\n            layout.autotypenumbers.\n        categoryarray\n            Sets the order in which categories on this axis appear.\n            Only has an effect if `categoryorder` is set to\n            "array". Used with `categoryorder`.\n        categoryarraysrc\n            Sets the source reference on Chart Studio Cloud for\n            `categoryarray`.\n        categoryorder\n            Specifies the ordering logic for the case of\n            categorical variables. By default, plotly uses "trace",\n            which specifies the order that is present in the data\n            supplied. Set `categoryorder` to *category ascending*\n            or *category descending* if order should be determined\n            by the alphanumerical order of the category names. Set\n            `categoryorder` to "array" to derive the ordering from\n            the attribute `categoryarray`. If a category is not\n            found in the `categoryarray` array, the sorting\n            behavior for that attribute will be identical to the\n            "trace" mode. The unspecified categories will follow\n            the categories in `categoryarray`.\n        cheatertype\n\n        color\n            Sets default for all colors associated with this axis\n            all at once: line, font, tick, and grid colors. Grid\n            color is lightened by blending this with the plot\n            background Individual pieces can override this.\n        dtick\n            The stride between grid lines along the axis\n        endline\n            Determines whether or not a line is drawn at along the\n            final value of this axis. If True, the end line is\n            drawn on top of the grid lines.\n        endlinecolor\n            Sets the line color of the end line.\n        endlinewidth\n            Sets the width (in px) of the end line.\n        exponentformat\n            Determines a formatting rule for the tick exponents.\n            For example, consider the number 1,000,000,000. If\n            "none", it appears as 1,000,000,000. If "e", 1e+9. If\n            "E", 1E+9. If "power", 1x10^9 (with 9 in a super\n            script). If "SI", 1G. If "B", 1B.\n        fixedrange\n            Determines whether or not this axis is zoom-able. If\n            true, then zoom is disabled.\n        gridcolor\n            Sets the axis line color.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the axis line.\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        labelpadding\n            Extra padding between label and the axis\n        labelprefix\n            Sets a axis label prefix.\n        labelsuffix\n            Sets a axis label suffix.\n        linecolor\n            Sets the axis line color.\n        linewidth\n            Sets the width (in px) of the axis line.\n        minexponent\n            Hide SI prefix for 10^n if |n| is below this number\n        minorgridcolor\n            Sets the color of the grid lines.\n        minorgridcount\n            Sets the number of minor grid ticks per major grid tick\n        minorgriddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        minorgridwidth\n            Sets the width (in px) of the grid lines.\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        range\n            Sets the range of this axis. If the axis `type` is\n            "log", then you must take the log of your desired range\n            (e.g. to set the range from 1 to 100, set the range\n            from 0 to 2). If the axis `type` is "date", it should\n            be date strings, like date data, though Date objects\n            and unix milliseconds will be accepted and converted to\n            strings. If the axis `type` is "category", it should be\n            numbers, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        rangemode\n            If "normal", the range is computed in relation to the\n            extrema of the input data. If *tozero*`, the range\n            extends to 0, regardless of the input data If\n            "nonnegative", the range is non-negative, regardless of\n            the input data.\n        separatethousands\n            If "true", even 4-digit integers are separated\n        showexponent\n            If "all", all exponents are shown besides their\n            significands. If "first", only the exponent of the\n            first tick is shown. If "last", only the exponent of\n            the last tick is shown. If "none", no exponents appear.\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        showline\n            Determines whether or not a line bounding this axis is\n            drawn.\n        showticklabels\n            Determines whether axis labels are drawn on the low\n            side, the high side, both, or neither side of the axis.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        smoothing\n\n        startline\n            Determines whether or not a line is drawn at along the\n            starting value of this axis. If True, the start line is\n            drawn on top of the grid lines.\n        startlinecolor\n            Sets the line color of the start line.\n        startlinewidth\n            Sets the width (in px) of the start line.\n        tick0\n            The starting index of grid lines along the axis\n        tickangle\n            Sets the angle of the tick labels with respect to the\n            horizontal. For example, a `tickangle` of -90 draws the\n            tick labels vertically.\n        tickfont\n            Sets the tick font.\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        tickformatstops\n            A tuple of :class:`plotly.graph_objects.carpet.baxis.Ti\n            ckformatstop` instances or dicts with compatible\n            properties\n        tickformatstopdefaults\n            When used in a template (as layout.template.data.carpet\n            .baxis.tickformatstopdefaults), sets the default\n            property values to use for elements of\n            carpet.baxis.tickformatstops\n        tickmode\n\n        tickprefix\n            Sets a tick label prefix.\n        ticksuffix\n            Sets a tick label suffix.\n        ticktext\n            Sets the text displayed at the ticks position via\n            `tickvals`. Only has an effect if `tickmode` is set to\n            "array". Used with `tickvals`.\n        ticktextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `ticktext`.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        title\n            :class:`plotly.graph_objects.carpet.baxis.Title`\n            instance or dict with compatible properties\n        titlefont\n            Deprecated: Please use carpet.baxis.title.font instead.\n            Sets this axis\' title font. Note that the title\'s font\n            used to be set by the now deprecated `titlefont`\n            attribute.\n        titleoffset\n            Deprecated: Please use carpet.baxis.title.offset\n            instead. An additional amount by which to offset the\n            title from the tick labels, given in pixels. Note that\n            this used to be set by the now deprecated `titleoffset`\n            attribute.\n        type\n            Sets the axis type. By default, plotly attempts to\n            determined the axis type by looking into the data of\n            the traces that referenced the axis in question.\n        '
    _mapped_properties = {'titlefont': ('title', 'font'), 'titleoffset': ('title', 'offset')}

    def __init__(self, arg=None, arraydtick=None, arraytick0=None, autorange=None, autotypenumbers=None, categoryarray=None, categoryarraysrc=None, categoryorder=None, cheatertype=None, color=None, dtick=None, endline=None, endlinecolor=None, endlinewidth=None, exponentformat=None, fixedrange=None, gridcolor=None, griddash=None, gridwidth=None, labelalias=None, labelpadding=None, labelprefix=None, labelsuffix=None, linecolor=None, linewidth=None, minexponent=None, minorgridcolor=None, minorgridcount=None, minorgriddash=None, minorgridwidth=None, nticks=None, range=None, rangemode=None, separatethousands=None, showexponent=None, showgrid=None, showline=None, showticklabels=None, showtickprefix=None, showticksuffix=None, smoothing=None, startline=None, startlinecolor=None, startlinewidth=None, tick0=None, tickangle=None, tickfont=None, tickformat=None, tickformatstops=None, tickformatstopdefaults=None, tickmode=None, tickprefix=None, ticksuffix=None, ticktext=None, ticktextsrc=None, tickvals=None, tickvalssrc=None, title=None, titlefont=None, titleoffset=None, type=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Baxis object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.carpet.Baxis`\n        arraydtick\n            The stride between grid lines along the axis\n        arraytick0\n            The starting index of grid lines along the axis\n        autorange\n            Determines whether or not the range of this axis is\n            computed in relation to the input data. See `rangemode`\n            for more info. If `range` is provided, then `autorange`\n            is set to False.\n        autotypenumbers\n            Using "strict" a numeric string in trace data is not\n            converted to a number. Using *convert types* a numeric\n            string in trace data may be treated as a number during\n            automatic axis `type` detection. Defaults to\n            layout.autotypenumbers.\n        categoryarray\n            Sets the order in which categories on this axis appear.\n            Only has an effect if `categoryorder` is set to\n            "array". Used with `categoryorder`.\n        categoryarraysrc\n            Sets the source reference on Chart Studio Cloud for\n            `categoryarray`.\n        categoryorder\n            Specifies the ordering logic for the case of\n            categorical variables. By default, plotly uses "trace",\n            which specifies the order that is present in the data\n            supplied. Set `categoryorder` to *category ascending*\n            or *category descending* if order should be determined\n            by the alphanumerical order of the category names. Set\n            `categoryorder` to "array" to derive the ordering from\n            the attribute `categoryarray`. If a category is not\n            found in the `categoryarray` array, the sorting\n            behavior for that attribute will be identical to the\n            "trace" mode. The unspecified categories will follow\n            the categories in `categoryarray`.\n        cheatertype\n\n        color\n            Sets default for all colors associated with this axis\n            all at once: line, font, tick, and grid colors. Grid\n            color is lightened by blending this with the plot\n            background Individual pieces can override this.\n        dtick\n            The stride between grid lines along the axis\n        endline\n            Determines whether or not a line is drawn at along the\n            final value of this axis. If True, the end line is\n            drawn on top of the grid lines.\n        endlinecolor\n            Sets the line color of the end line.\n        endlinewidth\n            Sets the width (in px) of the end line.\n        exponentformat\n            Determines a formatting rule for the tick exponents.\n            For example, consider the number 1,000,000,000. If\n            "none", it appears as 1,000,000,000. If "e", 1e+9. If\n            "E", 1E+9. If "power", 1x10^9 (with 9 in a super\n            script). If "SI", 1G. If "B", 1B.\n        fixedrange\n            Determines whether or not this axis is zoom-able. If\n            true, then zoom is disabled.\n        gridcolor\n            Sets the axis line color.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the axis line.\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        labelpadding\n            Extra padding between label and the axis\n        labelprefix\n            Sets a axis label prefix.\n        labelsuffix\n            Sets a axis label suffix.\n        linecolor\n            Sets the axis line color.\n        linewidth\n            Sets the width (in px) of the axis line.\n        minexponent\n            Hide SI prefix for 10^n if |n| is below this number\n        minorgridcolor\n            Sets the color of the grid lines.\n        minorgridcount\n            Sets the number of minor grid ticks per major grid tick\n        minorgriddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        minorgridwidth\n            Sets the width (in px) of the grid lines.\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        range\n            Sets the range of this axis. If the axis `type` is\n            "log", then you must take the log of your desired range\n            (e.g. to set the range from 1 to 100, set the range\n            from 0 to 2). If the axis `type` is "date", it should\n            be date strings, like date data, though Date objects\n            and unix milliseconds will be accepted and converted to\n            strings. If the axis `type` is "category", it should be\n            numbers, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        rangemode\n            If "normal", the range is computed in relation to the\n            extrema of the input data. If *tozero*`, the range\n            extends to 0, regardless of the input data If\n            "nonnegative", the range is non-negative, regardless of\n            the input data.\n        separatethousands\n            If "true", even 4-digit integers are separated\n        showexponent\n            If "all", all exponents are shown besides their\n            significands. If "first", only the exponent of the\n            first tick is shown. If "last", only the exponent of\n            the last tick is shown. If "none", no exponents appear.\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        showline\n            Determines whether or not a line bounding this axis is\n            drawn.\n        showticklabels\n            Determines whether axis labels are drawn on the low\n            side, the high side, both, or neither side of the axis.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        smoothing\n\n        startline\n            Determines whether or not a line is drawn at along the\n            starting value of this axis. If True, the start line is\n            drawn on top of the grid lines.\n        startlinecolor\n            Sets the line color of the start line.\n        startlinewidth\n            Sets the width (in px) of the start line.\n        tick0\n            The starting index of grid lines along the axis\n        tickangle\n            Sets the angle of the tick labels with respect to the\n            horizontal. For example, a `tickangle` of -90 draws the\n            tick labels vertically.\n        tickfont\n            Sets the tick font.\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        tickformatstops\n            A tuple of :class:`plotly.graph_objects.carpet.baxis.Ti\n            ckformatstop` instances or dicts with compatible\n            properties\n        tickformatstopdefaults\n            When used in a template (as layout.template.data.carpet\n            .baxis.tickformatstopdefaults), sets the default\n            property values to use for elements of\n            carpet.baxis.tickformatstops\n        tickmode\n\n        tickprefix\n            Sets a tick label prefix.\n        ticksuffix\n            Sets a tick label suffix.\n        ticktext\n            Sets the text displayed at the ticks position via\n            `tickvals`. Only has an effect if `tickmode` is set to\n            "array". Used with `tickvals`.\n        ticktextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `ticktext`.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        title\n            :class:`plotly.graph_objects.carpet.baxis.Title`\n            instance or dict with compatible properties\n        titlefont\n            Deprecated: Please use carpet.baxis.title.font instead.\n            Sets this axis\' title font. Note that the title\'s font\n            used to be set by the now deprecated `titlefont`\n            attribute.\n        titleoffset\n            Deprecated: Please use carpet.baxis.title.offset\n            instead. An additional amount by which to offset the\n            title from the tick labels, given in pixels. Note that\n            this used to be set by the now deprecated `titleoffset`\n            attribute.\n        type\n            Sets the axis type. By default, plotly attempts to\n            determined the axis type by looking into the data of\n            the traces that referenced the axis in question.\n\n        Returns\n        -------\n        Baxis\n        '
        super(Baxis, self).__init__('baxis')
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
            raise ValueError('The first argument to the plotly.graph_objs.carpet.Baxis\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.carpet.Baxis`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('arraydtick', None)
        _v = arraydtick if arraydtick is not None else _v
        if _v is not None:
            self['arraydtick'] = _v
        _v = arg.pop('arraytick0', None)
        _v = arraytick0 if arraytick0 is not None else _v
        if _v is not None:
            self['arraytick0'] = _v
        _v = arg.pop('autorange', None)
        _v = autorange if autorange is not None else _v
        if _v is not None:
            self['autorange'] = _v
        _v = arg.pop('autotypenumbers', None)
        _v = autotypenumbers if autotypenumbers is not None else _v
        if _v is not None:
            self['autotypenumbers'] = _v
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
        _v = arg.pop('cheatertype', None)
        _v = cheatertype if cheatertype is not None else _v
        if _v is not None:
            self['cheatertype'] = _v
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('dtick', None)
        _v = dtick if dtick is not None else _v
        if _v is not None:
            self['dtick'] = _v
        _v = arg.pop('endline', None)
        _v = endline if endline is not None else _v
        if _v is not None:
            self['endline'] = _v
        _v = arg.pop('endlinecolor', None)
        _v = endlinecolor if endlinecolor is not None else _v
        if _v is not None:
            self['endlinecolor'] = _v
        _v = arg.pop('endlinewidth', None)
        _v = endlinewidth if endlinewidth is not None else _v
        if _v is not None:
            self['endlinewidth'] = _v
        _v = arg.pop('exponentformat', None)
        _v = exponentformat if exponentformat is not None else _v
        if _v is not None:
            self['exponentformat'] = _v
        _v = arg.pop('fixedrange', None)
        _v = fixedrange if fixedrange is not None else _v
        if _v is not None:
            self['fixedrange'] = _v
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
        _v = arg.pop('labelalias', None)
        _v = labelalias if labelalias is not None else _v
        if _v is not None:
            self['labelalias'] = _v
        _v = arg.pop('labelpadding', None)
        _v = labelpadding if labelpadding is not None else _v
        if _v is not None:
            self['labelpadding'] = _v
        _v = arg.pop('labelprefix', None)
        _v = labelprefix if labelprefix is not None else _v
        if _v is not None:
            self['labelprefix'] = _v
        _v = arg.pop('labelsuffix', None)
        _v = labelsuffix if labelsuffix is not None else _v
        if _v is not None:
            self['labelsuffix'] = _v
        _v = arg.pop('linecolor', None)
        _v = linecolor if linecolor is not None else _v
        if _v is not None:
            self['linecolor'] = _v
        _v = arg.pop('linewidth', None)
        _v = linewidth if linewidth is not None else _v
        if _v is not None:
            self['linewidth'] = _v
        _v = arg.pop('minexponent', None)
        _v = minexponent if minexponent is not None else _v
        if _v is not None:
            self['minexponent'] = _v
        _v = arg.pop('minorgridcolor', None)
        _v = minorgridcolor if minorgridcolor is not None else _v
        if _v is not None:
            self['minorgridcolor'] = _v
        _v = arg.pop('minorgridcount', None)
        _v = minorgridcount if minorgridcount is not None else _v
        if _v is not None:
            self['minorgridcount'] = _v
        _v = arg.pop('minorgriddash', None)
        _v = minorgriddash if minorgriddash is not None else _v
        if _v is not None:
            self['minorgriddash'] = _v
        _v = arg.pop('minorgridwidth', None)
        _v = minorgridwidth if minorgridwidth is not None else _v
        if _v is not None:
            self['minorgridwidth'] = _v
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
        _v = arg.pop('smoothing', None)
        _v = smoothing if smoothing is not None else _v
        if _v is not None:
            self['smoothing'] = _v
        _v = arg.pop('startline', None)
        _v = startline if startline is not None else _v
        if _v is not None:
            self['startline'] = _v
        _v = arg.pop('startlinecolor', None)
        _v = startlinecolor if startlinecolor is not None else _v
        if _v is not None:
            self['startlinecolor'] = _v
        _v = arg.pop('startlinewidth', None)
        _v = startlinewidth if startlinewidth is not None else _v
        if _v is not None:
            self['startlinewidth'] = _v
        _v = arg.pop('tick0', None)
        _v = tick0 if tick0 is not None else _v
        if _v is not None:
            self['tick0'] = _v
        _v = arg.pop('tickangle', None)
        _v = tickangle if tickangle is not None else _v
        if _v is not None:
            self['tickangle'] = _v
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
        _v = arg.pop('tickmode', None)
        _v = tickmode if tickmode is not None else _v
        if _v is not None:
            self['tickmode'] = _v
        _v = arg.pop('tickprefix', None)
        _v = tickprefix if tickprefix is not None else _v
        if _v is not None:
            self['tickprefix'] = _v
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
        _v = arg.pop('title', None)
        _v = title if title is not None else _v
        if _v is not None:
            self['title'] = _v
        _v = arg.pop('titlefont', None)
        _v = titlefont if titlefont is not None else _v
        if _v is not None:
            self['titlefont'] = _v
        _v = arg.pop('titleoffset', None)
        _v = titleoffset if titleoffset is not None else _v
        if _v is not None:
            self['titleoffset'] = _v
        _v = arg.pop('type', None)
        _v = type if type is not None else _v
        if _v is not None:
            self['type'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
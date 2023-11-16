from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Imaginaryaxis(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.smith'
    _path_str = 'layout.smith.imaginaryaxis'
    _valid_props = {'color', 'gridcolor', 'griddash', 'gridwidth', 'hoverformat', 'labelalias', 'layer', 'linecolor', 'linewidth', 'showgrid', 'showline', 'showticklabels', 'showtickprefix', 'showticksuffix', 'tickcolor', 'tickfont', 'tickformat', 'ticklen', 'tickprefix', 'ticks', 'ticksuffix', 'tickvals', 'tickvalssrc', 'tickwidth', 'visible'}

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
    def gridcolor(self):
        if False:
            return 10
        "\n        Sets the color of the grid lines.\n\n        The 'gridcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['gridcolor']

    @gridcolor.setter
    def gridcolor(self, val):
        if False:
            i = 10
            return i + 15
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
            return 10
        self['griddash'] = val

    @property
    def gridwidth(self):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
        '\n        Sets the hover text formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display "09~15~23.46"\n\n        The \'hoverformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['hoverformat']

    @hoverformat.setter
    def hoverformat(self, val):
        if False:
            i = 10
            return i + 15
        self['hoverformat'] = val

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
            for i in range(10):
                print('nop')
        self['labelalias'] = val

    @property
    def layer(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the layer on which this axis is displayed. If *above\n        traces*, this axis is displayed above all the subplot's traces\n        If *below traces*, this axis is displayed below all the\n        subplot's traces, but above the grid lines. Useful when used\n        together with scatter-like traces with `cliponaxis` set to\n        False to show markers and/or text nodes above this axis.\n\n        The 'layer' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['above traces', 'below traces']\n\n        Returns\n        -------\n        Any\n        "
        return self['layer']

    @layer.setter
    def layer(self, val):
        if False:
            while True:
                i = 10
        self['layer'] = val

    @property
    def linecolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the axis line color.\n\n        The 'linecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['linecolor']

    @linecolor.setter
    def linecolor(self, val):
        if False:
            i = 10
            return i + 15
        self['linecolor'] = val

    @property
    def linewidth(self):
        if False:
            print('Hello World!')
        "\n        Sets the width (in px) of the axis line.\n\n        The 'linewidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['linewidth']

    @linewidth.setter
    def linewidth(self, val):
        if False:
            return 10
        self['linewidth'] = val

    @property
    def showgrid(self):
        if False:
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
        self['showline'] = val

    @property
    def showticklabels(self):
        if False:
            return 10
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
            for i in range(10):
                print('nop')
        '\n        If "all", all tick labels are displayed with a prefix. If\n        "first", only the first tick is displayed with a prefix. If\n        "last", only the last tick is displayed with a suffix. If\n        "none", tick prefixes are hidden.\n\n        The \'showtickprefix\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'all\', \'first\', \'last\', \'none\']\n\n        Returns\n        -------\n        Any\n        '
        return self['showtickprefix']

    @showtickprefix.setter
    def showtickprefix(self, val):
        if False:
            return 10
        self['showtickprefix'] = val

    @property
    def showticksuffix(self):
        if False:
            i = 10
            return i + 15
        "\n        Same as `showtickprefix` but for tick suffixes.\n\n        The 'showticksuffix' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['all', 'first', 'last', 'none']\n\n        Returns\n        -------\n        Any\n        "
        return self['showticksuffix']

    @showticksuffix.setter
    def showticksuffix(self, val):
        if False:
            return 10
        self['showticksuffix'] = val

    @property
    def tickcolor(self):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
        '\n        Sets the tick font.\n\n        The \'tickfont\' property is an instance of Tickfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.smith.imaginaryaxis.Tickfont`\n          - A dict of string/value properties that will be passed\n            to the Tickfont constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.smith.imaginaryaxis.Tickfont\n        '
        return self['tickfont']

    @tickfont.setter
    def tickfont(self, val):
        if False:
            return 10
        self['tickfont'] = val

    @property
    def tickformat(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the tick label formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display "09~15~23.46"\n\n        The \'tickformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['tickformat']

    @tickformat.setter
    def tickformat(self, val):
        if False:
            i = 10
            return i + 15
        self['tickformat'] = val

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
    def tickprefix(self):
        if False:
            print('Hello World!')
        "\n        Sets a tick label prefix.\n\n        The 'tickprefix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['tickprefix']

    @tickprefix.setter
    def tickprefix(self, val):
        if False:
            print('Hello World!')
        self['tickprefix'] = val

    @property
    def ticks(self):
        if False:
            return 10
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
            while True:
                i = 10
        "\n        Sets a tick label suffix.\n\n        The 'ticksuffix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['ticksuffix']

    @ticksuffix.setter
    def ticksuffix(self, val):
        if False:
            i = 10
            return i + 15
        self['ticksuffix'] = val

    @property
    def tickvals(self):
        if False:
            return 10
        "\n        Sets the values at which ticks on this axis appear. Defaults to\n        `realaxis.tickvals` plus the same as negatives and zero.\n\n        The 'tickvals' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['tickvals']

    @tickvals.setter
    def tickvals(self, val):
        if False:
            print('Hello World!')
        self['tickvals'] = val

    @property
    def tickvalssrc(self):
        if False:
            for i in range(10):
                print('nop')
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
    def visible(self):
        if False:
            print('Hello World!')
        "\n        A single toggle to hide the axis while preserving interaction\n        like dragging. Default is true when a cheater plot is present\n        on the axis, otherwise false\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['visible'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        color\n            Sets default for all colors associated with this axis\n            all at once: line, font, tick, and grid colors. Grid\n            color is lightened by blending this with the plot\n            background Individual pieces can override this.\n        gridcolor\n            Sets the color of the grid lines.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the grid lines.\n        hoverformat\n            Sets the hover text formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        layer\n            Sets the layer on which this axis is displayed. If\n            *above traces*, this axis is displayed above all the\n            subplot\'s traces If *below traces*, this axis is\n            displayed below all the subplot\'s traces, but above the\n            grid lines. Useful when used together with scatter-like\n            traces with `cliponaxis` set to False to show markers\n            and/or text nodes above this axis.\n        linecolor\n            Sets the axis line color.\n        linewidth\n            Sets the width (in px) of the axis line.\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        showline\n            Determines whether or not a line bounding this axis is\n            drawn.\n        showticklabels\n            Determines whether or not the tick labels are drawn.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        tickcolor\n            Sets the tick color.\n        tickfont\n            Sets the tick font.\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        ticklen\n            Sets the tick length (in px).\n        tickprefix\n            Sets a tick label prefix.\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        ticksuffix\n            Sets a tick label suffix.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Defaults to `realaxis.tickvals` plus the same as\n            negatives and zero.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n        visible\n            A single toggle to hide the axis while preserving\n            interaction like dragging. Default is true when a\n            cheater plot is present on the axis, otherwise false\n        '

    def __init__(self, arg=None, color=None, gridcolor=None, griddash=None, gridwidth=None, hoverformat=None, labelalias=None, layer=None, linecolor=None, linewidth=None, showgrid=None, showline=None, showticklabels=None, showtickprefix=None, showticksuffix=None, tickcolor=None, tickfont=None, tickformat=None, ticklen=None, tickprefix=None, ticks=None, ticksuffix=None, tickvals=None, tickvalssrc=None, tickwidth=None, visible=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Imaginaryaxis object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.smith.Imaginaryaxis`\n        color\n            Sets default for all colors associated with this axis\n            all at once: line, font, tick, and grid colors. Grid\n            color is lightened by blending this with the plot\n            background Individual pieces can override this.\n        gridcolor\n            Sets the color of the grid lines.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the grid lines.\n        hoverformat\n            Sets the hover text formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        layer\n            Sets the layer on which this axis is displayed. If\n            *above traces*, this axis is displayed above all the\n            subplot\'s traces If *below traces*, this axis is\n            displayed below all the subplot\'s traces, but above the\n            grid lines. Useful when used together with scatter-like\n            traces with `cliponaxis` set to False to show markers\n            and/or text nodes above this axis.\n        linecolor\n            Sets the axis line color.\n        linewidth\n            Sets the width (in px) of the axis line.\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        showline\n            Determines whether or not a line bounding this axis is\n            drawn.\n        showticklabels\n            Determines whether or not the tick labels are drawn.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        tickcolor\n            Sets the tick color.\n        tickfont\n            Sets the tick font.\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        ticklen\n            Sets the tick length (in px).\n        tickprefix\n            Sets a tick label prefix.\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        ticksuffix\n            Sets a tick label suffix.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Defaults to `realaxis.tickvals` plus the same as\n            negatives and zero.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n        visible\n            A single toggle to hide the axis while preserving\n            interaction like dragging. Default is true when a\n            cheater plot is present on the axis, otherwise false\n\n        Returns\n        -------\n        Imaginaryaxis\n        '
        super(Imaginaryaxis, self).__init__('imaginaryaxis')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.smith.Imaginaryaxis\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.smith.Imaginaryaxis`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
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
        _v = arg.pop('ticklen', None)
        _v = ticklen if ticklen is not None else _v
        if _v is not None:
            self['ticklen'] = _v
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
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
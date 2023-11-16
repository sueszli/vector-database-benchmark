from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class YAxis(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.yaxis'
    _valid_props = {'anchor', 'automargin', 'autorange', 'autorangeoptions', 'autoshift', 'autotypenumbers', 'calendar', 'categoryarray', 'categoryarraysrc', 'categoryorder', 'color', 'constrain', 'constraintoward', 'dividercolor', 'dividerwidth', 'domain', 'dtick', 'exponentformat', 'fixedrange', 'gridcolor', 'griddash', 'gridwidth', 'hoverformat', 'insiderange', 'labelalias', 'layer', 'linecolor', 'linewidth', 'matches', 'maxallowed', 'minallowed', 'minexponent', 'minor', 'mirror', 'nticks', 'overlaying', 'position', 'range', 'rangebreakdefaults', 'rangebreaks', 'rangemode', 'scaleanchor', 'scaleratio', 'separatethousands', 'shift', 'showdividers', 'showexponent', 'showgrid', 'showline', 'showspikes', 'showticklabels', 'showtickprefix', 'showticksuffix', 'side', 'spikecolor', 'spikedash', 'spikemode', 'spikesnap', 'spikethickness', 'tick0', 'tickangle', 'tickcolor', 'tickfont', 'tickformat', 'tickformatstopdefaults', 'tickformatstops', 'ticklabelmode', 'ticklabeloverflow', 'ticklabelposition', 'ticklabelstep', 'ticklen', 'tickmode', 'tickprefix', 'ticks', 'tickson', 'ticksuffix', 'ticktext', 'ticktextsrc', 'tickvals', 'tickvalssrc', 'tickwidth', 'title', 'titlefont', 'type', 'uirevision', 'visible', 'zeroline', 'zerolinecolor', 'zerolinewidth'}

    @property
    def anchor(self):
        if False:
            return 10
        '\n        If set to an opposite-letter axis id (e.g. `x2`, `y`), this\n        axis is bound to the corresponding opposite-letter axis. If set\n        to "free", this axis\' position is determined by `position`.\n\n        The \'anchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'free\']\n          - A string that matches one of the following regular expressions:\n                [\'^x([2-9]|[1-9][0-9]+)?( domain)?$\',\n                \'^y([2-9]|[1-9][0-9]+)?( domain)?$\']\n\n        Returns\n        -------\n        Any\n        '
        return self['anchor']

    @anchor.setter
    def anchor(self, val):
        if False:
            return 10
        self['anchor'] = val

    @property
    def automargin(self):
        if False:
            return 10
        "\n        Determines whether long tick labels automatically grow the\n        figure margins.\n\n        The 'automargin' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['height', 'width', 'left', 'right', 'top', 'bottom'] joined with '+' characters\n            (e.g. 'height+width')\n            OR exactly one of [True, False] (e.g. 'False')\n\n        Returns\n        -------\n        Any\n        "
        return self['automargin']

    @automargin.setter
    def automargin(self, val):
        if False:
            i = 10
            return i + 15
        self['automargin'] = val

    @property
    def autorange(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines whether or not the range of this axis is computed in\n        relation to the input data. See `rangemode` for more info. If\n        `range` is provided and it has a value for both the lower and\n        upper bound, `autorange` is set to False. Using "min" applies\n        autorange only to set the minimum. Using "max" applies\n        autorange only to set the maximum. Using *min reversed* applies\n        autorange only to set the minimum on a reversed axis. Using\n        *max reversed* applies autorange only to set the maximum on a\n        reversed axis. Using "reversed" applies autorange on both ends\n        and reverses the axis direction.\n\n        The \'autorange\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, \'reversed\', \'min reversed\', \'max reversed\',\n                \'min\', \'max\']\n\n        Returns\n        -------\n        Any\n        '
        return self['autorange']

    @autorange.setter
    def autorange(self, val):
        if False:
            return 10
        self['autorange'] = val

    @property
    def autorangeoptions(self):
        if False:
            while True:
                i = 10
        "\n        The 'autorangeoptions' property is an instance of Autorangeoptions\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.yaxis.Autorangeoptions`\n          - A dict of string/value properties that will be passed\n            to the Autorangeoptions constructor\n\n            Supported dict properties:\n\n                clipmax\n                    Clip autorange maximum if it goes beyond this\n                    value. Has no effect when\n                    `autorangeoptions.maxallowed` is provided.\n                clipmin\n                    Clip autorange minimum if it goes beyond this\n                    value. Has no effect when\n                    `autorangeoptions.minallowed` is provided.\n                include\n                    Ensure this value is included in autorange.\n                includesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `include`.\n                maxallowed\n                    Use this value exactly as autorange maximum.\n                minallowed\n                    Use this value exactly as autorange minimum.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.yaxis.Autorangeoptions\n        "
        return self['autorangeoptions']

    @autorangeoptions.setter
    def autorangeoptions(self, val):
        if False:
            print('Hello World!')
        self['autorangeoptions'] = val

    @property
    def autoshift(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Automatically reposition the axis to avoid overlap with other\n        axes with the same `overlaying` value. This repositioning will\n        account for any `shift` amount applied to other axes on the\n        same side with `autoshift` is set to true. Only has an effect\n        if `anchor` is set to "free".\n\n        The \'autoshift\' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        '
        return self['autoshift']

    @autoshift.setter
    def autoshift(self, val):
        if False:
            i = 10
            return i + 15
        self['autoshift'] = val

    @property
    def autotypenumbers(self):
        if False:
            print('Hello World!')
        '\n        Using "strict" a numeric string in trace data is not converted\n        to a number. Using *convert types* a numeric string in trace\n        data may be treated as a number during automatic axis `type`\n        detection. Defaults to layout.autotypenumbers.\n\n        The \'autotypenumbers\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'convert types\', \'strict\']\n\n        Returns\n        -------\n        Any\n        '
        return self['autotypenumbers']

    @autotypenumbers.setter
    def autotypenumbers(self, val):
        if False:
            print('Hello World!')
        self['autotypenumbers'] = val

    @property
    def calendar(self):
        if False:
            return 10
        "\n        Sets the calendar system to use for `range` and `tick0` if this\n        is a date axis. This does not set the calendar for interpreting\n        data on this axis, that's specified in the trace or via the\n        global `layout.calendar`\n\n        The 'calendar' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['chinese', 'coptic', 'discworld', 'ethiopian',\n                'gregorian', 'hebrew', 'islamic', 'jalali', 'julian',\n                'mayan', 'nanakshahi', 'nepali', 'persian', 'taiwan',\n                'thai', 'ummalqura']\n\n        Returns\n        -------\n        Any\n        "
        return self['calendar']

    @calendar.setter
    def calendar(self, val):
        if False:
            i = 10
            return i + 15
        self['calendar'] = val

    @property
    def categoryarray(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the order in which categories on this axis appear. Only\n        has an effect if `categoryorder` is set to "array". Used with\n        `categoryorder`.\n\n        The \'categoryarray\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['categoryarray']

    @categoryarray.setter
    def categoryarray(self, val):
        if False:
            return 10
        self['categoryarray'] = val

    @property
    def categoryarraysrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `categoryarray`.\n\n        The 'categoryarraysrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['categoryarraysrc']

    @categoryarraysrc.setter
    def categoryarraysrc(self, val):
        if False:
            i = 10
            return i + 15
        self['categoryarraysrc'] = val

    @property
    def categoryorder(self):
        if False:
            print('Hello World!')
        '\n        Specifies the ordering logic for the case of categorical\n        variables. By default, plotly uses "trace", which specifies the\n        order that is present in the data supplied. Set `categoryorder`\n        to *category ascending* or *category descending* if order\n        should be determined by the alphanumerical order of the\n        category names. Set `categoryorder` to "array" to derive the\n        ordering from the attribute `categoryarray`. If a category is\n        not found in the `categoryarray` array, the sorting behavior\n        for that attribute will be identical to the "trace" mode. The\n        unspecified categories will follow the categories in\n        `categoryarray`. Set `categoryorder` to *total ascending* or\n        *total descending* if order should be determined by the\n        numerical order of the values. Similarly, the order can be\n        determined by the min, max, sum, mean or median of all the\n        values.\n\n        The \'categoryorder\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'trace\', \'category ascending\', \'category descending\',\n                \'array\', \'total ascending\', \'total descending\', \'min\n                ascending\', \'min descending\', \'max ascending\', \'max\n                descending\', \'sum ascending\', \'sum descending\', \'mean\n                ascending\', \'mean descending\', \'median ascending\', \'median\n                descending\']\n\n        Returns\n        -------\n        Any\n        '
        return self['categoryorder']

    @categoryorder.setter
    def categoryorder(self, val):
        if False:
            print('Hello World!')
        self['categoryorder'] = val

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
            print('Hello World!')
        self['color'] = val

    @property
    def constrain(self):
        if False:
            print('Hello World!')
        '\n        If this axis needs to be compressed (either due to its own\n        `scaleanchor` and `scaleratio` or those of the other axis),\n        determines how that happens: by increasing the "range", or by\n        decreasing the "domain". Default is "domain" for axes\n        containing image traces, "range" otherwise.\n\n        The \'constrain\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'range\', \'domain\']\n\n        Returns\n        -------\n        Any\n        '
        return self['constrain']

    @constrain.setter
    def constrain(self, val):
        if False:
            print('Hello World!')
        self['constrain'] = val

    @property
    def constraintoward(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If this axis needs to be compressed (either due to its own\n        `scaleanchor` and `scaleratio` or those of the other axis),\n        determines which direction we push the originally specified\n        plot area. Options are "left", "center" (default), and "right"\n        for x axes, and "top", "middle" (default), and "bottom" for y\n        axes.\n\n        The \'constraintoward\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'left\', \'center\', \'right\', \'top\', \'middle\', \'bottom\']\n\n        Returns\n        -------\n        Any\n        '
        return self['constraintoward']

    @constraintoward.setter
    def constraintoward(self, val):
        if False:
            i = 10
            return i + 15
        self['constraintoward'] = val

    @property
    def dividercolor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the color of the dividers Only has an effect on\n        "multicategory" axes.\n\n        The \'dividercolor\' property is a color and may be specified as:\n          - A hex string (e.g. \'#ff0000\')\n          - An rgb/rgba string (e.g. \'rgb(255,0,0)\')\n          - An hsl/hsla string (e.g. \'hsl(0,100%,50%)\')\n          - An hsv/hsva string (e.g. \'hsv(0,100%,100%)\')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        '
        return self['dividercolor']

    @dividercolor.setter
    def dividercolor(self, val):
        if False:
            i = 10
            return i + 15
        self['dividercolor'] = val

    @property
    def dividerwidth(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the width (in px) of the dividers Only has an effect on\n        "multicategory" axes.\n\n        The \'dividerwidth\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['dividerwidth']

    @dividerwidth.setter
    def dividerwidth(self, val):
        if False:
            return 10
        self['dividerwidth'] = val

    @property
    def domain(self):
        if False:
            return 10
        "\n            Sets the domain of this axis (in plot fraction).\n\n            The 'domain' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The 'domain[0]' property is a number and may be specified as:\n              - An int or float in the interval [0, 1]\n        (1) The 'domain[1]' property is a number and may be specified as:\n              - An int or float in the interval [0, 1]\n\n            Returns\n            -------\n            list\n        "
        return self['domain']

    @domain.setter
    def domain(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['domain'] = val

    @property
    def dtick(self):
        if False:
            print('Hello World!')
        '\n        Sets the step in-between ticks on this axis. Use with `tick0`.\n        Must be a positive number, or special strings available to\n        "log" and "date" axes. If the axis `type` is "log", then ticks\n        are set every 10^(n*dtick) where n is the tick number. For\n        example, to set a tick mark at 1, 10, 100, 1000, ... set dtick\n        to 1. To set tick marks at 1, 100, 10000, ... set dtick to 2.\n        To set tick marks at 1, 5, 25, 125, 625, 3125, ... set dtick to\n        log_10(5), or 0.69897000433. "log" has several special values;\n        "L<f>", where `f` is a positive number, gives ticks linearly\n        spaced in value (but not position). For example `tick0` = 0.1,\n        `dtick` = "L0.5" will put ticks at 0.1, 0.6, 1.1, 1.6 etc. To\n        show powers of 10 plus small digits between, use "D1" (all\n        digits) or "D2" (only 2 and 5). `tick0` is ignored for "D1" and\n        "D2". If the axis `type` is "date", then you must convert the\n        time to milliseconds. For example, to set the interval between\n        ticks to one day, set `dtick` to 86400000.0. "date" also has\n        special values "M<n>" gives ticks spaced by a number of months.\n        `n` must be a positive integer. To set ticks on the 15th of\n        every third month, set `tick0` to "2000-01-15" and `dtick` to\n        "M3". To set ticks every 4 years, set `dtick` to "M48"\n\n        The \'dtick\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['dtick']

    @dtick.setter
    def dtick(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['dtick'] = val

    @property
    def exponentformat(self):
        if False:
            while True:
                i = 10
        '\n        Determines a formatting rule for the tick exponents. For\n        example, consider the number 1,000,000,000. If "none", it\n        appears as 1,000,000,000. If "e", 1e+9. If "E", 1E+9. If\n        "power", 1x10^9 (with 9 in a super script). If "SI", 1G. If\n        "B", 1B.\n\n        The \'exponentformat\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'none\', \'e\', \'E\', \'power\', \'SI\', \'B\']\n\n        Returns\n        -------\n        Any\n        '
        return self['exponentformat']

    @exponentformat.setter
    def exponentformat(self, val):
        if False:
            i = 10
            return i + 15
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
            i = 10
            return i + 15
        self['fixedrange'] = val

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
            return 10
        "\n        Sets the width (in px) of the grid lines.\n\n        The 'gridwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['gridwidth']

    @gridwidth.setter
    def gridwidth(self, val):
        if False:
            i = 10
            return i + 15
        self['gridwidth'] = val

    @property
    def hoverformat(self):
        if False:
            while True:
                i = 10
        '\n        Sets the hover text formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display "09~15~23.46"\n\n        The \'hoverformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['hoverformat']

    @hoverformat.setter
    def hoverformat(self, val):
        if False:
            print('Hello World!')
        self['hoverformat'] = val

    @property
    def insiderange(self):
        if False:
            while True:
                i = 10
        '\n            Could be used to set the desired inside range of this axis\n            (excluding the labels) when `ticklabelposition` of the anchored\n            axis has "inside". Not implemented for axes with `type` "log".\n            This would be ignored when `range` is provided.\n\n            The \'insiderange\' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The \'insiderange[0]\' property accepts values of any type\n        (1) The \'insiderange[1]\' property accepts values of any type\n\n            Returns\n            -------\n            list\n        '
        return self['insiderange']

    @insiderange.setter
    def insiderange(self, val):
        if False:
            print('Hello World!')
        self['insiderange'] = val

    @property
    def labelalias(self):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
        "\n        Sets the layer on which this axis is displayed. If *above\n        traces*, this axis is displayed above all the subplot's traces\n        If *below traces*, this axis is displayed below all the\n        subplot's traces, but above the grid lines. Useful when used\n        together with scatter-like traces with `cliponaxis` set to\n        False to show markers and/or text nodes above this axis.\n\n        The 'layer' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['above traces', 'below traces']\n\n        Returns\n        -------\n        Any\n        "
        return self['layer']

    @layer.setter
    def layer(self, val):
        if False:
            print('Hello World!')
        self['layer'] = val

    @property
    def linecolor(self):
        if False:
            return 10
        "\n        Sets the axis line color.\n\n        The 'linecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['linecolor']

    @linecolor.setter
    def linecolor(self, val):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        self['linewidth'] = val

    @property
    def matches(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If set to another axis id (e.g. `x2`, `y`), the range of this\n        axis will match the range of the corresponding axis in data-\n        coordinates space. Moreover, matching axes share auto-range\n        values, category lists and histogram auto-bins. Note that\n        setting axes simultaneously in both a `scaleanchor` and a\n        `matches` constraint is currently forbidden. Moreover, note\n        that matching axes must have the same `type`.\n\n        The 'matches' property is an enumeration that may be specified as:\n          - A string that matches one of the following regular expressions:\n                ['^x([2-9]|[1-9][0-9]+)?( domain)?$',\n                '^y([2-9]|[1-9][0-9]+)?( domain)?$']\n\n        Returns\n        -------\n        Any\n        "
        return self['matches']

    @matches.setter
    def matches(self, val):
        if False:
            print('Hello World!')
        self['matches'] = val

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
            return 10
        self['maxallowed'] = val

    @property
    def minallowed(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines the minimum range of this axis.\n\n        The 'minallowed' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['minallowed']

    @minallowed.setter
    def minallowed(self, val):
        if False:
            return 10
        self['minallowed'] = val

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
            print('Hello World!')
        self['minexponent'] = val

    @property
    def minor(self):
        if False:
            return 10
        '\n        The \'minor\' property is an instance of Minor\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.yaxis.Minor`\n          - A dict of string/value properties that will be passed\n            to the Minor constructor\n\n            Supported dict properties:\n\n                dtick\n                    Sets the step in-between ticks on this axis.\n                    Use with `tick0`. Must be a positive number, or\n                    special strings available to "log" and "date"\n                    axes. If the axis `type` is "log", then ticks\n                    are set every 10^(n*dtick) where n is the tick\n                    number. For example, to set a tick mark at 1,\n                    10, 100, 1000, ... set dtick to 1. To set tick\n                    marks at 1, 100, 10000, ... set dtick to 2. To\n                    set tick marks at 1, 5, 25, 125, 625, 3125, ...\n                    set dtick to log_10(5), or 0.69897000433. "log"\n                    has several special values; "L<f>", where `f`\n                    is a positive number, gives ticks linearly\n                    spaced in value (but not position). For example\n                    `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n                    at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10\n                    plus small digits between, use "D1" (all\n                    digits) or "D2" (only 2 and 5). `tick0` is\n                    ignored for "D1" and "D2". If the axis `type`\n                    is "date", then you must convert the time to\n                    milliseconds. For example, to set the interval\n                    between ticks to one day, set `dtick` to\n                    86400000.0. "date" also has special values\n                    "M<n>" gives ticks spaced by a number of\n                    months. `n` must be a positive integer. To set\n                    ticks on the 15th of every third month, set\n                    `tick0` to "2000-01-15" and `dtick` to "M3". To\n                    set ticks every 4 years, set `dtick` to "M48"\n                gridcolor\n                    Sets the color of the grid lines.\n                griddash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                gridwidth\n                    Sets the width (in px) of the grid lines.\n                nticks\n                    Specifies the maximum number of ticks for the\n                    particular axis. The actual number of ticks\n                    will be chosen automatically to be less than or\n                    equal to `nticks`. Has an effect only if\n                    `tickmode` is set to "auto".\n                showgrid\n                    Determines whether or not grid lines are drawn.\n                    If True, the grid lines are drawn at every tick\n                    mark.\n                tick0\n                    Sets the placement of the first tick on this\n                    axis. Use with `dtick`. If the axis `type` is\n                    "log", then you must take the log of your\n                    starting tick (e.g. to set the starting tick to\n                    100, set the `tick0` to 2) except when\n                    `dtick`=*L<f>* (see `dtick` for more info). If\n                    the axis `type` is "date", it should be a date\n                    string, like date data. If the axis `type` is\n                    "category", it should be a number, using the\n                    scale where each category is assigned a serial\n                    number from zero in the order it appears.\n                tickcolor\n                    Sets the tick color.\n                ticklen\n                    Sets the tick length (in px).\n                tickmode\n                    Sets the tick mode for this axis. If "auto",\n                    the number of ticks is set via `nticks`. If\n                    "linear", the placement of the ticks is\n                    determined by a starting position `tick0` and a\n                    tick step `dtick` ("linear" is the default\n                    value if `tick0` and `dtick` are provided). If\n                    "array", the placement of the ticks is set via\n                    `tickvals` and the tick text is `ticktext`.\n                    ("array" is the default value if `tickvals` is\n                    provided).\n                ticks\n                    Determines whether ticks are drawn or not. If\n                    "", this axis\' ticks are not drawn. If\n                    "outside" ("inside"), this axis\' are drawn\n                    outside (inside) the axis lines.\n                tickvals\n                    Sets the values at which ticks on this axis\n                    appear. Only has an effect if `tickmode` is set\n                    to "array". Used with `ticktext`.\n                tickvalssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `tickvals`.\n                tickwidth\n                    Sets the tick width (in px).\n\n        Returns\n        -------\n        plotly.graph_objs.layout.yaxis.Minor\n        '
        return self['minor']

    @minor.setter
    def minor(self, val):
        if False:
            return 10
        self['minor'] = val

    @property
    def mirror(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines if the axis lines or/and ticks are mirrored to the\n        opposite side of the plotting area. If True, the axis lines are\n        mirrored. If "ticks", the axis lines and ticks are mirrored. If\n        False, mirroring is disable. If "all", axis lines are mirrored\n        on all shared-axes subplots. If "allticks", axis lines and\n        ticks are mirrored on all shared-axes subplots.\n\n        The \'mirror\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, \'ticks\', False, \'all\', \'allticks\']\n\n        Returns\n        -------\n        Any\n        '
        return self['mirror']

    @mirror.setter
    def mirror(self, val):
        if False:
            print('Hello World!')
        self['mirror'] = val

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
    def overlaying(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If set a same-letter axis id, this axis is overlaid on top of\n        the corresponding same-letter axis, with traces and axes\n        visible for both axes. If False, this axis does not overlay any\n        same-letter axes. In this case, for axes with overlapping\n        domains only the highest-numbered axis will be visible.\n\n        The 'overlaying' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['free']\n          - A string that matches one of the following regular expressions:\n                ['^x([2-9]|[1-9][0-9]+)?( domain)?$',\n                '^y([2-9]|[1-9][0-9]+)?( domain)?$']\n\n        Returns\n        -------\n        Any\n        "
        return self['overlaying']

    @overlaying.setter
    def overlaying(self, val):
        if False:
            return 10
        self['overlaying'] = val

    @property
    def position(self):
        if False:
            print('Hello World!')
        '\n        Sets the position of this axis in the plotting space (in\n        normalized coordinates). Only has an effect if `anchor` is set\n        to "free".\n\n        The \'position\' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        '
        return self['position']

    @position.setter
    def position(self, val):
        if False:
            print('Hello World!')
        self['position'] = val

    @property
    def range(self):
        if False:
            return 10
        '\n            Sets the range of this axis. If the axis `type` is "log", then\n            you must take the log of your desired range (e.g. to set the\n            range from 1 to 100, set the range from 0 to 2). If the axis\n            `type` is "date", it should be date strings, like date data,\n            though Date objects and unix milliseconds will be accepted and\n            converted to strings. If the axis `type` is "category", it\n            should be numbers, using the scale where each category is\n            assigned a serial number from zero in the order it appears.\n            Leaving either or both elements `null` impacts the default\n            `autorange`.\n\n            The \'range\' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The \'range[0]\' property accepts values of any type\n        (1) The \'range[1]\' property accepts values of any type\n\n            Returns\n            -------\n            list\n        '
        return self['range']

    @range.setter
    def range(self, val):
        if False:
            return 10
        self['range'] = val

    @property
    def rangebreaks(self):
        if False:
            while True:
                i = 10
        '\n        The \'rangebreaks\' property is a tuple of instances of\n        Rangebreak that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.layout.yaxis.Rangebreak\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Rangebreak constructor\n\n            Supported dict properties:\n\n                bounds\n                    Sets the lower and upper bounds of this axis\n                    rangebreak. Can be used with `pattern`.\n                dvalue\n                    Sets the size of each `values` item. The\n                    default is one day in milliseconds.\n                enabled\n                    Determines whether this axis rangebreak is\n                    enabled or disabled. Please note that\n                    `rangebreaks` only work for "date" axis type.\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                pattern\n                    Determines a pattern on the time line that\n                    generates breaks. If *day of week* - days of\n                    the week in English e.g. \'Sunday\' or `sun`\n                    (matching is case-insensitive and considers\n                    only the first three characters), as well as\n                    Sunday-based integers between 0 and 6. If\n                    "hour" - hour (24-hour clock) as decimal\n                    numbers between 0 and 24. for more info.\n                    Examples: - { pattern: \'day of week\', bounds:\n                    [6, 1] }  or simply { bounds: [\'sat\', \'mon\'] }\n                    breaks from Saturday to Monday (i.e. skips the\n                    weekends). - { pattern: \'hour\', bounds: [17, 8]\n                    }   breaks from 5pm to 8am (i.e. skips non-work\n                    hours).\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n                values\n                    Sets the coordinate values corresponding to the\n                    rangebreaks. An alternative to `bounds`. Use\n                    `dvalue` to set the size of the values along\n                    the axis.\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.layout.yaxis.Rangebreak]\n        '
        return self['rangebreaks']

    @rangebreaks.setter
    def rangebreaks(self, val):
        if False:
            print('Hello World!')
        self['rangebreaks'] = val

    @property
    def rangebreakdefaults(self):
        if False:
            return 10
        "\n        When used in a template (as\n        layout.template.layout.yaxis.rangebreakdefaults), sets the\n        default property values to use for elements of\n        layout.yaxis.rangebreaks\n\n        The 'rangebreakdefaults' property is an instance of Rangebreak\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.yaxis.Rangebreak`\n          - A dict of string/value properties that will be passed\n            to the Rangebreak constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.layout.yaxis.Rangebreak\n        "
        return self['rangebreakdefaults']

    @rangebreakdefaults.setter
    def rangebreakdefaults(self, val):
        if False:
            i = 10
            return i + 15
        self['rangebreakdefaults'] = val

    @property
    def rangemode(self):
        if False:
            return 10
        '\n        If "normal", the range is computed in relation to the extrema\n        of the input data. If *tozero*`, the range extends to 0,\n        regardless of the input data If "nonnegative", the range is\n        non-negative, regardless of the input data. Applies only to\n        linear axes.\n\n        The \'rangemode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'normal\', \'tozero\', \'nonnegative\']\n\n        Returns\n        -------\n        Any\n        '
        return self['rangemode']

    @rangemode.setter
    def rangemode(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['rangemode'] = val

    @property
    def scaleanchor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If set to another axis id (e.g. `x2`, `y`), the range of this\n        axis changes together with the range of the corresponding axis\n        such that the scale of pixels per unit is in a constant ratio.\n        Both axes are still zoomable, but when you zoom one, the other\n        will zoom the same amount, keeping a fixed midpoint.\n        `constrain` and `constraintoward` determine how we enforce the\n        constraint. You can chain these, ie `yaxis: {scaleanchor: *x*},\n        xaxis2: {scaleanchor: *y*}` but you can only link axes of the\n        same `type`. The linked axis can have the opposite letter (to\n        constrain the aspect ratio) or the same letter (to match scales\n        across subplots). Loops (`yaxis: {scaleanchor: *x*}, xaxis:\n        {scaleanchor: *y*}` or longer) are redundant and the last\n        constraint encountered will be ignored to avoid possible\n        inconsistent constraints via `scaleratio`. Note that setting\n        axes simultaneously in both a `scaleanchor` and a `matches`\n        constraint is currently forbidden. Setting `false` allows to\n        remove a default constraint (occasionally, you may need to\n        prevent a default `scaleanchor` constraint from being applied,\n        eg. when having an image trace `yaxis: {scaleanchor: "x"}` is\n        set automatically in order for pixels to be rendered as\n        squares, setting `yaxis: {scaleanchor: false}` allows to remove\n        the constraint).\n\n        The \'scaleanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [False]\n          - A string that matches one of the following regular expressions:\n                [\'^x([2-9]|[1-9][0-9]+)?( domain)?$\',\n                \'^y([2-9]|[1-9][0-9]+)?( domain)?$\']\n\n        Returns\n        -------\n        Any\n        '
        return self['scaleanchor']

    @scaleanchor.setter
    def scaleanchor(self, val):
        if False:
            return 10
        self['scaleanchor'] = val

    @property
    def scaleratio(self):
        if False:
            return 10
        "\n        If this axis is linked to another by `scaleanchor`, this\n        determines the pixel to unit scale ratio. For example, if this\n        value is 10, then every unit on this axis spans 10 times the\n        number of pixels as a unit on the linked axis. Use this for\n        example to create an elevation profile where the vertical scale\n        is exaggerated a fixed amount with respect to the horizontal.\n\n        The 'scaleratio' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['scaleratio']

    @scaleratio.setter
    def scaleratio(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['scaleratio'] = val

    @property
    def separatethousands(self):
        if False:
            i = 10
            return i + 15
        '\n        If "true", even 4-digit integers are separated\n\n        The \'separatethousands\' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        '
        return self['separatethousands']

    @separatethousands.setter
    def separatethousands(self, val):
        if False:
            while True:
                i = 10
        self['separatethousands'] = val

    @property
    def shift(self):
        if False:
            while True:
                i = 10
        '\n        Moves the axis a given number of pixels from where it would\n        have been otherwise. Accepts both positive and negative values,\n        which will shift the axis either right or left, respectively.\n        If `autoshift` is set to true, then this defaults to a padding\n        of -3 if `side` is set to "left". and defaults to +3 if `side`\n        is set to "right". Defaults to 0 if `autoshift` is set to\n        false. Only has an effect if `anchor` is set to "free".\n\n        The \'shift\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['shift']

    @shift.setter
    def shift(self, val):
        if False:
            print('Hello World!')
        self['shift'] = val

    @property
    def showdividers(self):
        if False:
            print('Hello World!')
        '\n        Determines whether or not a dividers are drawn between the\n        category levels of this axis. Only has an effect on\n        "multicategory" axes.\n\n        The \'showdividers\' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        '
        return self['showdividers']

    @showdividers.setter
    def showdividers(self, val):
        if False:
            print('Hello World!')
        self['showdividers'] = val

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
            print('Hello World!')
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
            while True:
                i = 10
        "\n        Determines whether or not a line bounding this axis is drawn.\n\n        The 'showline' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showline']

    @showline.setter
    def showline(self, val):
        if False:
            return 10
        self['showline'] = val

    @property
    def showspikes(self):
        if False:
            return 10
        "\n        Determines whether or not spikes (aka droplines) are drawn for\n        this axis. Note: This only takes affect when hovermode =\n        closest\n\n        The 'showspikes' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showspikes']

    @showspikes.setter
    def showspikes(self, val):
        if False:
            return 10
        self['showspikes'] = val

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
            while True:
                i = 10
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
            while True:
                i = 10
        self['showticksuffix'] = val

    @property
    def side(self):
        if False:
            i = 10
            return i + 15
        '\n        Determines whether a x (y) axis is positioned at the "bottom"\n        ("left") or "top" ("right") of the plotting area.\n\n        The \'side\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'top\', \'bottom\', \'left\', \'right\']\n\n        Returns\n        -------\n        Any\n        '
        return self['side']

    @side.setter
    def side(self, val):
        if False:
            print('Hello World!')
        self['side'] = val

    @property
    def spikecolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the spike color. If undefined, will use the series color\n\n        The 'spikecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['spikecolor']

    @spikecolor.setter
    def spikecolor(self, val):
        if False:
            print('Hello World!')
        self['spikecolor'] = val

    @property
    def spikedash(self):
        if False:
            while True:
                i = 10
        '\n        Sets the dash style of lines. Set to a dash type string\n        ("solid", "dot", "dash", "longdash", "dashdot", or\n        "longdashdot") or a dash length list in px (eg\n        "5px,10px,2px,2px").\n\n        The \'spikedash\' property is an enumeration that may be specified as:\n          - One of the following dash styles:\n                [\'solid\', \'dot\', \'dash\', \'longdash\', \'dashdot\', \'longdashdot\']\n          - A string containing a dash length list in pixels or percentages\n                (e.g. \'5px 10px 2px 2px\', \'5, 10, 2, 2\', \'10% 20% 40%\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['spikedash']

    @spikedash.setter
    def spikedash(self, val):
        if False:
            return 10
        self['spikedash'] = val

    @property
    def spikemode(self):
        if False:
            i = 10
            return i + 15
        '\n        Determines the drawing mode for the spike line If "toaxis", the\n        line is drawn from the data point to the axis the  series is\n        plotted on. If "across", the line is drawn across the entire\n        plot area, and supercedes "toaxis". If "marker", then a marker\n        dot is drawn on the axis the series is plotted on\n\n        The \'spikemode\' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of [\'toaxis\', \'across\', \'marker\'] joined with \'+\' characters\n            (e.g. \'toaxis+across\')\n\n        Returns\n        -------\n        Any\n        '
        return self['spikemode']

    @spikemode.setter
    def spikemode(self, val):
        if False:
            while True:
                i = 10
        self['spikemode'] = val

    @property
    def spikesnap(self):
        if False:
            return 10
        "\n        Determines whether spikelines are stuck to the cursor or to the\n        closest datapoints.\n\n        The 'spikesnap' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['data', 'cursor', 'hovered data']\n\n        Returns\n        -------\n        Any\n        "
        return self['spikesnap']

    @spikesnap.setter
    def spikesnap(self, val):
        if False:
            print('Hello World!')
        self['spikesnap'] = val

    @property
    def spikethickness(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the width (in px) of the zero line.\n\n        The 'spikethickness' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['spikethickness']

    @spikethickness.setter
    def spikethickness(self, val):
        if False:
            i = 10
            return i + 15
        self['spikethickness'] = val

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
            while True:
                i = 10
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
            print('Hello World!')
        self['tickangle'] = val

    @property
    def tickcolor(self):
        if False:
            i = 10
            return i + 15
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
            return 10
        '\n        Sets the tick font.\n\n        The \'tickfont\' property is an instance of Tickfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.yaxis.Tickfont`\n          - A dict of string/value properties that will be passed\n            to the Tickfont constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.yaxis.Tickfont\n        '
        return self['tickfont']

    @tickfont.setter
    def tickfont(self, val):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
        self['tickformat'] = val

    @property
    def tickformatstops(self):
        if False:
            print('Hello World!')
        '\n        The \'tickformatstops\' property is a tuple of instances of\n        Tickformatstop that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.layout.yaxis.Tickformatstop\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Tickformatstop constructor\n\n            Supported dict properties:\n\n                dtickrange\n                    range [*min*, *max*], where "min", "max" -\n                    dtick values which describe some zoom level, it\n                    is possible to omit "min" or "max" value by\n                    passing "null"\n                enabled\n                    Determines whether or not this stop is used. If\n                    `false`, this stop is ignored even within its\n                    `dtickrange`.\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n                value\n                    string - dtickformat for described zoom level,\n                    the same as "tickformat"\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.layout.yaxis.Tickformatstop]\n        '
        return self['tickformatstops']

    @tickformatstops.setter
    def tickformatstops(self, val):
        if False:
            print('Hello World!')
        self['tickformatstops'] = val

    @property
    def tickformatstopdefaults(self):
        if False:
            print('Hello World!')
        "\n        When used in a template (as\n        layout.template.layout.yaxis.tickformatstopdefaults), sets the\n        default property values to use for elements of\n        layout.yaxis.tickformatstops\n\n        The 'tickformatstopdefaults' property is an instance of Tickformatstop\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.yaxis.Tickformatstop`\n          - A dict of string/value properties that will be passed\n            to the Tickformatstop constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.layout.yaxis.Tickformatstop\n        "
        return self['tickformatstopdefaults']

    @tickformatstopdefaults.setter
    def tickformatstopdefaults(self, val):
        if False:
            while True:
                i = 10
        self['tickformatstopdefaults'] = val

    @property
    def ticklabelmode(self):
        if False:
            while True:
                i = 10
        '\n        Determines where tick labels are drawn with respect to their\n        corresponding ticks and grid lines. Only has an effect for axes\n        of `type` "date" When set to "period", tick labels are drawn in\n        the middle of the period between ticks.\n\n        The \'ticklabelmode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'instant\', \'period\']\n\n        Returns\n        -------\n        Any\n        '
        return self['ticklabelmode']

    @ticklabelmode.setter
    def ticklabelmode(self, val):
        if False:
            return 10
        self['ticklabelmode'] = val

    @property
    def ticklabeloverflow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines how we handle tick labels that would overflow either\n        the graph div or the domain of the axis. The default value for\n        inside tick labels is *hide past domain*. Otherwise on\n        "category" and "multicategory" axes the default is "allow". In\n        other cases the default is *hide past div*.\n\n        The \'ticklabeloverflow\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'allow\', \'hide past div\', \'hide past domain\']\n\n        Returns\n        -------\n        Any\n        '
        return self['ticklabeloverflow']

    @ticklabeloverflow.setter
    def ticklabeloverflow(self, val):
        if False:
            print('Hello World!')
        self['ticklabeloverflow'] = val

    @property
    def ticklabelposition(self):
        if False:
            while True:
                i = 10
        '\n        Determines where tick labels are drawn with respect to the axis\n        Please note that top or bottom has no effect on x axes or when\n        `ticklabelmode` is set to "period". Similarly left or right has\n        no effect on y axes or when `ticklabelmode` is set to "period".\n        Has no effect on "multicategory" axes or when `tickson` is set\n        to "boundaries". When used on axes linked by `matches` or\n        `scaleanchor`, no extra padding for inside labels would be\n        added by autorange, so that the scales could match.\n\n        The \'ticklabelposition\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'outside\', \'inside\', \'outside top\', \'inside top\',\n                \'outside left\', \'inside left\', \'outside right\', \'inside\n                right\', \'outside bottom\', \'inside bottom\']\n\n        Returns\n        -------\n        Any\n        '
        return self['ticklabelposition']

    @ticklabelposition.setter
    def ticklabelposition(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['ticklabelposition'] = val

    @property
    def ticklabelstep(self):
        if False:
            print('Hello World!')
        '\n        Sets the spacing between tick labels as compared to the spacing\n        between ticks. A value of 1 (default) means each tick gets a\n        label. A value of 2 means shows every 2nd label. A larger value\n        n means only every nth tick is labeled. `tick0` determines\n        which labels are shown. Not implemented for axes with `type`\n        "log" or "multicategory", or when `tickmode` is "array".\n\n        The \'ticklabelstep\' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [1, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        '
        return self['ticklabelstep']

    @ticklabelstep.setter
    def ticklabelstep(self, val):
        if False:
            print('Hello World!')
        self['ticklabelstep'] = val

    @property
    def ticklen(self):
        if False:
            while True:
                i = 10
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
            print('Hello World!')
        '\n        Sets the tick mode for this axis. If "auto", the number of\n        ticks is set via `nticks`. If "linear", the placement of the\n        ticks is determined by a starting position `tick0` and a tick\n        step `dtick` ("linear" is the default value if `tick0` and\n        `dtick` are provided). If "array", the placement of the ticks\n        is set via `tickvals` and the tick text is `ticktext`. ("array"\n        is the default value if `tickvals` is provided). If "sync", the\n        number of ticks will sync with the overlayed axis set by\n        `overlaying` property.\n\n        The \'tickmode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'auto\', \'linear\', \'array\', \'sync\']\n\n        Returns\n        -------\n        Any\n        '
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
            return 10
        self['tickprefix'] = val

    @property
    def ticks(self):
        if False:
            while True:
                i = 10
        '\n        Determines whether ticks are drawn or not. If "", this axis\'\n        ticks are not drawn. If "outside" ("inside"), this axis\' are\n        drawn outside (inside) the axis lines.\n\n        The \'ticks\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'outside\', \'inside\', \'\']\n\n        Returns\n        -------\n        Any\n        '
        return self['ticks']

    @ticks.setter
    def ticks(self, val):
        if False:
            print('Hello World!')
        self['ticks'] = val

    @property
    def tickson(self):
        if False:
            return 10
        '\n        Determines where ticks and grid lines are drawn with respect to\n        their corresponding tick labels. Only has an effect for axes of\n        `type` "category" or "multicategory". When set to "boundaries",\n        ticks and grid lines are drawn half a category to the\n        left/bottom of labels.\n\n        The \'tickson\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'labels\', \'boundaries\']\n\n        Returns\n        -------\n        Any\n        '
        return self['tickson']

    @tickson.setter
    def tickson(self, val):
        if False:
            return 10
        self['tickson'] = val

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
            print('Hello World!')
        self['ticksuffix'] = val

    @property
    def ticktext(self):
        if False:
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `ticktext`.\n\n        The 'ticktextsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['ticktextsrc']

    @ticktextsrc.setter
    def ticktextsrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['ticktextsrc'] = val

    @property
    def tickvals(self):
        if False:
            return 10
        '\n        Sets the values at which ticks on this axis appear. Only has an\n        effect if `tickmode` is set to "array". Used with `ticktext`.\n\n        The \'tickvals\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['tickvals']

    @tickvals.setter
    def tickvals(self, val):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        self['tickwidth'] = val

    @property
    def title(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'title' property is an instance of Title\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.yaxis.Title`\n          - A dict of string/value properties that will be passed\n            to the Title constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this axis' title font. Note that the\n                    title's font used to be customized by the now\n                    deprecated `titlefont` attribute.\n                standoff\n                    Sets the standoff distance (in px) between the\n                    axis labels and the title text The default\n                    value is a function of the axis tick labels,\n                    the title `font.size` and the axis `linewidth`.\n                    Note that the axis title position is always\n                    constrained within the margins, so the actual\n                    standoff distance is always less than the set\n                    or default value. By setting `standoff` and\n                    turning on `automargin`, plotly.js will push\n                    the margins to fit the axis title at given\n                    standoff distance.\n                text\n                    Sets the title of this axis. Note that before\n                    the existence of `title.text`, the title's\n                    contents used to be defined as the `title`\n                    attribute itself. This behavior has been\n                    deprecated.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.yaxis.Title\n        "
        return self['title']

    @title.setter
    def title(self, val):
        if False:
            i = 10
            return i + 15
        self['title'] = val

    @property
    def titlefont(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deprecated: Please use layout.yaxis.title.font instead. Sets\n        this axis\' title font. Note that the title\'s font used to be\n        customized by the now deprecated `titlefont` attribute.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.yaxis.title.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n\n        '
        return self['titlefont']

    @titlefont.setter
    def titlefont(self, val):
        if False:
            while True:
                i = 10
        self['titlefont'] = val

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the axis type. By default, plotly attempts to determined\n        the axis type by looking into the data of the traces that\n        referenced the axis in question.\n\n        The 'type' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['-', 'linear', 'log', 'date', 'category',\n                'multicategory']\n\n        Returns\n        -------\n        Any\n        "
        return self['type']

    @type.setter
    def type(self, val):
        if False:
            return 10
        self['type'] = val

    @property
    def uirevision(self):
        if False:
            print('Hello World!')
        "\n        Controls persistence of user-driven changes in axis `range`,\n        `autorange`, and `title` if in `editable: true` configuration.\n        Defaults to `layout.uirevision`.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['uirevision'] = val

    @property
    def visible(self):
        if False:
            i = 10
            return i + 15
        "\n        A single toggle to hide the axis while preserving interaction\n        like dragging. Default is true when a cheater plot is present\n        on the axis, otherwise false\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            return 10
        self['visible'] = val

    @property
    def zeroline(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not a line is drawn at along the 0 value\n        of this axis. If True, the zero line is drawn on top of the\n        grid lines.\n\n        The 'zeroline' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['zeroline']

    @zeroline.setter
    def zeroline(self, val):
        if False:
            print('Hello World!')
        self['zeroline'] = val

    @property
    def zerolinecolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the line color of the zero line.\n\n        The 'zerolinecolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['zerolinecolor']

    @zerolinecolor.setter
    def zerolinecolor(self, val):
        if False:
            print('Hello World!')
        self['zerolinecolor'] = val

    @property
    def zerolinewidth(self):
        if False:
            return 10
        "\n        Sets the width (in px) of the zero line.\n\n        The 'zerolinewidth' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['zerolinewidth']

    @zerolinewidth.setter
    def zerolinewidth(self, val):
        if False:
            print('Hello World!')
        self['zerolinewidth'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        anchor\n            If set to an opposite-letter axis id (e.g. `x2`, `y`),\n            this axis is bound to the corresponding opposite-letter\n            axis. If set to "free", this axis\' position is\n            determined by `position`.\n        automargin\n            Determines whether long tick labels automatically grow\n            the figure margins.\n        autorange\n            Determines whether or not the range of this axis is\n            computed in relation to the input data. See `rangemode`\n            for more info. If `range` is provided and it has a\n            value for both the lower and upper bound, `autorange`\n            is set to False. Using "min" applies autorange only to\n            set the minimum. Using "max" applies autorange only to\n            set the maximum. Using *min reversed* applies autorange\n            only to set the minimum on a reversed axis. Using *max\n            reversed* applies autorange only to set the maximum on\n            a reversed axis. Using "reversed" applies autorange on\n            both ends and reverses the axis direction.\n        autorangeoptions\n            :class:`plotly.graph_objects.layout.yaxis.Autorangeopti\n            ons` instance or dict with compatible properties\n        autoshift\n            Automatically reposition the axis to avoid overlap with\n            other axes with the same `overlaying` value. This\n            repositioning will account for any `shift` amount\n            applied to other axes on the same side with `autoshift`\n            is set to true. Only has an effect if `anchor` is set\n            to "free".\n        autotypenumbers\n            Using "strict" a numeric string in trace data is not\n            converted to a number. Using *convert types* a numeric\n            string in trace data may be treated as a number during\n            automatic axis `type` detection. Defaults to\n            layout.autotypenumbers.\n        calendar\n            Sets the calendar system to use for `range` and `tick0`\n            if this is a date axis. This does not set the calendar\n            for interpreting data on this axis, that\'s specified in\n            the trace or via the global `layout.calendar`\n        categoryarray\n            Sets the order in which categories on this axis appear.\n            Only has an effect if `categoryorder` is set to\n            "array". Used with `categoryorder`.\n        categoryarraysrc\n            Sets the source reference on Chart Studio Cloud for\n            `categoryarray`.\n        categoryorder\n            Specifies the ordering logic for the case of\n            categorical variables. By default, plotly uses "trace",\n            which specifies the order that is present in the data\n            supplied. Set `categoryorder` to *category ascending*\n            or *category descending* if order should be determined\n            by the alphanumerical order of the category names. Set\n            `categoryorder` to "array" to derive the ordering from\n            the attribute `categoryarray`. If a category is not\n            found in the `categoryarray` array, the sorting\n            behavior for that attribute will be identical to the\n            "trace" mode. The unspecified categories will follow\n            the categories in `categoryarray`. Set `categoryorder`\n            to *total ascending* or *total descending* if order\n            should be determined by the numerical order of the\n            values. Similarly, the order can be determined by the\n            min, max, sum, mean or median of all the values.\n        color\n            Sets default for all colors associated with this axis\n            all at once: line, font, tick, and grid colors. Grid\n            color is lightened by blending this with the plot\n            background Individual pieces can override this.\n        constrain\n            If this axis needs to be compressed (either due to its\n            own `scaleanchor` and `scaleratio` or those of the\n            other axis), determines how that happens: by increasing\n            the "range", or by decreasing the "domain". Default is\n            "domain" for axes containing image traces, "range"\n            otherwise.\n        constraintoward\n            If this axis needs to be compressed (either due to its\n            own `scaleanchor` and `scaleratio` or those of the\n            other axis), determines which direction we push the\n            originally specified plot area. Options are "left",\n            "center" (default), and "right" for x axes, and "top",\n            "middle" (default), and "bottom" for y axes.\n        dividercolor\n            Sets the color of the dividers Only has an effect on\n            "multicategory" axes.\n        dividerwidth\n            Sets the width (in px) of the dividers Only has an\n            effect on "multicategory" axes.\n        domain\n            Sets the domain of this axis (in plot fraction).\n        dtick\n            Sets the step in-between ticks on this axis. Use with\n            `tick0`. Must be a positive number, or special strings\n            available to "log" and "date" axes. If the axis `type`\n            is "log", then ticks are set every 10^(n*dtick) where n\n            is the tick number. For example, to set a tick mark at\n            1, 10, 100, 1000, ... set dtick to 1. To set tick marks\n            at 1, 100, 10000, ... set dtick to 2. To set tick marks\n            at 1, 5, 25, 125, 625, 3125, ... set dtick to\n            log_10(5), or 0.69897000433. "log" has several special\n            values; "L<f>", where `f` is a positive number, gives\n            ticks linearly spaced in value (but not position). For\n            example `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n            at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus\n            small digits between, use "D1" (all digits) or "D2"\n            (only 2 and 5). `tick0` is ignored for "D1" and "D2".\n            If the axis `type` is "date", then you must convert the\n            time to milliseconds. For example, to set the interval\n            between ticks to one day, set `dtick` to 86400000.0.\n            "date" also has special values "M<n>" gives ticks\n            spaced by a number of months. `n` must be a positive\n            integer. To set ticks on the 15th of every third month,\n            set `tick0` to "2000-01-15" and `dtick` to "M3". To set\n            ticks every 4 years, set `dtick` to "M48"\n        exponentformat\n            Determines a formatting rule for the tick exponents.\n            For example, consider the number 1,000,000,000. If\n            "none", it appears as 1,000,000,000. If "e", 1e+9. If\n            "E", 1E+9. If "power", 1x10^9 (with 9 in a super\n            script). If "SI", 1G. If "B", 1B.\n        fixedrange\n            Determines whether or not this axis is zoom-able. If\n            true, then zoom is disabled.\n        gridcolor\n            Sets the color of the grid lines.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the grid lines.\n        hoverformat\n            Sets the hover text formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        insiderange\n            Could be used to set the desired inside range of this\n            axis (excluding the labels) when `ticklabelposition` of\n            the anchored axis has "inside". Not implemented for\n            axes with `type` "log". This would be ignored when\n            `range` is provided.\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        layer\n            Sets the layer on which this axis is displayed. If\n            *above traces*, this axis is displayed above all the\n            subplot\'s traces If *below traces*, this axis is\n            displayed below all the subplot\'s traces, but above the\n            grid lines. Useful when used together with scatter-like\n            traces with `cliponaxis` set to False to show markers\n            and/or text nodes above this axis.\n        linecolor\n            Sets the axis line color.\n        linewidth\n            Sets the width (in px) of the axis line.\n        matches\n            If set to another axis id (e.g. `x2`, `y`), the range\n            of this axis will match the range of the corresponding\n            axis in data-coordinates space. Moreover, matching axes\n            share auto-range values, category lists and histogram\n            auto-bins. Note that setting axes simultaneously in\n            both a `scaleanchor` and a `matches` constraint is\n            currently forbidden. Moreover, note that matching axes\n            must have the same `type`.\n        maxallowed\n            Determines the maximum range of this axis.\n        minallowed\n            Determines the minimum range of this axis.\n        minexponent\n            Hide SI prefix for 10^n if |n| is below this number.\n            This only has an effect when `tickformat` is "SI" or\n            "B".\n        minor\n            :class:`plotly.graph_objects.layout.yaxis.Minor`\n            instance or dict with compatible properties\n        mirror\n            Determines if the axis lines or/and ticks are mirrored\n            to the opposite side of the plotting area. If True, the\n            axis lines are mirrored. If "ticks", the axis lines and\n            ticks are mirrored. If False, mirroring is disable. If\n            "all", axis lines are mirrored on all shared-axes\n            subplots. If "allticks", axis lines and ticks are\n            mirrored on all shared-axes subplots.\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        overlaying\n            If set a same-letter axis id, this axis is overlaid on\n            top of the corresponding same-letter axis, with traces\n            and axes visible for both axes. If False, this axis\n            does not overlay any same-letter axes. In this case,\n            for axes with overlapping domains only the highest-\n            numbered axis will be visible.\n        position\n            Sets the position of this axis in the plotting space\n            (in normalized coordinates). Only has an effect if\n            `anchor` is set to "free".\n        range\n            Sets the range of this axis. If the axis `type` is\n            "log", then you must take the log of your desired range\n            (e.g. to set the range from 1 to 100, set the range\n            from 0 to 2). If the axis `type` is "date", it should\n            be date strings, like date data, though Date objects\n            and unix milliseconds will be accepted and converted to\n            strings. If the axis `type` is "category", it should be\n            numbers, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears. Leaving either or both elements `null` impacts\n            the default `autorange`.\n        rangebreaks\n            A tuple of\n            :class:`plotly.graph_objects.layout.yaxis.Rangebreak`\n            instances or dicts with compatible properties\n        rangebreakdefaults\n            When used in a template (as\n            layout.template.layout.yaxis.rangebreakdefaults), sets\n            the default property values to use for elements of\n            layout.yaxis.rangebreaks\n        rangemode\n            If "normal", the range is computed in relation to the\n            extrema of the input data. If *tozero*`, the range\n            extends to 0, regardless of the input data If\n            "nonnegative", the range is non-negative, regardless of\n            the input data. Applies only to linear axes.\n        scaleanchor\n            If set to another axis id (e.g. `x2`, `y`), the range\n            of this axis changes together with the range of the\n            corresponding axis such that the scale of pixels per\n            unit is in a constant ratio. Both axes are still\n            zoomable, but when you zoom one, the other will zoom\n            the same amount, keeping a fixed midpoint. `constrain`\n            and `constraintoward` determine how we enforce the\n            constraint. You can chain these, ie `yaxis:\n            {scaleanchor: *x*}, xaxis2: {scaleanchor: *y*}` but you\n            can only link axes of the same `type`. The linked axis\n            can have the opposite letter (to constrain the aspect\n            ratio) or the same letter (to match scales across\n            subplots). Loops (`yaxis: {scaleanchor: *x*}, xaxis:\n            {scaleanchor: *y*}` or longer) are redundant and the\n            last constraint encountered will be ignored to avoid\n            possible inconsistent constraints via `scaleratio`.\n            Note that setting axes simultaneously in both a\n            `scaleanchor` and a `matches` constraint is currently\n            forbidden. Setting `false` allows to remove a default\n            constraint (occasionally, you may need to prevent a\n            default `scaleanchor` constraint from being applied,\n            eg. when having an image trace `yaxis: {scaleanchor:\n            "x"}` is set automatically in order for pixels to be\n            rendered as squares, setting `yaxis: {scaleanchor:\n            false}` allows to remove the constraint).\n        scaleratio\n            If this axis is linked to another by `scaleanchor`,\n            this determines the pixel to unit scale ratio. For\n            example, if this value is 10, then every unit on this\n            axis spans 10 times the number of pixels as a unit on\n            the linked axis. Use this for example to create an\n            elevation profile where the vertical scale is\n            exaggerated a fixed amount with respect to the\n            horizontal.\n        separatethousands\n            If "true", even 4-digit integers are separated\n        shift\n            Moves the axis a given number of pixels from where it\n            would have been otherwise. Accepts both positive and\n            negative values, which will shift the axis either right\n            or left, respectively. If `autoshift` is set to true,\n            then this defaults to a padding of -3 if `side` is set\n            to "left". and defaults to +3 if `side` is set to\n            "right". Defaults to 0 if `autoshift` is set to false.\n            Only has an effect if `anchor` is set to "free".\n        showdividers\n            Determines whether or not a dividers are drawn between\n            the category levels of this axis. Only has an effect on\n            "multicategory" axes.\n        showexponent\n            If "all", all exponents are shown besides their\n            significands. If "first", only the exponent of the\n            first tick is shown. If "last", only the exponent of\n            the last tick is shown. If "none", no exponents appear.\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        showline\n            Determines whether or not a line bounding this axis is\n            drawn.\n        showspikes\n            Determines whether or not spikes (aka droplines) are\n            drawn for this axis. Note: This only takes affect when\n            hovermode = closest\n        showticklabels\n            Determines whether or not the tick labels are drawn.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        side\n            Determines whether a x (y) axis is positioned at the\n            "bottom" ("left") or "top" ("right") of the plotting\n            area.\n        spikecolor\n            Sets the spike color. If undefined, will use the series\n            color\n        spikedash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        spikemode\n            Determines the drawing mode for the spike line If\n            "toaxis", the line is drawn from the data point to the\n            axis the  series is plotted on. If "across", the line\n            is drawn across the entire plot area, and supercedes\n            "toaxis". If "marker", then a marker dot is drawn on\n            the axis the series is plotted on\n        spikesnap\n            Determines whether spikelines are stuck to the cursor\n            or to the closest datapoints.\n        spikethickness\n            Sets the width (in px) of the zero line.\n        tick0\n            Sets the placement of the first tick on this axis. Use\n            with `dtick`. If the axis `type` is "log", then you\n            must take the log of your starting tick (e.g. to set\n            the starting tick to 100, set the `tick0` to 2) except\n            when `dtick`=*L<f>* (see `dtick` for more info). If the\n            axis `type` is "date", it should be a date string, like\n            date data. If the axis `type` is "category", it should\n            be a number, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        tickangle\n            Sets the angle of the tick labels with respect to the\n            horizontal. For example, a `tickangle` of -90 draws the\n            tick labels vertically.\n        tickcolor\n            Sets the tick color.\n        tickfont\n            Sets the tick font.\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        tickformatstops\n            A tuple of :class:`plotly.graph_objects.layout.yaxis.Ti\n            ckformatstop` instances or dicts with compatible\n            properties\n        tickformatstopdefaults\n            When used in a template (as\n            layout.template.layout.yaxis.tickformatstopdefaults),\n            sets the default property values to use for elements of\n            layout.yaxis.tickformatstops\n        ticklabelmode\n            Determines where tick labels are drawn with respect to\n            their corresponding ticks and grid lines. Only has an\n            effect for axes of `type` "date" When set to "period",\n            tick labels are drawn in the middle of the period\n            between ticks.\n        ticklabeloverflow\n            Determines how we handle tick labels that would\n            overflow either the graph div or the domain of the\n            axis. The default value for inside tick labels is *hide\n            past domain*. Otherwise on "category" and\n            "multicategory" axes the default is "allow". In other\n            cases the default is *hide past div*.\n        ticklabelposition\n            Determines where tick labels are drawn with respect to\n            the axis Please note that top or bottom has no effect\n            on x axes or when `ticklabelmode` is set to "period".\n            Similarly left or right has no effect on y axes or when\n            `ticklabelmode` is set to "period". Has no effect on\n            "multicategory" axes or when `tickson` is set to\n            "boundaries". When used on axes linked by `matches` or\n            `scaleanchor`, no extra padding for inside labels would\n            be added by autorange, so that the scales could match.\n        ticklabelstep\n            Sets the spacing between tick labels as compared to the\n            spacing between ticks. A value of 1 (default) means\n            each tick gets a label. A value of 2 means shows every\n            2nd label. A larger value n means only every nth tick\n            is labeled. `tick0` determines which labels are shown.\n            Not implemented for axes with `type` "log" or\n            "multicategory", or when `tickmode` is "array".\n        ticklen\n            Sets the tick length (in px).\n        tickmode\n            Sets the tick mode for this axis. If "auto", the number\n            of ticks is set via `nticks`. If "linear", the\n            placement of the ticks is determined by a starting\n            position `tick0` and a tick step `dtick` ("linear" is\n            the default value if `tick0` and `dtick` are provided).\n            If "array", the placement of the ticks is set via\n            `tickvals` and the tick text is `ticktext`. ("array" is\n            the default value if `tickvals` is provided). If\n            "sync", the number of ticks will sync with the\n            overlayed axis set by `overlaying` property.\n        tickprefix\n            Sets a tick label prefix.\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        tickson\n            Determines where ticks and grid lines are drawn with\n            respect to their corresponding tick labels. Only has an\n            effect for axes of `type` "category" or\n            "multicategory". When set to "boundaries", ticks and\n            grid lines are drawn half a category to the left/bottom\n            of labels.\n        ticksuffix\n            Sets a tick label suffix.\n        ticktext\n            Sets the text displayed at the ticks position via\n            `tickvals`. Only has an effect if `tickmode` is set to\n            "array". Used with `tickvals`.\n        ticktextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `ticktext`.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n        title\n            :class:`plotly.graph_objects.layout.yaxis.Title`\n            instance or dict with compatible properties\n        titlefont\n            Deprecated: Please use layout.yaxis.title.font instead.\n            Sets this axis\' title font. Note that the title\'s font\n            used to be customized by the now deprecated `titlefont`\n            attribute.\n        type\n            Sets the axis type. By default, plotly attempts to\n            determined the axis type by looking into the data of\n            the traces that referenced the axis in question.\n        uirevision\n            Controls persistence of user-driven changes in axis\n            `range`, `autorange`, and `title` if in `editable:\n            true` configuration. Defaults to `layout.uirevision`.\n        visible\n            A single toggle to hide the axis while preserving\n            interaction like dragging. Default is true when a\n            cheater plot is present on the axis, otherwise false\n        zeroline\n            Determines whether or not a line is drawn at along the\n            0 value of this axis. If True, the zero line is drawn\n            on top of the grid lines.\n        zerolinecolor\n            Sets the line color of the zero line.\n        zerolinewidth\n            Sets the width (in px) of the zero line.\n        '
    _mapped_properties = {'titlefont': ('title', 'font')}

    def __init__(self, arg=None, anchor=None, automargin=None, autorange=None, autorangeoptions=None, autoshift=None, autotypenumbers=None, calendar=None, categoryarray=None, categoryarraysrc=None, categoryorder=None, color=None, constrain=None, constraintoward=None, dividercolor=None, dividerwidth=None, domain=None, dtick=None, exponentformat=None, fixedrange=None, gridcolor=None, griddash=None, gridwidth=None, hoverformat=None, insiderange=None, labelalias=None, layer=None, linecolor=None, linewidth=None, matches=None, maxallowed=None, minallowed=None, minexponent=None, minor=None, mirror=None, nticks=None, overlaying=None, position=None, range=None, rangebreaks=None, rangebreakdefaults=None, rangemode=None, scaleanchor=None, scaleratio=None, separatethousands=None, shift=None, showdividers=None, showexponent=None, showgrid=None, showline=None, showspikes=None, showticklabels=None, showtickprefix=None, showticksuffix=None, side=None, spikecolor=None, spikedash=None, spikemode=None, spikesnap=None, spikethickness=None, tick0=None, tickangle=None, tickcolor=None, tickfont=None, tickformat=None, tickformatstops=None, tickformatstopdefaults=None, ticklabelmode=None, ticklabeloverflow=None, ticklabelposition=None, ticklabelstep=None, ticklen=None, tickmode=None, tickprefix=None, ticks=None, tickson=None, ticksuffix=None, ticktext=None, ticktextsrc=None, tickvals=None, tickvalssrc=None, tickwidth=None, title=None, titlefont=None, type=None, uirevision=None, visible=None, zeroline=None, zerolinecolor=None, zerolinewidth=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new YAxis object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.layout.YAxis`\n        anchor\n            If set to an opposite-letter axis id (e.g. `x2`, `y`),\n            this axis is bound to the corresponding opposite-letter\n            axis. If set to "free", this axis\' position is\n            determined by `position`.\n        automargin\n            Determines whether long tick labels automatically grow\n            the figure margins.\n        autorange\n            Determines whether or not the range of this axis is\n            computed in relation to the input data. See `rangemode`\n            for more info. If `range` is provided and it has a\n            value for both the lower and upper bound, `autorange`\n            is set to False. Using "min" applies autorange only to\n            set the minimum. Using "max" applies autorange only to\n            set the maximum. Using *min reversed* applies autorange\n            only to set the minimum on a reversed axis. Using *max\n            reversed* applies autorange only to set the maximum on\n            a reversed axis. Using "reversed" applies autorange on\n            both ends and reverses the axis direction.\n        autorangeoptions\n            :class:`plotly.graph_objects.layout.yaxis.Autorangeopti\n            ons` instance or dict with compatible properties\n        autoshift\n            Automatically reposition the axis to avoid overlap with\n            other axes with the same `overlaying` value. This\n            repositioning will account for any `shift` amount\n            applied to other axes on the same side with `autoshift`\n            is set to true. Only has an effect if `anchor` is set\n            to "free".\n        autotypenumbers\n            Using "strict" a numeric string in trace data is not\n            converted to a number. Using *convert types* a numeric\n            string in trace data may be treated as a number during\n            automatic axis `type` detection. Defaults to\n            layout.autotypenumbers.\n        calendar\n            Sets the calendar system to use for `range` and `tick0`\n            if this is a date axis. This does not set the calendar\n            for interpreting data on this axis, that\'s specified in\n            the trace or via the global `layout.calendar`\n        categoryarray\n            Sets the order in which categories on this axis appear.\n            Only has an effect if `categoryorder` is set to\n            "array". Used with `categoryorder`.\n        categoryarraysrc\n            Sets the source reference on Chart Studio Cloud for\n            `categoryarray`.\n        categoryorder\n            Specifies the ordering logic for the case of\n            categorical variables. By default, plotly uses "trace",\n            which specifies the order that is present in the data\n            supplied. Set `categoryorder` to *category ascending*\n            or *category descending* if order should be determined\n            by the alphanumerical order of the category names. Set\n            `categoryorder` to "array" to derive the ordering from\n            the attribute `categoryarray`. If a category is not\n            found in the `categoryarray` array, the sorting\n            behavior for that attribute will be identical to the\n            "trace" mode. The unspecified categories will follow\n            the categories in `categoryarray`. Set `categoryorder`\n            to *total ascending* or *total descending* if order\n            should be determined by the numerical order of the\n            values. Similarly, the order can be determined by the\n            min, max, sum, mean or median of all the values.\n        color\n            Sets default for all colors associated with this axis\n            all at once: line, font, tick, and grid colors. Grid\n            color is lightened by blending this with the plot\n            background Individual pieces can override this.\n        constrain\n            If this axis needs to be compressed (either due to its\n            own `scaleanchor` and `scaleratio` or those of the\n            other axis), determines how that happens: by increasing\n            the "range", or by decreasing the "domain". Default is\n            "domain" for axes containing image traces, "range"\n            otherwise.\n        constraintoward\n            If this axis needs to be compressed (either due to its\n            own `scaleanchor` and `scaleratio` or those of the\n            other axis), determines which direction we push the\n            originally specified plot area. Options are "left",\n            "center" (default), and "right" for x axes, and "top",\n            "middle" (default), and "bottom" for y axes.\n        dividercolor\n            Sets the color of the dividers Only has an effect on\n            "multicategory" axes.\n        dividerwidth\n            Sets the width (in px) of the dividers Only has an\n            effect on "multicategory" axes.\n        domain\n            Sets the domain of this axis (in plot fraction).\n        dtick\n            Sets the step in-between ticks on this axis. Use with\n            `tick0`. Must be a positive number, or special strings\n            available to "log" and "date" axes. If the axis `type`\n            is "log", then ticks are set every 10^(n*dtick) where n\n            is the tick number. For example, to set a tick mark at\n            1, 10, 100, 1000, ... set dtick to 1. To set tick marks\n            at 1, 100, 10000, ... set dtick to 2. To set tick marks\n            at 1, 5, 25, 125, 625, 3125, ... set dtick to\n            log_10(5), or 0.69897000433. "log" has several special\n            values; "L<f>", where `f` is a positive number, gives\n            ticks linearly spaced in value (but not position). For\n            example `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n            at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus\n            small digits between, use "D1" (all digits) or "D2"\n            (only 2 and 5). `tick0` is ignored for "D1" and "D2".\n            If the axis `type` is "date", then you must convert the\n            time to milliseconds. For example, to set the interval\n            between ticks to one day, set `dtick` to 86400000.0.\n            "date" also has special values "M<n>" gives ticks\n            spaced by a number of months. `n` must be a positive\n            integer. To set ticks on the 15th of every third month,\n            set `tick0` to "2000-01-15" and `dtick` to "M3". To set\n            ticks every 4 years, set `dtick` to "M48"\n        exponentformat\n            Determines a formatting rule for the tick exponents.\n            For example, consider the number 1,000,000,000. If\n            "none", it appears as 1,000,000,000. If "e", 1e+9. If\n            "E", 1E+9. If "power", 1x10^9 (with 9 in a super\n            script). If "SI", 1G. If "B", 1B.\n        fixedrange\n            Determines whether or not this axis is zoom-able. If\n            true, then zoom is disabled.\n        gridcolor\n            Sets the color of the grid lines.\n        griddash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        gridwidth\n            Sets the width (in px) of the grid lines.\n        hoverformat\n            Sets the hover text formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        insiderange\n            Could be used to set the desired inside range of this\n            axis (excluding the labels) when `ticklabelposition` of\n            the anchored axis has "inside". Not implemented for\n            axes with `type` "log". This would be ignored when\n            `range` is provided.\n        labelalias\n            Replacement text for specific tick or hover labels. For\n            example using {US: \'USA\', CA: \'Canada\'} changes US to\n            USA and CA to Canada. The labels we would have shown\n            must match the keys exactly, after adding any\n            tickprefix or ticksuffix. For negative numbers the\n            minus sign symbol used (U+2212) is wider than the\n            regular ascii dash. That means you need to use −1\n            instead of -1. labelalias can be used with any axis\n            type, and both keys (if needed) and values (if desired)\n            can include html-like tags or MathJax.\n        layer\n            Sets the layer on which this axis is displayed. If\n            *above traces*, this axis is displayed above all the\n            subplot\'s traces If *below traces*, this axis is\n            displayed below all the subplot\'s traces, but above the\n            grid lines. Useful when used together with scatter-like\n            traces with `cliponaxis` set to False to show markers\n            and/or text nodes above this axis.\n        linecolor\n            Sets the axis line color.\n        linewidth\n            Sets the width (in px) of the axis line.\n        matches\n            If set to another axis id (e.g. `x2`, `y`), the range\n            of this axis will match the range of the corresponding\n            axis in data-coordinates space. Moreover, matching axes\n            share auto-range values, category lists and histogram\n            auto-bins. Note that setting axes simultaneously in\n            both a `scaleanchor` and a `matches` constraint is\n            currently forbidden. Moreover, note that matching axes\n            must have the same `type`.\n        maxallowed\n            Determines the maximum range of this axis.\n        minallowed\n            Determines the minimum range of this axis.\n        minexponent\n            Hide SI prefix for 10^n if |n| is below this number.\n            This only has an effect when `tickformat` is "SI" or\n            "B".\n        minor\n            :class:`plotly.graph_objects.layout.yaxis.Minor`\n            instance or dict with compatible properties\n        mirror\n            Determines if the axis lines or/and ticks are mirrored\n            to the opposite side of the plotting area. If True, the\n            axis lines are mirrored. If "ticks", the axis lines and\n            ticks are mirrored. If False, mirroring is disable. If\n            "all", axis lines are mirrored on all shared-axes\n            subplots. If "allticks", axis lines and ticks are\n            mirrored on all shared-axes subplots.\n        nticks\n            Specifies the maximum number of ticks for the\n            particular axis. The actual number of ticks will be\n            chosen automatically to be less than or equal to\n            `nticks`. Has an effect only if `tickmode` is set to\n            "auto".\n        overlaying\n            If set a same-letter axis id, this axis is overlaid on\n            top of the corresponding same-letter axis, with traces\n            and axes visible for both axes. If False, this axis\n            does not overlay any same-letter axes. In this case,\n            for axes with overlapping domains only the highest-\n            numbered axis will be visible.\n        position\n            Sets the position of this axis in the plotting space\n            (in normalized coordinates). Only has an effect if\n            `anchor` is set to "free".\n        range\n            Sets the range of this axis. If the axis `type` is\n            "log", then you must take the log of your desired range\n            (e.g. to set the range from 1 to 100, set the range\n            from 0 to 2). If the axis `type` is "date", it should\n            be date strings, like date data, though Date objects\n            and unix milliseconds will be accepted and converted to\n            strings. If the axis `type` is "category", it should be\n            numbers, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears. Leaving either or both elements `null` impacts\n            the default `autorange`.\n        rangebreaks\n            A tuple of\n            :class:`plotly.graph_objects.layout.yaxis.Rangebreak`\n            instances or dicts with compatible properties\n        rangebreakdefaults\n            When used in a template (as\n            layout.template.layout.yaxis.rangebreakdefaults), sets\n            the default property values to use for elements of\n            layout.yaxis.rangebreaks\n        rangemode\n            If "normal", the range is computed in relation to the\n            extrema of the input data. If *tozero*`, the range\n            extends to 0, regardless of the input data If\n            "nonnegative", the range is non-negative, regardless of\n            the input data. Applies only to linear axes.\n        scaleanchor\n            If set to another axis id (e.g. `x2`, `y`), the range\n            of this axis changes together with the range of the\n            corresponding axis such that the scale of pixels per\n            unit is in a constant ratio. Both axes are still\n            zoomable, but when you zoom one, the other will zoom\n            the same amount, keeping a fixed midpoint. `constrain`\n            and `constraintoward` determine how we enforce the\n            constraint. You can chain these, ie `yaxis:\n            {scaleanchor: *x*}, xaxis2: {scaleanchor: *y*}` but you\n            can only link axes of the same `type`. The linked axis\n            can have the opposite letter (to constrain the aspect\n            ratio) or the same letter (to match scales across\n            subplots). Loops (`yaxis: {scaleanchor: *x*}, xaxis:\n            {scaleanchor: *y*}` or longer) are redundant and the\n            last constraint encountered will be ignored to avoid\n            possible inconsistent constraints via `scaleratio`.\n            Note that setting axes simultaneously in both a\n            `scaleanchor` and a `matches` constraint is currently\n            forbidden. Setting `false` allows to remove a default\n            constraint (occasionally, you may need to prevent a\n            default `scaleanchor` constraint from being applied,\n            eg. when having an image trace `yaxis: {scaleanchor:\n            "x"}` is set automatically in order for pixels to be\n            rendered as squares, setting `yaxis: {scaleanchor:\n            false}` allows to remove the constraint).\n        scaleratio\n            If this axis is linked to another by `scaleanchor`,\n            this determines the pixel to unit scale ratio. For\n            example, if this value is 10, then every unit on this\n            axis spans 10 times the number of pixels as a unit on\n            the linked axis. Use this for example to create an\n            elevation profile where the vertical scale is\n            exaggerated a fixed amount with respect to the\n            horizontal.\n        separatethousands\n            If "true", even 4-digit integers are separated\n        shift\n            Moves the axis a given number of pixels from where it\n            would have been otherwise. Accepts both positive and\n            negative values, which will shift the axis either right\n            or left, respectively. If `autoshift` is set to true,\n            then this defaults to a padding of -3 if `side` is set\n            to "left". and defaults to +3 if `side` is set to\n            "right". Defaults to 0 if `autoshift` is set to false.\n            Only has an effect if `anchor` is set to "free".\n        showdividers\n            Determines whether or not a dividers are drawn between\n            the category levels of this axis. Only has an effect on\n            "multicategory" axes.\n        showexponent\n            If "all", all exponents are shown besides their\n            significands. If "first", only the exponent of the\n            first tick is shown. If "last", only the exponent of\n            the last tick is shown. If "none", no exponents appear.\n        showgrid\n            Determines whether or not grid lines are drawn. If\n            True, the grid lines are drawn at every tick mark.\n        showline\n            Determines whether or not a line bounding this axis is\n            drawn.\n        showspikes\n            Determines whether or not spikes (aka droplines) are\n            drawn for this axis. Note: This only takes affect when\n            hovermode = closest\n        showticklabels\n            Determines whether or not the tick labels are drawn.\n        showtickprefix\n            If "all", all tick labels are displayed with a prefix.\n            If "first", only the first tick is displayed with a\n            prefix. If "last", only the last tick is displayed with\n            a suffix. If "none", tick prefixes are hidden.\n        showticksuffix\n            Same as `showtickprefix` but for tick suffixes.\n        side\n            Determines whether a x (y) axis is positioned at the\n            "bottom" ("left") or "top" ("right") of the plotting\n            area.\n        spikecolor\n            Sets the spike color. If undefined, will use the series\n            color\n        spikedash\n            Sets the dash style of lines. Set to a dash type string\n            ("solid", "dot", "dash", "longdash", "dashdot", or\n            "longdashdot") or a dash length list in px (eg\n            "5px,10px,2px,2px").\n        spikemode\n            Determines the drawing mode for the spike line If\n            "toaxis", the line is drawn from the data point to the\n            axis the  series is plotted on. If "across", the line\n            is drawn across the entire plot area, and supercedes\n            "toaxis". If "marker", then a marker dot is drawn on\n            the axis the series is plotted on\n        spikesnap\n            Determines whether spikelines are stuck to the cursor\n            or to the closest datapoints.\n        spikethickness\n            Sets the width (in px) of the zero line.\n        tick0\n            Sets the placement of the first tick on this axis. Use\n            with `dtick`. If the axis `type` is "log", then you\n            must take the log of your starting tick (e.g. to set\n            the starting tick to 100, set the `tick0` to 2) except\n            when `dtick`=*L<f>* (see `dtick` for more info). If the\n            axis `type` is "date", it should be a date string, like\n            date data. If the axis `type` is "category", it should\n            be a number, using the scale where each category is\n            assigned a serial number from zero in the order it\n            appears.\n        tickangle\n            Sets the angle of the tick labels with respect to the\n            horizontal. For example, a `tickangle` of -90 draws the\n            tick labels vertically.\n        tickcolor\n            Sets the tick color.\n        tickfont\n            Sets the tick font.\n        tickformat\n            Sets the tick label formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display "09~15~23.46"\n        tickformatstops\n            A tuple of :class:`plotly.graph_objects.layout.yaxis.Ti\n            ckformatstop` instances or dicts with compatible\n            properties\n        tickformatstopdefaults\n            When used in a template (as\n            layout.template.layout.yaxis.tickformatstopdefaults),\n            sets the default property values to use for elements of\n            layout.yaxis.tickformatstops\n        ticklabelmode\n            Determines where tick labels are drawn with respect to\n            their corresponding ticks and grid lines. Only has an\n            effect for axes of `type` "date" When set to "period",\n            tick labels are drawn in the middle of the period\n            between ticks.\n        ticklabeloverflow\n            Determines how we handle tick labels that would\n            overflow either the graph div or the domain of the\n            axis. The default value for inside tick labels is *hide\n            past domain*. Otherwise on "category" and\n            "multicategory" axes the default is "allow". In other\n            cases the default is *hide past div*.\n        ticklabelposition\n            Determines where tick labels are drawn with respect to\n            the axis Please note that top or bottom has no effect\n            on x axes or when `ticklabelmode` is set to "period".\n            Similarly left or right has no effect on y axes or when\n            `ticklabelmode` is set to "period". Has no effect on\n            "multicategory" axes or when `tickson` is set to\n            "boundaries". When used on axes linked by `matches` or\n            `scaleanchor`, no extra padding for inside labels would\n            be added by autorange, so that the scales could match.\n        ticklabelstep\n            Sets the spacing between tick labels as compared to the\n            spacing between ticks. A value of 1 (default) means\n            each tick gets a label. A value of 2 means shows every\n            2nd label. A larger value n means only every nth tick\n            is labeled. `tick0` determines which labels are shown.\n            Not implemented for axes with `type` "log" or\n            "multicategory", or when `tickmode` is "array".\n        ticklen\n            Sets the tick length (in px).\n        tickmode\n            Sets the tick mode for this axis. If "auto", the number\n            of ticks is set via `nticks`. If "linear", the\n            placement of the ticks is determined by a starting\n            position `tick0` and a tick step `dtick` ("linear" is\n            the default value if `tick0` and `dtick` are provided).\n            If "array", the placement of the ticks is set via\n            `tickvals` and the tick text is `ticktext`. ("array" is\n            the default value if `tickvals` is provided). If\n            "sync", the number of ticks will sync with the\n            overlayed axis set by `overlaying` property.\n        tickprefix\n            Sets a tick label prefix.\n        ticks\n            Determines whether ticks are drawn or not. If "", this\n            axis\' ticks are not drawn. If "outside" ("inside"),\n            this axis\' are drawn outside (inside) the axis lines.\n        tickson\n            Determines where ticks and grid lines are drawn with\n            respect to their corresponding tick labels. Only has an\n            effect for axes of `type` "category" or\n            "multicategory". When set to "boundaries", ticks and\n            grid lines are drawn half a category to the left/bottom\n            of labels.\n        ticksuffix\n            Sets a tick label suffix.\n        ticktext\n            Sets the text displayed at the ticks position via\n            `tickvals`. Only has an effect if `tickmode` is set to\n            "array". Used with `tickvals`.\n        ticktextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `ticktext`.\n        tickvals\n            Sets the values at which ticks on this axis appear.\n            Only has an effect if `tickmode` is set to "array".\n            Used with `ticktext`.\n        tickvalssrc\n            Sets the source reference on Chart Studio Cloud for\n            `tickvals`.\n        tickwidth\n            Sets the tick width (in px).\n        title\n            :class:`plotly.graph_objects.layout.yaxis.Title`\n            instance or dict with compatible properties\n        titlefont\n            Deprecated: Please use layout.yaxis.title.font instead.\n            Sets this axis\' title font. Note that the title\'s font\n            used to be customized by the now deprecated `titlefont`\n            attribute.\n        type\n            Sets the axis type. By default, plotly attempts to\n            determined the axis type by looking into the data of\n            the traces that referenced the axis in question.\n        uirevision\n            Controls persistence of user-driven changes in axis\n            `range`, `autorange`, and `title` if in `editable:\n            true` configuration. Defaults to `layout.uirevision`.\n        visible\n            A single toggle to hide the axis while preserving\n            interaction like dragging. Default is true when a\n            cheater plot is present on the axis, otherwise false\n        zeroline\n            Determines whether or not a line is drawn at along the\n            0 value of this axis. If True, the zero line is drawn\n            on top of the grid lines.\n        zerolinecolor\n            Sets the line color of the zero line.\n        zerolinewidth\n            Sets the width (in px) of the zero line.\n\n        Returns\n        -------\n        YAxis\n        '
        super(YAxis, self).__init__('yaxis')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.YAxis\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.YAxis`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('anchor', None)
        _v = anchor if anchor is not None else _v
        if _v is not None:
            self['anchor'] = _v
        _v = arg.pop('automargin', None)
        _v = automargin if automargin is not None else _v
        if _v is not None:
            self['automargin'] = _v
        _v = arg.pop('autorange', None)
        _v = autorange if autorange is not None else _v
        if _v is not None:
            self['autorange'] = _v
        _v = arg.pop('autorangeoptions', None)
        _v = autorangeoptions if autorangeoptions is not None else _v
        if _v is not None:
            self['autorangeoptions'] = _v
        _v = arg.pop('autoshift', None)
        _v = autoshift if autoshift is not None else _v
        if _v is not None:
            self['autoshift'] = _v
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
        _v = arg.pop('constrain', None)
        _v = constrain if constrain is not None else _v
        if _v is not None:
            self['constrain'] = _v
        _v = arg.pop('constraintoward', None)
        _v = constraintoward if constraintoward is not None else _v
        if _v is not None:
            self['constraintoward'] = _v
        _v = arg.pop('dividercolor', None)
        _v = dividercolor if dividercolor is not None else _v
        if _v is not None:
            self['dividercolor'] = _v
        _v = arg.pop('dividerwidth', None)
        _v = dividerwidth if dividerwidth is not None else _v
        if _v is not None:
            self['dividerwidth'] = _v
        _v = arg.pop('domain', None)
        _v = domain if domain is not None else _v
        if _v is not None:
            self['domain'] = _v
        _v = arg.pop('dtick', None)
        _v = dtick if dtick is not None else _v
        if _v is not None:
            self['dtick'] = _v
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
        _v = arg.pop('hoverformat', None)
        _v = hoverformat if hoverformat is not None else _v
        if _v is not None:
            self['hoverformat'] = _v
        _v = arg.pop('insiderange', None)
        _v = insiderange if insiderange is not None else _v
        if _v is not None:
            self['insiderange'] = _v
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
        _v = arg.pop('matches', None)
        _v = matches if matches is not None else _v
        if _v is not None:
            self['matches'] = _v
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
        _v = arg.pop('minor', None)
        _v = minor if minor is not None else _v
        if _v is not None:
            self['minor'] = _v
        _v = arg.pop('mirror', None)
        _v = mirror if mirror is not None else _v
        if _v is not None:
            self['mirror'] = _v
        _v = arg.pop('nticks', None)
        _v = nticks if nticks is not None else _v
        if _v is not None:
            self['nticks'] = _v
        _v = arg.pop('overlaying', None)
        _v = overlaying if overlaying is not None else _v
        if _v is not None:
            self['overlaying'] = _v
        _v = arg.pop('position', None)
        _v = position if position is not None else _v
        if _v is not None:
            self['position'] = _v
        _v = arg.pop('range', None)
        _v = range if range is not None else _v
        if _v is not None:
            self['range'] = _v
        _v = arg.pop('rangebreaks', None)
        _v = rangebreaks if rangebreaks is not None else _v
        if _v is not None:
            self['rangebreaks'] = _v
        _v = arg.pop('rangebreakdefaults', None)
        _v = rangebreakdefaults if rangebreakdefaults is not None else _v
        if _v is not None:
            self['rangebreakdefaults'] = _v
        _v = arg.pop('rangemode', None)
        _v = rangemode if rangemode is not None else _v
        if _v is not None:
            self['rangemode'] = _v
        _v = arg.pop('scaleanchor', None)
        _v = scaleanchor if scaleanchor is not None else _v
        if _v is not None:
            self['scaleanchor'] = _v
        _v = arg.pop('scaleratio', None)
        _v = scaleratio if scaleratio is not None else _v
        if _v is not None:
            self['scaleratio'] = _v
        _v = arg.pop('separatethousands', None)
        _v = separatethousands if separatethousands is not None else _v
        if _v is not None:
            self['separatethousands'] = _v
        _v = arg.pop('shift', None)
        _v = shift if shift is not None else _v
        if _v is not None:
            self['shift'] = _v
        _v = arg.pop('showdividers', None)
        _v = showdividers if showdividers is not None else _v
        if _v is not None:
            self['showdividers'] = _v
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
        _v = arg.pop('showspikes', None)
        _v = showspikes if showspikes is not None else _v
        if _v is not None:
            self['showspikes'] = _v
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
        _v = arg.pop('spikecolor', None)
        _v = spikecolor if spikecolor is not None else _v
        if _v is not None:
            self['spikecolor'] = _v
        _v = arg.pop('spikedash', None)
        _v = spikedash if spikedash is not None else _v
        if _v is not None:
            self['spikedash'] = _v
        _v = arg.pop('spikemode', None)
        _v = spikemode if spikemode is not None else _v
        if _v is not None:
            self['spikemode'] = _v
        _v = arg.pop('spikesnap', None)
        _v = spikesnap if spikesnap is not None else _v
        if _v is not None:
            self['spikesnap'] = _v
        _v = arg.pop('spikethickness', None)
        _v = spikethickness if spikethickness is not None else _v
        if _v is not None:
            self['spikethickness'] = _v
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
        _v = arg.pop('ticklabelmode', None)
        _v = ticklabelmode if ticklabelmode is not None else _v
        if _v is not None:
            self['ticklabelmode'] = _v
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
        _v = arg.pop('tickson', None)
        _v = tickson if tickson is not None else _v
        if _v is not None:
            self['tickson'] = _v
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
        _v = arg.pop('zeroline', None)
        _v = zeroline if zeroline is not None else _v
        if _v is not None:
            self['zeroline'] = _v
        _v = arg.pop('zerolinecolor', None)
        _v = zerolinecolor if zerolinecolor is not None else _v
        if _v is not None:
            self['zerolinecolor'] = _v
        _v = arg.pop('zerolinewidth', None)
        _v = zerolinewidth if zerolinewidth is not None else _v
        if _v is not None:
            self['zerolinewidth'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy

class Scatterternary(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'scatterternary'
    _valid_props = {'a', 'asrc', 'b', 'bsrc', 'c', 'cliponaxis', 'connectgaps', 'csrc', 'customdata', 'customdatasrc', 'fill', 'fillcolor', 'hoverinfo', 'hoverinfosrc', 'hoverlabel', 'hoveron', 'hovertemplate', 'hovertemplatesrc', 'hovertext', 'hovertextsrc', 'ids', 'idssrc', 'legend', 'legendgroup', 'legendgrouptitle', 'legendrank', 'legendwidth', 'line', 'marker', 'meta', 'metasrc', 'mode', 'name', 'opacity', 'selected', 'selectedpoints', 'showlegend', 'stream', 'subplot', 'sum', 'text', 'textfont', 'textposition', 'textpositionsrc', 'textsrc', 'texttemplate', 'texttemplatesrc', 'type', 'uid', 'uirevision', 'unselected', 'visible'}

    @property
    def a(self):
        if False:
            return 10
        "\n        Sets the quantity of component `a` in each data point. If `a`,\n        `b`, and `c` are all provided, they need not be normalized,\n        only the relative values matter. If only two arrays are\n        provided they must be normalized to match `ternary<i>.sum`.\n\n        The 'a' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['a']

    @a.setter
    def a(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['a'] = val

    @property
    def asrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `a`.\n\n        The 'asrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['asrc']

    @asrc.setter
    def asrc(self, val):
        if False:
            while True:
                i = 10
        self['asrc'] = val

    @property
    def b(self):
        if False:
            while True:
                i = 10
        "\n        Sets the quantity of component `a` in each data point. If `a`,\n        `b`, and `c` are all provided, they need not be normalized,\n        only the relative values matter. If only two arrays are\n        provided they must be normalized to match `ternary<i>.sum`.\n\n        The 'b' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['b']

    @b.setter
    def b(self, val):
        if False:
            print('Hello World!')
        self['b'] = val

    @property
    def bsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `b`.\n\n        The 'bsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['bsrc']

    @bsrc.setter
    def bsrc(self, val):
        if False:
            return 10
        self['bsrc'] = val

    @property
    def c(self):
        if False:
            while True:
                i = 10
        "\n        Sets the quantity of component `a` in each data point. If `a`,\n        `b`, and `c` are all provided, they need not be normalized,\n        only the relative values matter. If only two arrays are\n        provided they must be normalized to match `ternary<i>.sum`.\n\n        The 'c' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['c']

    @c.setter
    def c(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['c'] = val

    @property
    def cliponaxis(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether or not markers and text nodes are clipped\n        about the subplot axes. To show markers and text nodes above\n        axis lines and tick labels, make sure to set `xaxis.layer` and\n        `yaxis.layer` to *below traces*.\n\n        The 'cliponaxis' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['cliponaxis']

    @cliponaxis.setter
    def cliponaxis(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['cliponaxis'] = val

    @property
    def connectgaps(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not gaps (i.e. {nan} or missing values)\n        in the provided data arrays are connected.\n\n        The 'connectgaps' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['connectgaps']

    @connectgaps.setter
    def connectgaps(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['connectgaps'] = val

    @property
    def csrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `c`.\n\n        The 'csrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['csrc']

    @csrc.setter
    def csrc(self, val):
        if False:
            i = 10
            return i + 15
        self['csrc'] = val

    @property
    def customdata(self):
        if False:
            return 10
        '\n        Assigns extra data each datum. This may be useful when\n        listening to hover, click and selection events. Note that,\n        "scatter" traces also appends customdata items in the markers\n        DOM elements\n\n        The \'customdata\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['customdata']

    @customdata.setter
    def customdata(self, val):
        if False:
            return 10
        self['customdata'] = val

    @property
    def customdatasrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `customdata`.\n\n        The 'customdatasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['customdatasrc']

    @customdatasrc.setter
    def customdatasrc(self, val):
        if False:
            print('Hello World!')
        self['customdatasrc'] = val

    @property
    def fill(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the area to fill with a solid color. Use with `fillcolor`\n        if not "none". scatterternary has a subset of the options\n        available to scatter. "toself" connects the endpoints of the\n        trace (or each segment of the trace if it has gaps) into a\n        closed shape. "tonext" fills the space between two traces if\n        one completely encloses the other (eg consecutive contour\n        lines), and behaves like "toself" if there is no trace before\n        it. "tonext" should not be used if one trace does not enclose\n        the other.\n\n        The \'fill\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'none\', \'toself\', \'tonext\']\n\n        Returns\n        -------\n        Any\n        '
        return self['fill']

    @fill.setter
    def fill(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['fill'] = val

    @property
    def fillcolor(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the fill color. Defaults to a half-transparent variant of\n        the line color, marker color, or marker line color, whichever\n        is available.\n\n        The 'fillcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['fillcolor']

    @fillcolor.setter
    def fillcolor(self, val):
        if False:
            i = 10
            return i + 15
        self['fillcolor'] = val

    @property
    def hoverinfo(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines which trace information appear on hover. If `none`\n        or `skip` are set, no information is displayed upon hovering.\n        But, if `none` is set, click and hover events are still fired.\n\n        The 'hoverinfo' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['a', 'b', 'c', 'text', 'name'] joined with '+' characters\n            (e.g. 'a+b')\n            OR exactly one of ['all', 'none', 'skip'] (e.g. 'skip')\n          - A list or array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['hoverinfo']

    @hoverinfo.setter
    def hoverinfo(self, val):
        if False:
            while True:
                i = 10
        self['hoverinfo'] = val

    @property
    def hoverinfosrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `hoverinfo`.\n\n        The 'hoverinfosrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['hoverinfosrc']

    @hoverinfosrc.setter
    def hoverinfosrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['hoverinfosrc'] = val

    @property
    def hoverlabel(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'hoverlabel' property is an instance of Hoverlabel\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterternary.Hoverlabel`\n          - A dict of string/value properties that will be passed\n            to the Hoverlabel constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the text\n                    content within hover label box. Has an effect\n                    only if the hover label text spans more two or\n                    more lines\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                bgcolor\n                    Sets the background color of the hover labels\n                    for this trace\n                bgcolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bgcolor`.\n                bordercolor\n                    Sets the border color of the hover labels for\n                    this trace.\n                bordercolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bordercolor`.\n                font\n                    Sets the font used in hover labels.\n                namelength\n                    Sets the default length (in number of\n                    characters) of the trace name in the hover\n                    labels for all traces. -1 shows the whole name\n                    regardless of length. 0-3 shows the first 0-3\n                    characters, and an integer >3 will show the\n                    whole name if it is less than that many\n                    characters, but if it is longer, will truncate\n                    to `namelength - 3` characters and add an\n                    ellipsis.\n                namelengthsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `namelength`.\n\n        Returns\n        -------\n        plotly.graph_objs.scatterternary.Hoverlabel\n        "
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['hoverlabel'] = val

    @property
    def hoveron(self):
        if False:
            print('Hello World!')
        '\n        Do the hover effects highlight individual points (markers or\n        line points) or do they highlight filled regions? If the fill\n        is "toself" or "tonext" and there are no markers or text, then\n        the default is "fills", otherwise it is "points".\n\n        The \'hoveron\' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of [\'points\', \'fills\'] joined with \'+\' characters\n            (e.g. \'points+fills\')\n\n        Returns\n        -------\n        Any\n        '
        return self['hoveron']

    @hoveron.setter
    def hoveron(self, val):
        if False:
            print('Hello World!')
        self['hoveron'] = val

    @property
    def hovertemplate(self):
        if False:
            print('Hello World!')
        '\n        Template string used for rendering the information that appear\n        on hover box. Note that this will override `hoverinfo`.\n        Variables are inserted using %{variable}, for example "y: %{y}"\n        as well as %{xother}, {%_xother}, {%_xother_}, {%xother_}. When\n        showing info for several points, "xother" will be added to\n        those with different x positions from the first point. An\n        underscore before or after "(x|y)other" will add a space on\n        that side, only when this field is shown. Numbers are formatted\n        using d3-format\'s syntax %{variable:d3-format}, for example\n        "Price: %{y:$.2f}".\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for\n        details on the formatting syntax. Dates are formatted using\n        d3-time-format\'s syntax %{variable|d3-time-format}, for example\n        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format for details on the date\n        formatting syntax. The variables available in `hovertemplate`\n        are the ones emitted as event data described at this link\n        https://plotly.com/javascript/plotlyjs-events/#event-data.\n        Additionally, every attributes that can be specified per-point\n        (the ones that are `arrayOk: true`) are available.  Anything\n        contained in tag `<extra>` is displayed in the secondary box,\n        for example "<extra>{fullData.name}</extra>". To hide the\n        secondary box completely, use an empty tag `<extra></extra>`.\n\n        The \'hovertemplate\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['hovertemplate']

    @hovertemplate.setter
    def hovertemplate(self, val):
        if False:
            i = 10
            return i + 15
        self['hovertemplate'] = val

    @property
    def hovertemplatesrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for\n        `hovertemplate`.\n\n        The 'hovertemplatesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['hovertemplatesrc']

    @hovertemplatesrc.setter
    def hovertemplatesrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['hovertemplatesrc'] = val

    @property
    def hovertext(self):
        if False:
            print('Hello World!')
        '\n        Sets hover text elements associated with each (a,b,c) point. If\n        a single string, the same string appears over all the data\n        points. If an array of strings, the items are mapped in order\n        to the the data points in (a,b,c). To be seen, trace\n        `hoverinfo` must contain a "text" flag.\n\n        The \'hovertext\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['hovertext']

    @hovertext.setter
    def hovertext(self, val):
        if False:
            print('Hello World!')
        self['hovertext'] = val

    @property
    def hovertextsrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `hovertext`.\n\n        The 'hovertextsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['hovertextsrc']

    @hovertextsrc.setter
    def hovertextsrc(self, val):
        if False:
            i = 10
            return i + 15
        self['hovertextsrc'] = val

    @property
    def ids(self):
        if False:
            print('Hello World!')
        "\n        Assigns id labels to each datum. These ids for object constancy\n        of data points during animation. Should be an array of strings,\n        not numbers or any other type.\n\n        The 'ids' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['ids']

    @ids.setter
    def ids(self, val):
        if False:
            print('Hello World!')
        self['ids'] = val

    @property
    def idssrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `ids`.\n\n        The 'idssrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['idssrc']

    @idssrc.setter
    def idssrc(self, val):
        if False:
            i = 10
            return i + 15
        self['idssrc'] = val

    @property
    def legend(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the reference to a legend to show this trace in.\n        References to these legends are "legend", "legend2", "legend3",\n        etc. Settings for these legends are set in the layout, under\n        `layout.legend`, `layout.legend2`, etc.\n\n        The \'legend\' property is an identifier of a particular\n        subplot, of type \'legend\', that may be specified as the string \'legend\'\n        optionally followed by an integer >= 1\n        (e.g. \'legend\', \'legend1\', \'legend2\', \'legend3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['legend']

    @legend.setter
    def legend(self, val):
        if False:
            i = 10
            return i + 15
        self['legend'] = val

    @property
    def legendgroup(self):
        if False:
            print('Hello World!')
        "\n        Sets the legend group for this trace. Traces and shapes part of\n        the same legend group hide/show at the same time when toggling\n        legend items.\n\n        The 'legendgroup' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['legendgroup']

    @legendgroup.setter
    def legendgroup(self, val):
        if False:
            return 10
        self['legendgroup'] = val

    @property
    def legendgrouptitle(self):
        if False:
            return 10
        "\n        The 'legendgrouptitle' property is an instance of Legendgrouptitle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterternary.Legendgrouptitle`\n          - A dict of string/value properties that will be passed\n            to the Legendgrouptitle constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend group's title font.\n                text\n                    Sets the title of the legend group.\n\n        Returns\n        -------\n        plotly.graph_objs.scatterternary.Legendgrouptitle\n        "
        return self['legendgrouptitle']

    @legendgrouptitle.setter
    def legendgrouptitle(self, val):
        if False:
            i = 10
            return i + 15
        self['legendgrouptitle'] = val

    @property
    def legendrank(self):
        if False:
            return 10
        '\n        Sets the legend rank for this trace. Items and groups with\n        smaller ranks are presented on top/left side while with\n        "reversed" `legend.traceorder` they are on bottom/right side.\n        The default legendrank is 1000, so that you can use ranks less\n        than 1000 to place certain items before all unranked items, and\n        ranks greater than 1000 to go after all unranked items. When\n        having unranked or equal rank items shapes would be displayed\n        after traces i.e. according to their order in data and layout.\n\n        The \'legendrank\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['legendrank']

    @legendrank.setter
    def legendrank(self, val):
        if False:
            while True:
                i = 10
        self['legendrank'] = val

    @property
    def legendwidth(self):
        if False:
            while True:
                i = 10
        "\n        Sets the width (in px or fraction) of the legend for this\n        trace.\n\n        The 'legendwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['legendwidth']

    @legendwidth.setter
    def legendwidth(self, val):
        if False:
            print('Hello World!')
        self['legendwidth'] = val

    @property
    def line(self):
        if False:
            return 10
        '\n        The \'line\' property is an instance of Line\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterternary.Line`\n          - A dict of string/value properties that will be passed\n            to the Line constructor\n\n            Supported dict properties:\n\n                backoff\n                    Sets the line back off from the end point of\n                    the nth line segment (in px). This option is\n                    useful e.g. to avoid overlap with arrowhead\n                    markers. With "auto" the lines would trim\n                    before markers if `marker.angleref` is set to\n                    "previous".\n                backoffsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `backoff`.\n                color\n                    Sets the line color.\n                dash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                shape\n                    Determines the line shape. With "spline" the\n                    lines are drawn using spline interpolation. The\n                    other available values correspond to step-wise\n                    line shapes.\n                smoothing\n                    Has an effect only if `shape` is set to\n                    "spline" Sets the amount of smoothing. 0\n                    corresponds to no smoothing (equivalent to a\n                    "linear" shape).\n                width\n                    Sets the line width (in px).\n\n        Returns\n        -------\n        plotly.graph_objs.scatterternary.Line\n        '
        return self['line']

    @line.setter
    def line(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['line'] = val

    @property
    def marker(self):
        if False:
            print('Hello World!')
        '\n        The \'marker\' property is an instance of Marker\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterternary.Marker`\n          - A dict of string/value properties that will be passed\n            to the Marker constructor\n\n            Supported dict properties:\n\n                angle\n                    Sets the marker angle in respect to `angleref`.\n                angleref\n                    Sets the reference for marker angle. With\n                    "previous", angle 0 points along the line from\n                    the previous point to this one. With "up",\n                    angle 0 points toward the top of the screen.\n                anglesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `angle`.\n                autocolorscale\n                    Determines whether the colorscale is a default\n                    palette (`autocolorscale: true`) or the palette\n                    determined by `marker.colorscale`. Has an\n                    effect only if in `marker.color` is set to a\n                    numerical array. In case `colorscale` is\n                    unspecified or `autocolorscale` is true, the\n                    default palette will be chosen according to\n                    whether numbers in the `color` array are all\n                    positive, all negative or mixed.\n                cauto\n                    Determines whether or not the color domain is\n                    computed with respect to the input data (here\n                    in `marker.color`) or the bounds set in\n                    `marker.cmin` and `marker.cmax` Has an effect\n                    only if in `marker.color` is set to a numerical\n                    array. Defaults to `false` when `marker.cmin`\n                    and `marker.cmax` are set by the user.\n                cmax\n                    Sets the upper bound of the color domain. Has\n                    an effect only if in `marker.color` is set to a\n                    numerical array. Value should have the same\n                    units as in `marker.color` and if set,\n                    `marker.cmin` must be set as well.\n                cmid\n                    Sets the mid-point of the color domain by\n                    scaling `marker.cmin` and/or `marker.cmax` to\n                    be equidistant to this point. Has an effect\n                    only if in `marker.color` is set to a numerical\n                    array. Value should have the same units as in\n                    `marker.color`. Has no effect when\n                    `marker.cauto` is `false`.\n                cmin\n                    Sets the lower bound of the color domain. Has\n                    an effect only if in `marker.color` is set to a\n                    numerical array. Value should have the same\n                    units as in `marker.color` and if set,\n                    `marker.cmax` must be set as well.\n                color\n                    Sets the marker color. It accepts either a\n                    specific color or an array of numbers that are\n                    mapped to the colorscale relative to the max\n                    and min values of the array or relative to\n                    `marker.cmin` and `marker.cmax` if set.\n                coloraxis\n                    Sets a reference to a shared color axis.\n                    References to these shared color axes are\n                    "coloraxis", "coloraxis2", "coloraxis3", etc.\n                    Settings for these shared color axes are set in\n                    the layout, under `layout.coloraxis`,\n                    `layout.coloraxis2`, etc. Note that multiple\n                    color scales can be linked to the same color\n                    axis.\n                colorbar\n                    :class:`plotly.graph_objects.scatterternary.mar\n                    ker.ColorBar` instance or dict with compatible\n                    properties\n                colorscale\n                    Sets the colorscale. Has an effect only if in\n                    `marker.color` is set to a numerical array. The\n                    colorscale must be an array containing arrays\n                    mapping a normalized value to an rgb, rgba,\n                    hex, hsl, hsv, or named color string. At\n                    minimum, a mapping for the lowest (0) and\n                    highest (1) values are required. For example,\n                    `[[0, \'rgb(0,0,255)\'], [1, \'rgb(255,0,0)\']]`.\n                    To control the bounds of the colorscale in\n                    color space, use `marker.cmin` and\n                    `marker.cmax`. Alternatively, `colorscale` may\n                    be a palette name string of the following list:\n                    Blackbody,Bluered,Blues,Cividis,Earth,Electric,\n                    Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,Rd\n                    Bu,Reds,Viridis,YlGnBu,YlOrRd.\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                gradient\n                    :class:`plotly.graph_objects.scatterternary.mar\n                    ker.Gradient` instance or dict with compatible\n                    properties\n                line\n                    :class:`plotly.graph_objects.scatterternary.mar\n                    ker.Line` instance or dict with compatible\n                    properties\n                maxdisplayed\n                    Sets a maximum number of points to be drawn on\n                    the graph. 0 corresponds to no limit.\n                opacity\n                    Sets the marker opacity.\n                opacitysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `opacity`.\n                reversescale\n                    Reverses the color mapping if true. Has an\n                    effect only if in `marker.color` is set to a\n                    numerical array. If true, `marker.cmin` will\n                    correspond to the last color in the array and\n                    `marker.cmax` will correspond to the first\n                    color.\n                showscale\n                    Determines whether or not a colorbar is\n                    displayed for this trace. Has an effect only if\n                    in `marker.color` is set to a numerical array.\n                size\n                    Sets the marker size (in px).\n                sizemin\n                    Has an effect only if `marker.size` is set to a\n                    numerical array. Sets the minimum size (in px)\n                    of the rendered marker points.\n                sizemode\n                    Has an effect only if `marker.size` is set to a\n                    numerical array. Sets the rule for which the\n                    data in `size` is converted to pixels.\n                sizeref\n                    Has an effect only if `marker.size` is set to a\n                    numerical array. Sets the scale factor used to\n                    determine the rendered size of marker points.\n                    Use with `sizemin` and `sizemode`.\n                sizesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `size`.\n                standoff\n                    Moves the marker away from the data point in\n                    the direction of `angle` (in px). This can be\n                    useful for example if you have another marker\n                    at this location and you want to point an\n                    arrowhead marker at it.\n                standoffsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `standoff`.\n                symbol\n                    Sets the marker symbol type. Adding 100 is\n                    equivalent to appending "-open" to a symbol\n                    name. Adding 200 is equivalent to appending\n                    "-dot" to a symbol name. Adding 300 is\n                    equivalent to appending "-open-dot" or "dot-\n                    open" to a symbol name.\n                symbolsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `symbol`.\n\n        Returns\n        -------\n        plotly.graph_objs.scatterternary.Marker\n        '
        return self['marker']

    @marker.setter
    def marker(self, val):
        if False:
            return 10
        self['marker'] = val

    @property
    def meta(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Assigns extra meta information associated with this trace that\n        can be used in various text attributes. Attributes such as\n        trace `name`, graph, axis and colorbar `title.text`, annotation\n        `text` `rangeselector`, `updatemenues` and `sliders` `label`\n        text all support `meta`. To access the trace `meta` values in\n        an attribute in the same trace, simply use `%{meta[i]}` where\n        `i` is the index or key of the `meta` item in question. To\n        access trace `meta` in layout attributes, use\n        `%{data[n[.meta[i]}` where `i` is the index or key of the\n        `meta` and `n` is the trace index.\n\n        The 'meta' property accepts values of any type\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['meta']

    @meta.setter
    def meta(self, val):
        if False:
            while True:
                i = 10
        self['meta'] = val

    @property
    def metasrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `meta`.\n\n        The 'metasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['metasrc']

    @metasrc.setter
    def metasrc(self, val):
        if False:
            while True:
                i = 10
        self['metasrc'] = val

    @property
    def mode(self):
        if False:
            return 10
        '\n        Determines the drawing mode for this scatter trace. If the\n        provided `mode` includes "text" then the `text` elements appear\n        at the coordinates. Otherwise, the `text` elements appear on\n        hover. If there are less than 20 points and the trace is not\n        stacked then the default is "lines+markers". Otherwise,\n        "lines".\n\n        The \'mode\' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of [\'lines\', \'markers\', \'text\'] joined with \'+\' characters\n            (e.g. \'lines+markers\')\n            OR exactly one of [\'none\'] (e.g. \'none\')\n\n        Returns\n        -------\n        Any\n        '
        return self['mode']

    @mode.setter
    def mode(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['mode'] = val

    @property
    def name(self):
        if False:
            print('Hello World!')
        "\n        Sets the trace name. The trace name appears as the legend item\n        and on hover.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            while True:
                i = 10
        self['name'] = val

    @property
    def opacity(self):
        if False:
            return 10
        "\n        Sets the opacity of the trace.\n\n        The 'opacity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            print('Hello World!')
        self['opacity'] = val

    @property
    def selected(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'selected' property is an instance of Selected\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterternary.Selected`\n          - A dict of string/value properties that will be passed\n            to the Selected constructor\n\n            Supported dict properties:\n\n                marker\n                    :class:`plotly.graph_objects.scatterternary.sel\n                    ected.Marker` instance or dict with compatible\n                    properties\n                textfont\n                    :class:`plotly.graph_objects.scatterternary.sel\n                    ected.Textfont` instance or dict with\n                    compatible properties\n\n        Returns\n        -------\n        plotly.graph_objs.scatterternary.Selected\n        "
        return self['selected']

    @selected.setter
    def selected(self, val):
        if False:
            i = 10
            return i + 15
        self['selected'] = val

    @property
    def selectedpoints(self):
        if False:
            i = 10
            return i + 15
        "\n        Array containing integer indices of selected points. Has an\n        effect only for traces that support selections. Note that an\n        empty array means an empty selection where the `unselected` are\n        turned on for all points, whereas, any other non-array values\n        means no selection all where the `selected` and `unselected`\n        styles have no effect.\n\n        The 'selectedpoints' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['selectedpoints']

    @selectedpoints.setter
    def selectedpoints(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['selectedpoints'] = val

    @property
    def showlegend(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether or not an item corresponding to this trace\n        is shown in the legend.\n\n        The 'showlegend' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showlegend']

    @showlegend.setter
    def showlegend(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showlegend'] = val

    @property
    def stream(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'stream' property is an instance of Stream\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterternary.Stream`\n          - A dict of string/value properties that will be passed\n            to the Stream constructor\n\n            Supported dict properties:\n\n                maxpoints\n                    Sets the maximum number of points to keep on\n                    the plots from an incoming stream. If\n                    `maxpoints` is set to 50, only the newest 50\n                    points will be displayed on the plot.\n                token\n                    The stream id number links a data trace on a\n                    plot with a stream. See https://chart-\n                    studio.plotly.com/settings for more details.\n\n        Returns\n        -------\n        plotly.graph_objs.scatterternary.Stream\n        "
        return self['stream']

    @stream.setter
    def stream(self, val):
        if False:
            return 10
        self['stream'] = val

    @property
    def subplot(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets a reference between this trace\'s data coordinates and a\n        ternary subplot. If "ternary" (the default value), the data\n        refer to `layout.ternary`. If "ternary2", the data refer to\n        `layout.ternary2`, and so on.\n\n        The \'subplot\' property is an identifier of a particular\n        subplot, of type \'ternary\', that may be specified as the string \'ternary\'\n        optionally followed by an integer >= 1\n        (e.g. \'ternary\', \'ternary1\', \'ternary2\', \'ternary3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['subplot']

    @subplot.setter
    def subplot(self, val):
        if False:
            print('Hello World!')
        self['subplot'] = val

    @property
    def sum(self):
        if False:
            return 10
        "\n        The number each triplet should sum to, if only two of `a`, `b`,\n        and `c` are provided. This overrides `ternary<i>.sum` to\n        normalize this specific trace, but does not affect the values\n        displayed on the axes. 0 (or missing) means to use\n        ternary<i>.sum\n\n        The 'sum' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['sum']

    @sum.setter
    def sum(self, val):
        if False:
            i = 10
            return i + 15
        self['sum'] = val

    @property
    def text(self):
        if False:
            while True:
                i = 10
        '\n        Sets text elements associated with each (a,b,c) point. If a\n        single string, the same string appears over all the data\n        points. If an array of strings, the items are mapped in order\n        to the the data points in (a,b,c). If trace `hoverinfo`\n        contains a "text" flag and "hovertext" is not set, these\n        elements will be seen in the hover labels.\n\n        The \'text\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['text']

    @text.setter
    def text(self, val):
        if False:
            print('Hello World!')
        self['text'] = val

    @property
    def textfont(self):
        if False:
            return 10
        '\n        Sets the text font.\n\n        The \'textfont\' property is an instance of Textfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterternary.Textfont`\n          - A dict of string/value properties that will be passed\n            to the Textfont constructor\n\n            Supported dict properties:\n\n                color\n\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                familysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `family`.\n                size\n\n                sizesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `size`.\n\n        Returns\n        -------\n        plotly.graph_objs.scatterternary.Textfont\n        '
        return self['textfont']

    @textfont.setter
    def textfont(self, val):
        if False:
            return 10
        self['textfont'] = val

    @property
    def textposition(self):
        if False:
            print('Hello World!')
        "\n        Sets the positions of the `text` elements with respects to the\n        (x,y) coordinates.\n\n        The 'textposition' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['top left', 'top center', 'top right', 'middle left',\n                'middle center', 'middle right', 'bottom left', 'bottom\n                center', 'bottom right']\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['textposition']

    @textposition.setter
    def textposition(self, val):
        if False:
            i = 10
            return i + 15
        self['textposition'] = val

    @property
    def textpositionsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for\n        `textposition`.\n\n        The 'textpositionsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['textpositionsrc']

    @textpositionsrc.setter
    def textpositionsrc(self, val):
        if False:
            print('Hello World!')
        self['textpositionsrc'] = val

    @property
    def textsrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `text`.\n\n        The 'textsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['textsrc']

    @textsrc.setter
    def textsrc(self, val):
        if False:
            i = 10
            return i + 15
        self['textsrc'] = val

    @property
    def texttemplate(self):
        if False:
            i = 10
            return i + 15
        '\n        Template string used for rendering the information text that\n        appear on points. Note that this will override `textinfo`.\n        Variables are inserted using %{variable}, for example "y:\n        %{y}". Numbers are formatted using d3-format\'s syntax\n        %{variable:d3-format}, for example "Price: %{y:$.2f}".\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for\n        details on the formatting syntax. Dates are formatted using\n        d3-time-format\'s syntax %{variable|d3-time-format}, for example\n        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format for details on the date\n        formatting syntax. Every attributes that can be specified per-\n        point (the ones that are `arrayOk: true`) are available.\n        Finally, the template string has access to variables `a`, `b`,\n        `c` and `text`.\n\n        The \'texttemplate\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['texttemplate']

    @texttemplate.setter
    def texttemplate(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['texttemplate'] = val

    @property
    def texttemplatesrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for\n        `texttemplate`.\n\n        The 'texttemplatesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['texttemplatesrc']

    @texttemplatesrc.setter
    def texttemplatesrc(self, val):
        if False:
            return 10
        self['texttemplatesrc'] = val

    @property
    def uid(self):
        if False:
            while True:
                i = 10
        "\n        Assign an id to this trace, Use this to provide object\n        constancy between traces during animations and transitions.\n\n        The 'uid' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['uid']

    @uid.setter
    def uid(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['uid'] = val

    @property
    def uirevision(self):
        if False:
            return 10
        "\n        Controls persistence of some user-driven changes to the trace:\n        `constraintrange` in `parcoords` traces, as well as some\n        `editable: true` modifications such as `name` and\n        `colorbar.title`. Defaults to `layout.uirevision`. Note that\n        other user-driven trace attribute changes are controlled by\n        `layout` attributes: `trace.visible` is controlled by\n        `layout.legend.uirevision`, `selectedpoints` is controlled by\n        `layout.selectionrevision`, and `colorbar.(x|y)` (accessible\n        with `config: {editable: true}`) is controlled by\n        `layout.editrevision`. Trace changes are tracked by `uid`,\n        which only falls back on trace index if no `uid` is provided.\n        So if your app can add/remove traces before the end of the\n        `data` array, such that the same trace has a different index,\n        you can still preserve user-driven changes if you give each\n        trace a `uid` that stays with it as it moves.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            i = 10
            return i + 15
        self['uirevision'] = val

    @property
    def unselected(self):
        if False:
            while True:
                i = 10
        "\n        The 'unselected' property is an instance of Unselected\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scatterternary.Unselected`\n          - A dict of string/value properties that will be passed\n            to the Unselected constructor\n\n            Supported dict properties:\n\n                marker\n                    :class:`plotly.graph_objects.scatterternary.uns\n                    elected.Marker` instance or dict with\n                    compatible properties\n                textfont\n                    :class:`plotly.graph_objects.scatterternary.uns\n                    elected.Textfont` instance or dict with\n                    compatible properties\n\n        Returns\n        -------\n        plotly.graph_objs.scatterternary.Unselected\n        "
        return self['unselected']

    @unselected.setter
    def unselected(self, val):
        if False:
            print('Hello World!')
        self['unselected'] = val

    @property
    def visible(self):
        if False:
            print('Hello World!')
        '\n        Determines whether or not this trace is visible. If\n        "legendonly", the trace is not drawn, but can appear as a\n        legend item (provided that the legend itself is visible).\n\n        The \'visible\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, \'legendonly\']\n\n        Returns\n        -------\n        Any\n        '
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            print('Hello World!')
        self['visible'] = val

    @property
    def type(self):
        if False:
            for i in range(10):
                print('nop')
        return self._props['type']

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        a\n            Sets the quantity of component `a` in each data point.\n            If `a`, `b`, and `c` are all provided, they need not be\n            normalized, only the relative values matter. If only\n            two arrays are provided they must be normalized to\n            match `ternary<i>.sum`.\n        asrc\n            Sets the source reference on Chart Studio Cloud for\n            `a`.\n        b\n            Sets the quantity of component `a` in each data point.\n            If `a`, `b`, and `c` are all provided, they need not be\n            normalized, only the relative values matter. If only\n            two arrays are provided they must be normalized to\n            match `ternary<i>.sum`.\n        bsrc\n            Sets the source reference on Chart Studio Cloud for\n            `b`.\n        c\n            Sets the quantity of component `a` in each data point.\n            If `a`, `b`, and `c` are all provided, they need not be\n            normalized, only the relative values matter. If only\n            two arrays are provided they must be normalized to\n            match `ternary<i>.sum`.\n        cliponaxis\n            Determines whether or not markers and text nodes are\n            clipped about the subplot axes. To show markers and\n            text nodes above axis lines and tick labels, make sure\n            to set `xaxis.layer` and `yaxis.layer` to *below\n            traces*.\n        connectgaps\n            Determines whether or not gaps (i.e. {nan} or missing\n            values) in the provided data arrays are connected.\n        csrc\n            Sets the source reference on Chart Studio Cloud for\n            `c`.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        fill\n            Sets the area to fill with a solid color. Use with\n            `fillcolor` if not "none". scatterternary has a subset\n            of the options available to scatter. "toself" connects\n            the endpoints of the trace (or each segment of the\n            trace if it has gaps) into a closed shape. "tonext"\n            fills the space between two traces if one completely\n            encloses the other (eg consecutive contour lines), and\n            behaves like "toself" if there is no trace before it.\n            "tonext" should not be used if one trace does not\n            enclose the other.\n        fillcolor\n            Sets the fill color. Defaults to a half-transparent\n            variant of the line color, marker color, or marker line\n            color, whichever is available.\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.scatterternary.Hoverlabel`\n            instance or dict with compatible properties\n        hoveron\n            Do the hover effects highlight individual points\n            (markers or line points) or do they highlight filled\n            regions? If the fill is "toself" or "tonext" and there\n            are no markers or text, then the default is "fills",\n            otherwise it is "points".\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available.  Anything contained in tag `<extra>` is\n            displayed in the secondary box, for example\n            "<extra>{fullData.name}</extra>". To hide the secondary\n            box completely, use an empty tag `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Sets hover text elements associated with each (a,b,c)\n            point. If a single string, the same string appears over\n            all the data points. If an array of strings, the items\n            are mapped in order to the the data points in (a,b,c).\n            To be seen, trace `hoverinfo` must contain a "text"\n            flag.\n        hovertextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertext`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.scatterternary.Legendgroup\n            title` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        line\n            :class:`plotly.graph_objects.scatterternary.Line`\n            instance or dict with compatible properties\n        marker\n            :class:`plotly.graph_objects.scatterternary.Marker`\n            instance or dict with compatible properties\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        mode\n            Determines the drawing mode for this scatter trace. If\n            the provided `mode` includes "text" then the `text`\n            elements appear at the coordinates. Otherwise, the\n            `text` elements appear on hover. If there are less than\n            20 points and the trace is not stacked then the default\n            is "lines+markers". Otherwise, "lines".\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        opacity\n            Sets the opacity of the trace.\n        selected\n            :class:`plotly.graph_objects.scatterternary.Selected`\n            instance or dict with compatible properties\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        stream\n            :class:`plotly.graph_objects.scatterternary.Stream`\n            instance or dict with compatible properties\n        subplot\n            Sets a reference between this trace\'s data coordinates\n            and a ternary subplot. If "ternary" (the default\n            value), the data refer to `layout.ternary`. If\n            "ternary2", the data refer to `layout.ternary2`, and so\n            on.\n        sum\n            The number each triplet should sum to, if only two of\n            `a`, `b`, and `c` are provided. This overrides\n            `ternary<i>.sum` to normalize this specific trace, but\n            does not affect the values displayed on the axes. 0 (or\n            missing) means to use ternary<i>.sum\n        text\n            Sets text elements associated with each (a,b,c) point.\n            If a single string, the same string appears over all\n            the data points. If an array of strings, the items are\n            mapped in order to the the data points in (a,b,c). If\n            trace `hoverinfo` contains a "text" flag and\n            "hovertext" is not set, these elements will be seen in\n            the hover labels.\n        textfont\n            Sets the text font.\n        textposition\n            Sets the positions of the `text` elements with respects\n            to the (x,y) coordinates.\n        textpositionsrc\n            Sets the source reference on Chart Studio Cloud for\n            `textposition`.\n        textsrc\n            Sets the source reference on Chart Studio Cloud for\n            `text`.\n        texttemplate\n            Template string used for rendering the information text\n            that appear on points. Note that this will override\n            `textinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}". Numbers are formatted using\n            d3-format\'s syntax %{variable:d3-format}, for example\n            "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. Every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `a`, `b`, `c` and `text`.\n        texttemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `texttemplate`.\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        unselected\n            :class:`plotly.graph_objects.scatterternary.Unselected`\n            instance or dict with compatible properties\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        '

    def __init__(self, arg=None, a=None, asrc=None, b=None, bsrc=None, c=None, cliponaxis=None, connectgaps=None, csrc=None, customdata=None, customdatasrc=None, fill=None, fillcolor=None, hoverinfo=None, hoverinfosrc=None, hoverlabel=None, hoveron=None, hovertemplate=None, hovertemplatesrc=None, hovertext=None, hovertextsrc=None, ids=None, idssrc=None, legend=None, legendgroup=None, legendgrouptitle=None, legendrank=None, legendwidth=None, line=None, marker=None, meta=None, metasrc=None, mode=None, name=None, opacity=None, selected=None, selectedpoints=None, showlegend=None, stream=None, subplot=None, sum=None, text=None, textfont=None, textposition=None, textpositionsrc=None, textsrc=None, texttemplate=None, texttemplatesrc=None, uid=None, uirevision=None, unselected=None, visible=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Scatterternary object\n\n        Provides similar functionality to the "scatter" type but on a\n        ternary phase diagram. The data is provided by at least two\n        arrays out of `a`, `b`, `c` triplets.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.Scatterternary`\n        a\n            Sets the quantity of component `a` in each data point.\n            If `a`, `b`, and `c` are all provided, they need not be\n            normalized, only the relative values matter. If only\n            two arrays are provided they must be normalized to\n            match `ternary<i>.sum`.\n        asrc\n            Sets the source reference on Chart Studio Cloud for\n            `a`.\n        b\n            Sets the quantity of component `a` in each data point.\n            If `a`, `b`, and `c` are all provided, they need not be\n            normalized, only the relative values matter. If only\n            two arrays are provided they must be normalized to\n            match `ternary<i>.sum`.\n        bsrc\n            Sets the source reference on Chart Studio Cloud for\n            `b`.\n        c\n            Sets the quantity of component `a` in each data point.\n            If `a`, `b`, and `c` are all provided, they need not be\n            normalized, only the relative values matter. If only\n            two arrays are provided they must be normalized to\n            match `ternary<i>.sum`.\n        cliponaxis\n            Determines whether or not markers and text nodes are\n            clipped about the subplot axes. To show markers and\n            text nodes above axis lines and tick labels, make sure\n            to set `xaxis.layer` and `yaxis.layer` to *below\n            traces*.\n        connectgaps\n            Determines whether or not gaps (i.e. {nan} or missing\n            values) in the provided data arrays are connected.\n        csrc\n            Sets the source reference on Chart Studio Cloud for\n            `c`.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        fill\n            Sets the area to fill with a solid color. Use with\n            `fillcolor` if not "none". scatterternary has a subset\n            of the options available to scatter. "toself" connects\n            the endpoints of the trace (or each segment of the\n            trace if it has gaps) into a closed shape. "tonext"\n            fills the space between two traces if one completely\n            encloses the other (eg consecutive contour lines), and\n            behaves like "toself" if there is no trace before it.\n            "tonext" should not be used if one trace does not\n            enclose the other.\n        fillcolor\n            Sets the fill color. Defaults to a half-transparent\n            variant of the line color, marker color, or marker line\n            color, whichever is available.\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.scatterternary.Hoverlabel`\n            instance or dict with compatible properties\n        hoveron\n            Do the hover effects highlight individual points\n            (markers or line points) or do they highlight filled\n            regions? If the fill is "toself" or "tonext" and there\n            are no markers or text, then the default is "fills",\n            otherwise it is "points".\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available.  Anything contained in tag `<extra>` is\n            displayed in the secondary box, for example\n            "<extra>{fullData.name}</extra>". To hide the secondary\n            box completely, use an empty tag `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Sets hover text elements associated with each (a,b,c)\n            point. If a single string, the same string appears over\n            all the data points. If an array of strings, the items\n            are mapped in order to the the data points in (a,b,c).\n            To be seen, trace `hoverinfo` must contain a "text"\n            flag.\n        hovertextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertext`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.scatterternary.Legendgroup\n            title` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        line\n            :class:`plotly.graph_objects.scatterternary.Line`\n            instance or dict with compatible properties\n        marker\n            :class:`plotly.graph_objects.scatterternary.Marker`\n            instance or dict with compatible properties\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        mode\n            Determines the drawing mode for this scatter trace. If\n            the provided `mode` includes "text" then the `text`\n            elements appear at the coordinates. Otherwise, the\n            `text` elements appear on hover. If there are less than\n            20 points and the trace is not stacked then the default\n            is "lines+markers". Otherwise, "lines".\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        opacity\n            Sets the opacity of the trace.\n        selected\n            :class:`plotly.graph_objects.scatterternary.Selected`\n            instance or dict with compatible properties\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        stream\n            :class:`plotly.graph_objects.scatterternary.Stream`\n            instance or dict with compatible properties\n        subplot\n            Sets a reference between this trace\'s data coordinates\n            and a ternary subplot. If "ternary" (the default\n            value), the data refer to `layout.ternary`. If\n            "ternary2", the data refer to `layout.ternary2`, and so\n            on.\n        sum\n            The number each triplet should sum to, if only two of\n            `a`, `b`, and `c` are provided. This overrides\n            `ternary<i>.sum` to normalize this specific trace, but\n            does not affect the values displayed on the axes. 0 (or\n            missing) means to use ternary<i>.sum\n        text\n            Sets text elements associated with each (a,b,c) point.\n            If a single string, the same string appears over all\n            the data points. If an array of strings, the items are\n            mapped in order to the the data points in (a,b,c). If\n            trace `hoverinfo` contains a "text" flag and\n            "hovertext" is not set, these elements will be seen in\n            the hover labels.\n        textfont\n            Sets the text font.\n        textposition\n            Sets the positions of the `text` elements with respects\n            to the (x,y) coordinates.\n        textpositionsrc\n            Sets the source reference on Chart Studio Cloud for\n            `textposition`.\n        textsrc\n            Sets the source reference on Chart Studio Cloud for\n            `text`.\n        texttemplate\n            Template string used for rendering the information text\n            that appear on points. Note that this will override\n            `textinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}". Numbers are formatted using\n            d3-format\'s syntax %{variable:d3-format}, for example\n            "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. Every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `a`, `b`, `c` and `text`.\n        texttemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `texttemplate`.\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        unselected\n            :class:`plotly.graph_objects.scatterternary.Unselected`\n            instance or dict with compatible properties\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n\n        Returns\n        -------\n        Scatterternary\n        '
        super(Scatterternary, self).__init__('scatterternary')
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
            raise ValueError('The first argument to the plotly.graph_objs.Scatterternary\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Scatterternary`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('a', None)
        _v = a if a is not None else _v
        if _v is not None:
            self['a'] = _v
        _v = arg.pop('asrc', None)
        _v = asrc if asrc is not None else _v
        if _v is not None:
            self['asrc'] = _v
        _v = arg.pop('b', None)
        _v = b if b is not None else _v
        if _v is not None:
            self['b'] = _v
        _v = arg.pop('bsrc', None)
        _v = bsrc if bsrc is not None else _v
        if _v is not None:
            self['bsrc'] = _v
        _v = arg.pop('c', None)
        _v = c if c is not None else _v
        if _v is not None:
            self['c'] = _v
        _v = arg.pop('cliponaxis', None)
        _v = cliponaxis if cliponaxis is not None else _v
        if _v is not None:
            self['cliponaxis'] = _v
        _v = arg.pop('connectgaps', None)
        _v = connectgaps if connectgaps is not None else _v
        if _v is not None:
            self['connectgaps'] = _v
        _v = arg.pop('csrc', None)
        _v = csrc if csrc is not None else _v
        if _v is not None:
            self['csrc'] = _v
        _v = arg.pop('customdata', None)
        _v = customdata if customdata is not None else _v
        if _v is not None:
            self['customdata'] = _v
        _v = arg.pop('customdatasrc', None)
        _v = customdatasrc if customdatasrc is not None else _v
        if _v is not None:
            self['customdatasrc'] = _v
        _v = arg.pop('fill', None)
        _v = fill if fill is not None else _v
        if _v is not None:
            self['fill'] = _v
        _v = arg.pop('fillcolor', None)
        _v = fillcolor if fillcolor is not None else _v
        if _v is not None:
            self['fillcolor'] = _v
        _v = arg.pop('hoverinfo', None)
        _v = hoverinfo if hoverinfo is not None else _v
        if _v is not None:
            self['hoverinfo'] = _v
        _v = arg.pop('hoverinfosrc', None)
        _v = hoverinfosrc if hoverinfosrc is not None else _v
        if _v is not None:
            self['hoverinfosrc'] = _v
        _v = arg.pop('hoverlabel', None)
        _v = hoverlabel if hoverlabel is not None else _v
        if _v is not None:
            self['hoverlabel'] = _v
        _v = arg.pop('hoveron', None)
        _v = hoveron if hoveron is not None else _v
        if _v is not None:
            self['hoveron'] = _v
        _v = arg.pop('hovertemplate', None)
        _v = hovertemplate if hovertemplate is not None else _v
        if _v is not None:
            self['hovertemplate'] = _v
        _v = arg.pop('hovertemplatesrc', None)
        _v = hovertemplatesrc if hovertemplatesrc is not None else _v
        if _v is not None:
            self['hovertemplatesrc'] = _v
        _v = arg.pop('hovertext', None)
        _v = hovertext if hovertext is not None else _v
        if _v is not None:
            self['hovertext'] = _v
        _v = arg.pop('hovertextsrc', None)
        _v = hovertextsrc if hovertextsrc is not None else _v
        if _v is not None:
            self['hovertextsrc'] = _v
        _v = arg.pop('ids', None)
        _v = ids if ids is not None else _v
        if _v is not None:
            self['ids'] = _v
        _v = arg.pop('idssrc', None)
        _v = idssrc if idssrc is not None else _v
        if _v is not None:
            self['idssrc'] = _v
        _v = arg.pop('legend', None)
        _v = legend if legend is not None else _v
        if _v is not None:
            self['legend'] = _v
        _v = arg.pop('legendgroup', None)
        _v = legendgroup if legendgroup is not None else _v
        if _v is not None:
            self['legendgroup'] = _v
        _v = arg.pop('legendgrouptitle', None)
        _v = legendgrouptitle if legendgrouptitle is not None else _v
        if _v is not None:
            self['legendgrouptitle'] = _v
        _v = arg.pop('legendrank', None)
        _v = legendrank if legendrank is not None else _v
        if _v is not None:
            self['legendrank'] = _v
        _v = arg.pop('legendwidth', None)
        _v = legendwidth if legendwidth is not None else _v
        if _v is not None:
            self['legendwidth'] = _v
        _v = arg.pop('line', None)
        _v = line if line is not None else _v
        if _v is not None:
            self['line'] = _v
        _v = arg.pop('marker', None)
        _v = marker if marker is not None else _v
        if _v is not None:
            self['marker'] = _v
        _v = arg.pop('meta', None)
        _v = meta if meta is not None else _v
        if _v is not None:
            self['meta'] = _v
        _v = arg.pop('metasrc', None)
        _v = metasrc if metasrc is not None else _v
        if _v is not None:
            self['metasrc'] = _v
        _v = arg.pop('mode', None)
        _v = mode if mode is not None else _v
        if _v is not None:
            self['mode'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        _v = arg.pop('selected', None)
        _v = selected if selected is not None else _v
        if _v is not None:
            self['selected'] = _v
        _v = arg.pop('selectedpoints', None)
        _v = selectedpoints if selectedpoints is not None else _v
        if _v is not None:
            self['selectedpoints'] = _v
        _v = arg.pop('showlegend', None)
        _v = showlegend if showlegend is not None else _v
        if _v is not None:
            self['showlegend'] = _v
        _v = arg.pop('stream', None)
        _v = stream if stream is not None else _v
        if _v is not None:
            self['stream'] = _v
        _v = arg.pop('subplot', None)
        _v = subplot if subplot is not None else _v
        if _v is not None:
            self['subplot'] = _v
        _v = arg.pop('sum', None)
        _v = sum if sum is not None else _v
        if _v is not None:
            self['sum'] = _v
        _v = arg.pop('text', None)
        _v = text if text is not None else _v
        if _v is not None:
            self['text'] = _v
        _v = arg.pop('textfont', None)
        _v = textfont if textfont is not None else _v
        if _v is not None:
            self['textfont'] = _v
        _v = arg.pop('textposition', None)
        _v = textposition if textposition is not None else _v
        if _v is not None:
            self['textposition'] = _v
        _v = arg.pop('textpositionsrc', None)
        _v = textpositionsrc if textpositionsrc is not None else _v
        if _v is not None:
            self['textpositionsrc'] = _v
        _v = arg.pop('textsrc', None)
        _v = textsrc if textsrc is not None else _v
        if _v is not None:
            self['textsrc'] = _v
        _v = arg.pop('texttemplate', None)
        _v = texttemplate if texttemplate is not None else _v
        if _v is not None:
            self['texttemplate'] = _v
        _v = arg.pop('texttemplatesrc', None)
        _v = texttemplatesrc if texttemplatesrc is not None else _v
        if _v is not None:
            self['texttemplatesrc'] = _v
        _v = arg.pop('uid', None)
        _v = uid if uid is not None else _v
        if _v is not None:
            self['uid'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('unselected', None)
        _v = unselected if unselected is not None else _v
        if _v is not None:
            self['unselected'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._props['type'] = 'scatterternary'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
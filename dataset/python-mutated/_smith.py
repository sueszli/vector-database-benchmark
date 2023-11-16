from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Smith(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.smith'
    _valid_props = {'bgcolor', 'domain', 'imaginaryaxis', 'realaxis'}

    @property
    def bgcolor(self):
        if False:
            i = 10
            return i + 15
        "\n        Set the background color of the subplot\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bgcolor']

    @bgcolor.setter
    def bgcolor(self, val):
        if False:
            i = 10
            return i + 15
        self['bgcolor'] = val

    @property
    def domain(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'domain' property is an instance of Domain\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.smith.Domain`\n          - A dict of string/value properties that will be passed\n            to the Domain constructor\n\n            Supported dict properties:\n\n                column\n                    If there is a layout grid, use the domain for\n                    this column in the grid for this smith subplot\n                    .\n                row\n                    If there is a layout grid, use the domain for\n                    this row in the grid for this smith subplot .\n                x\n                    Sets the horizontal domain of this smith\n                    subplot (in plot fraction).\n                y\n                    Sets the vertical domain of this smith subplot\n                    (in plot fraction).\n\n        Returns\n        -------\n        plotly.graph_objs.layout.smith.Domain\n        "
        return self['domain']

    @domain.setter
    def domain(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['domain'] = val

    @property
    def imaginaryaxis(self):
        if False:
            i = 10
            return i + 15
        '\n        The \'imaginaryaxis\' property is an instance of Imaginaryaxis\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.smith.Imaginaryaxis`\n          - A dict of string/value properties that will be passed\n            to the Imaginaryaxis constructor\n\n            Supported dict properties:\n\n                color\n                    Sets default for all colors associated with\n                    this axis all at once: line, font, tick, and\n                    grid colors. Grid color is lightened by\n                    blending this with the plot background\n                    Individual pieces can override this.\n                gridcolor\n                    Sets the color of the grid lines.\n                griddash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                gridwidth\n                    Sets the width (in px) of the grid lines.\n                hoverformat\n                    Sets the hover text formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format. And for dates see:\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format. We add two\n                    items to d3\'s date formatter: "%h" for half of\n                    the year as a decimal number as well as "%{n}f"\n                    for fractional seconds with n digits. For\n                    example, *2016-10-13 09:15:23.456* with\n                    tickformat "%H~%M~%S.%2f" would display\n                    "09~15~23.46"\n                labelalias\n                    Replacement text for specific tick or hover\n                    labels. For example using {US: \'USA\', CA:\n                    \'Canada\'} changes US to USA and CA to Canada.\n                    The labels we would have shown must match the\n                    keys exactly, after adding any tickprefix or\n                    ticksuffix. For negative numbers the minus sign\n                    symbol used (U+2212) is wider than the regular\n                    ascii dash. That means you need to use −1\n                    instead of -1. labelalias can be used with any\n                    axis type, and both keys (if needed) and values\n                    (if desired) can include html-like tags or\n                    MathJax.\n                layer\n                    Sets the layer on which this axis is displayed.\n                    If *above traces*, this axis is displayed above\n                    all the subplot\'s traces If *below traces*,\n                    this axis is displayed below all the subplot\'s\n                    traces, but above the grid lines. Useful when\n                    used together with scatter-like traces with\n                    `cliponaxis` set to False to show markers\n                    and/or text nodes above this axis.\n                linecolor\n                    Sets the axis line color.\n                linewidth\n                    Sets the width (in px) of the axis line.\n                showgrid\n                    Determines whether or not grid lines are drawn.\n                    If True, the grid lines are drawn at every tick\n                    mark.\n                showline\n                    Determines whether or not a line bounding this\n                    axis is drawn.\n                showticklabels\n                    Determines whether or not the tick labels are\n                    drawn.\n                showtickprefix\n                    If "all", all tick labels are displayed with a\n                    prefix. If "first", only the first tick is\n                    displayed with a prefix. If "last", only the\n                    last tick is displayed with a suffix. If\n                    "none", tick prefixes are hidden.\n                showticksuffix\n                    Same as `showtickprefix` but for tick suffixes.\n                tickcolor\n                    Sets the tick color.\n                tickfont\n                    Sets the tick font.\n                tickformat\n                    Sets the tick label formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format. And for dates see:\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format. We add two\n                    items to d3\'s date formatter: "%h" for half of\n                    the year as a decimal number as well as "%{n}f"\n                    for fractional seconds with n digits. For\n                    example, *2016-10-13 09:15:23.456* with\n                    tickformat "%H~%M~%S.%2f" would display\n                    "09~15~23.46"\n                ticklen\n                    Sets the tick length (in px).\n                tickprefix\n                    Sets a tick label prefix.\n                ticks\n                    Determines whether ticks are drawn or not. If\n                    "", this axis\' ticks are not drawn. If\n                    "outside" ("inside"), this axis\' are drawn\n                    outside (inside) the axis lines.\n                ticksuffix\n                    Sets a tick label suffix.\n                tickvals\n                    Sets the values at which ticks on this axis\n                    appear. Defaults to `realaxis.tickvals` plus\n                    the same as negatives and zero.\n                tickvalssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `tickvals`.\n                tickwidth\n                    Sets the tick width (in px).\n                visible\n                    A single toggle to hide the axis while\n                    preserving interaction like dragging. Default\n                    is true when a cheater plot is present on the\n                    axis, otherwise false\n\n        Returns\n        -------\n        plotly.graph_objs.layout.smith.Imaginaryaxis\n        '
        return self['imaginaryaxis']

    @imaginaryaxis.setter
    def imaginaryaxis(self, val):
        if False:
            while True:
                i = 10
        self['imaginaryaxis'] = val

    @property
    def realaxis(self):
        if False:
            i = 10
            return i + 15
        '\n        The \'realaxis\' property is an instance of Realaxis\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.smith.Realaxis`\n          - A dict of string/value properties that will be passed\n            to the Realaxis constructor\n\n            Supported dict properties:\n\n                color\n                    Sets default for all colors associated with\n                    this axis all at once: line, font, tick, and\n                    grid colors. Grid color is lightened by\n                    blending this with the plot background\n                    Individual pieces can override this.\n                gridcolor\n                    Sets the color of the grid lines.\n                griddash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                gridwidth\n                    Sets the width (in px) of the grid lines.\n                hoverformat\n                    Sets the hover text formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format. And for dates see:\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format. We add two\n                    items to d3\'s date formatter: "%h" for half of\n                    the year as a decimal number as well as "%{n}f"\n                    for fractional seconds with n digits. For\n                    example, *2016-10-13 09:15:23.456* with\n                    tickformat "%H~%M~%S.%2f" would display\n                    "09~15~23.46"\n                labelalias\n                    Replacement text for specific tick or hover\n                    labels. For example using {US: \'USA\', CA:\n                    \'Canada\'} changes US to USA and CA to Canada.\n                    The labels we would have shown must match the\n                    keys exactly, after adding any tickprefix or\n                    ticksuffix. For negative numbers the minus sign\n                    symbol used (U+2212) is wider than the regular\n                    ascii dash. That means you need to use −1\n                    instead of -1. labelalias can be used with any\n                    axis type, and both keys (if needed) and values\n                    (if desired) can include html-like tags or\n                    MathJax.\n                layer\n                    Sets the layer on which this axis is displayed.\n                    If *above traces*, this axis is displayed above\n                    all the subplot\'s traces If *below traces*,\n                    this axis is displayed below all the subplot\'s\n                    traces, but above the grid lines. Useful when\n                    used together with scatter-like traces with\n                    `cliponaxis` set to False to show markers\n                    and/or text nodes above this axis.\n                linecolor\n                    Sets the axis line color.\n                linewidth\n                    Sets the width (in px) of the axis line.\n                showgrid\n                    Determines whether or not grid lines are drawn.\n                    If True, the grid lines are drawn at every tick\n                    mark.\n                showline\n                    Determines whether or not a line bounding this\n                    axis is drawn.\n                showticklabels\n                    Determines whether or not the tick labels are\n                    drawn.\n                showtickprefix\n                    If "all", all tick labels are displayed with a\n                    prefix. If "first", only the first tick is\n                    displayed with a prefix. If "last", only the\n                    last tick is displayed with a suffix. If\n                    "none", tick prefixes are hidden.\n                showticksuffix\n                    Same as `showtickprefix` but for tick suffixes.\n                side\n                    Determines on which side of real axis line the\n                    tick and tick labels appear.\n                tickangle\n                    Sets the angle of the tick labels with respect\n                    to the horizontal. For example, a `tickangle`\n                    of -90 draws the tick labels vertically.\n                tickcolor\n                    Sets the tick color.\n                tickfont\n                    Sets the tick font.\n                tickformat\n                    Sets the tick label formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format. And for dates see:\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format. We add two\n                    items to d3\'s date formatter: "%h" for half of\n                    the year as a decimal number as well as "%{n}f"\n                    for fractional seconds with n digits. For\n                    example, *2016-10-13 09:15:23.456* with\n                    tickformat "%H~%M~%S.%2f" would display\n                    "09~15~23.46"\n                ticklen\n                    Sets the tick length (in px).\n                tickprefix\n                    Sets a tick label prefix.\n                ticks\n                    Determines whether ticks are drawn or not. If\n                    "", this axis\' ticks are not drawn. If "top"\n                    ("bottom"), this axis\' are drawn above (below)\n                    the axis line.\n                ticksuffix\n                    Sets a tick label suffix.\n                tickvals\n                    Sets the values at which ticks on this axis\n                    appear.\n                tickvalssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `tickvals`.\n                tickwidth\n                    Sets the tick width (in px).\n                visible\n                    A single toggle to hide the axis while\n                    preserving interaction like dragging. Default\n                    is true when a cheater plot is present on the\n                    axis, otherwise false\n\n        Returns\n        -------\n        plotly.graph_objs.layout.smith.Realaxis\n        '
        return self['realaxis']

    @realaxis.setter
    def realaxis(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['realaxis'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        bgcolor\n            Set the background color of the subplot\n        domain\n            :class:`plotly.graph_objects.layout.smith.Domain`\n            instance or dict with compatible properties\n        imaginaryaxis\n            :class:`plotly.graph_objects.layout.smith.Imaginaryaxis\n            ` instance or dict with compatible properties\n        realaxis\n            :class:`plotly.graph_objects.layout.smith.Realaxis`\n            instance or dict with compatible properties\n        '

    def __init__(self, arg=None, bgcolor=None, domain=None, imaginaryaxis=None, realaxis=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Smith object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.layout.Smith`\n        bgcolor\n            Set the background color of the subplot\n        domain\n            :class:`plotly.graph_objects.layout.smith.Domain`\n            instance or dict with compatible properties\n        imaginaryaxis\n            :class:`plotly.graph_objects.layout.smith.Imaginaryaxis\n            ` instance or dict with compatible properties\n        realaxis\n            :class:`plotly.graph_objects.layout.smith.Realaxis`\n            instance or dict with compatible properties\n\n        Returns\n        -------\n        Smith\n        '
        super(Smith, self).__init__('smith')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Smith\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Smith`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('bgcolor', None)
        _v = bgcolor if bgcolor is not None else _v
        if _v is not None:
            self['bgcolor'] = _v
        _v = arg.pop('domain', None)
        _v = domain if domain is not None else _v
        if _v is not None:
            self['domain'] = _v
        _v = arg.pop('imaginaryaxis', None)
        _v = imaginaryaxis if imaginaryaxis is not None else _v
        if _v is not None:
            self['imaginaryaxis'] = _v
        _v = arg.pop('realaxis', None)
        _v = realaxis if realaxis is not None else _v
        if _v is not None:
            self['realaxis'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
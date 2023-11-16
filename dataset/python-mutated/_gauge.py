from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Gauge(_BaseTraceHierarchyType):
    _parent_path_str = 'indicator'
    _path_str = 'indicator.gauge'
    _valid_props = {'axis', 'bar', 'bgcolor', 'bordercolor', 'borderwidth', 'shape', 'stepdefaults', 'steps', 'threshold'}

    @property
    def axis(self):
        if False:
            return 10
        '\n        The \'axis\' property is an instance of Axis\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.gauge.Axis`\n          - A dict of string/value properties that will be passed\n            to the Axis constructor\n\n            Supported dict properties:\n\n                dtick\n                    Sets the step in-between ticks on this axis.\n                    Use with `tick0`. Must be a positive number, or\n                    special strings available to "log" and "date"\n                    axes. If the axis `type` is "log", then ticks\n                    are set every 10^(n*dtick) where n is the tick\n                    number. For example, to set a tick mark at 1,\n                    10, 100, 1000, ... set dtick to 1. To set tick\n                    marks at 1, 100, 10000, ... set dtick to 2. To\n                    set tick marks at 1, 5, 25, 125, 625, 3125, ...\n                    set dtick to log_10(5), or 0.69897000433. "log"\n                    has several special values; "L<f>", where `f`\n                    is a positive number, gives ticks linearly\n                    spaced in value (but not position). For example\n                    `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n                    at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10\n                    plus small digits between, use "D1" (all\n                    digits) or "D2" (only 2 and 5). `tick0` is\n                    ignored for "D1" and "D2". If the axis `type`\n                    is "date", then you must convert the time to\n                    milliseconds. For example, to set the interval\n                    between ticks to one day, set `dtick` to\n                    86400000.0. "date" also has special values\n                    "M<n>" gives ticks spaced by a number of\n                    months. `n` must be a positive integer. To set\n                    ticks on the 15th of every third month, set\n                    `tick0` to "2000-01-15" and `dtick` to "M3". To\n                    set ticks every 4 years, set `dtick` to "M48"\n                exponentformat\n                    Determines a formatting rule for the tick\n                    exponents. For example, consider the number\n                    1,000,000,000. If "none", it appears as\n                    1,000,000,000. If "e", 1e+9. If "E", 1E+9. If\n                    "power", 1x10^9 (with 9 in a super script). If\n                    "SI", 1G. If "B", 1B.\n                labelalias\n                    Replacement text for specific tick or hover\n                    labels. For example using {US: \'USA\', CA:\n                    \'Canada\'} changes US to USA and CA to Canada.\n                    The labels we would have shown must match the\n                    keys exactly, after adding any tickprefix or\n                    ticksuffix. For negative numbers the minus sign\n                    symbol used (U+2212) is wider than the regular\n                    ascii dash. That means you need to use −1\n                    instead of -1. labelalias can be used with any\n                    axis type, and both keys (if needed) and values\n                    (if desired) can include html-like tags or\n                    MathJax.\n                minexponent\n                    Hide SI prefix for 10^n if |n| is below this\n                    number. This only has an effect when\n                    `tickformat` is "SI" or "B".\n                nticks\n                    Specifies the maximum number of ticks for the\n                    particular axis. The actual number of ticks\n                    will be chosen automatically to be less than or\n                    equal to `nticks`. Has an effect only if\n                    `tickmode` is set to "auto".\n                range\n                    Sets the range of this axis.\n                separatethousands\n                    If "true", even 4-digit integers are separated\n                showexponent\n                    If "all", all exponents are shown besides their\n                    significands. If "first", only the exponent of\n                    the first tick is shown. If "last", only the\n                    exponent of the last tick is shown. If "none",\n                    no exponents appear.\n                showticklabels\n                    Determines whether or not the tick labels are\n                    drawn.\n                showtickprefix\n                    If "all", all tick labels are displayed with a\n                    prefix. If "first", only the first tick is\n                    displayed with a prefix. If "last", only the\n                    last tick is displayed with a suffix. If\n                    "none", tick prefixes are hidden.\n                showticksuffix\n                    Same as `showtickprefix` but for tick suffixes.\n                tick0\n                    Sets the placement of the first tick on this\n                    axis. Use with `dtick`. If the axis `type` is\n                    "log", then you must take the log of your\n                    starting tick (e.g. to set the starting tick to\n                    100, set the `tick0` to 2) except when\n                    `dtick`=*L<f>* (see `dtick` for more info). If\n                    the axis `type` is "date", it should be a date\n                    string, like date data. If the axis `type` is\n                    "category", it should be a number, using the\n                    scale where each category is assigned a serial\n                    number from zero in the order it appears.\n                tickangle\n                    Sets the angle of the tick labels with respect\n                    to the horizontal. For example, a `tickangle`\n                    of -90 draws the tick labels vertically.\n                tickcolor\n                    Sets the tick color.\n                tickfont\n                    Sets the color bar\'s tick label font\n                tickformat\n                    Sets the tick label formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format. And for dates see:\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format. We add two\n                    items to d3\'s date formatter: "%h" for half of\n                    the year as a decimal number as well as "%{n}f"\n                    for fractional seconds with n digits. For\n                    example, *2016-10-13 09:15:23.456* with\n                    tickformat "%H~%M~%S.%2f" would display\n                    "09~15~23.46"\n                tickformatstops\n                    A tuple of :class:`plotly.graph_objects.indicat\n                    or.gauge.axis.Tickformatstop` instances or\n                    dicts with compatible properties\n                tickformatstopdefaults\n                    When used in a template (as layout.template.dat\n                    a.indicator.gauge.axis.tickformatstopdefaults),\n                    sets the default property values to use for\n                    elements of\n                    indicator.gauge.axis.tickformatstops\n                ticklabelstep\n                    Sets the spacing between tick labels as\n                    compared to the spacing between ticks. A value\n                    of 1 (default) means each tick gets a label. A\n                    value of 2 means shows every 2nd label. A\n                    larger value n means only every nth tick is\n                    labeled. `tick0` determines which labels are\n                    shown. Not implemented for axes with `type`\n                    "log" or "multicategory", or when `tickmode` is\n                    "array".\n                ticklen\n                    Sets the tick length (in px).\n                tickmode\n                    Sets the tick mode for this axis. If "auto",\n                    the number of ticks is set via `nticks`. If\n                    "linear", the placement of the ticks is\n                    determined by a starting position `tick0` and a\n                    tick step `dtick` ("linear" is the default\n                    value if `tick0` and `dtick` are provided). If\n                    "array", the placement of the ticks is set via\n                    `tickvals` and the tick text is `ticktext`.\n                    ("array" is the default value if `tickvals` is\n                    provided).\n                tickprefix\n                    Sets a tick label prefix.\n                ticks\n                    Determines whether ticks are drawn or not. If\n                    "", this axis\' ticks are not drawn. If\n                    "outside" ("inside"), this axis\' are drawn\n                    outside (inside) the axis lines.\n                ticksuffix\n                    Sets a tick label suffix.\n                ticktext\n                    Sets the text displayed at the ticks position\n                    via `tickvals`. Only has an effect if\n                    `tickmode` is set to "array". Used with\n                    `tickvals`.\n                ticktextsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `ticktext`.\n                tickvals\n                    Sets the values at which ticks on this axis\n                    appear. Only has an effect if `tickmode` is set\n                    to "array". Used with `ticktext`.\n                tickvalssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `tickvals`.\n                tickwidth\n                    Sets the tick width (in px).\n                visible\n                    A single toggle to hide the axis while\n                    preserving interaction like dragging. Default\n                    is true when a cheater plot is present on the\n                    axis, otherwise false\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.gauge.Axis\n        '
        return self['axis']

    @axis.setter
    def axis(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['axis'] = val

    @property
    def bar(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the appearance of the gauge's value\n\n        The 'bar' property is an instance of Bar\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.gauge.Bar`\n          - A dict of string/value properties that will be passed\n            to the Bar constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the background color of the arc.\n                line\n                    :class:`plotly.graph_objects.indicator.gauge.ba\n                    r.Line` instance or dict with compatible\n                    properties\n                thickness\n                    Sets the thickness of the bar as a fraction of\n                    the total thickness of the gauge.\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.gauge.Bar\n        "
        return self['bar']

    @bar.setter
    def bar(self, val):
        if False:
            print('Hello World!')
        self['bar'] = val

    @property
    def bgcolor(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the gauge background color.\n\n        The 'bgcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
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
        "\n        Sets the color of the border enclosing the gauge.\n\n        The 'bordercolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['bordercolor']

    @bordercolor.setter
    def bordercolor(self, val):
        if False:
            print('Hello World!')
        self['bordercolor'] = val

    @property
    def borderwidth(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the width (in px) of the border enclosing the gauge.\n\n        The 'borderwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['borderwidth']

    @borderwidth.setter
    def borderwidth(self, val):
        if False:
            return 10
        self['borderwidth'] = val

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the shape of the gauge\n\n        The 'shape' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['angular', 'bullet']\n\n        Returns\n        -------\n        Any\n        "
        return self['shape']

    @shape.setter
    def shape(self, val):
        if False:
            i = 10
            return i + 15
        self['shape'] = val

    @property
    def steps(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'steps' property is a tuple of instances of\n        Step that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.indicator.gauge.Step\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Step constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the background color of the arc.\n                line\n                    :class:`plotly.graph_objects.indicator.gauge.st\n                    ep.Line` instance or dict with compatible\n                    properties\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                range\n                    Sets the range of this axis.\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n                thickness\n                    Sets the thickness of the bar as a fraction of\n                    the total thickness of the gauge.\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.indicator.gauge.Step]\n        "
        return self['steps']

    @steps.setter
    def steps(self, val):
        if False:
            i = 10
            return i + 15
        self['steps'] = val

    @property
    def stepdefaults(self):
        if False:
            print('Hello World!')
        "\n        When used in a template (as\n        layout.template.data.indicator.gauge.stepdefaults), sets the\n        default property values to use for elements of\n        indicator.gauge.steps\n\n        The 'stepdefaults' property is an instance of Step\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.gauge.Step`\n          - A dict of string/value properties that will be passed\n            to the Step constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.gauge.Step\n        "
        return self['stepdefaults']

    @stepdefaults.setter
    def stepdefaults(self, val):
        if False:
            return 10
        self['stepdefaults'] = val

    @property
    def threshold(self):
        if False:
            print('Hello World!')
        "\n        The 'threshold' property is an instance of Threshold\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.gauge.Threshold`\n          - A dict of string/value properties that will be passed\n            to the Threshold constructor\n\n            Supported dict properties:\n\n                line\n                    :class:`plotly.graph_objects.indicator.gauge.th\n                    reshold.Line` instance or dict with compatible\n                    properties\n                thickness\n                    Sets the thickness of the threshold line as a\n                    fraction of the thickness of the gauge.\n                value\n                    Sets a treshold value drawn as a line.\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.gauge.Threshold\n        "
        return self['threshold']

    @threshold.setter
    def threshold(self, val):
        if False:
            print('Hello World!')
        self['threshold'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return "        axis\n            :class:`plotly.graph_objects.indicator.gauge.Axis`\n            instance or dict with compatible properties\n        bar\n            Set the appearance of the gauge's value\n        bgcolor\n            Sets the gauge background color.\n        bordercolor\n            Sets the color of the border enclosing the gauge.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            gauge.\n        shape\n            Set the shape of the gauge\n        steps\n            A tuple of\n            :class:`plotly.graph_objects.indicator.gauge.Step`\n            instances or dicts with compatible properties\n        stepdefaults\n            When used in a template (as\n            layout.template.data.indicator.gauge.stepdefaults),\n            sets the default property values to use for elements of\n            indicator.gauge.steps\n        threshold\n            :class:`plotly.graph_objects.indicator.gauge.Threshold`\n            instance or dict with compatible properties\n        "

    def __init__(self, arg=None, axis=None, bar=None, bgcolor=None, bordercolor=None, borderwidth=None, shape=None, steps=None, stepdefaults=None, threshold=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a new Gauge object\n\n        The gauge of the Indicator plot.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.indicator.Gauge`\n        axis\n            :class:`plotly.graph_objects.indicator.gauge.Axis`\n            instance or dict with compatible properties\n        bar\n            Set the appearance of the gauge's value\n        bgcolor\n            Sets the gauge background color.\n        bordercolor\n            Sets the color of the border enclosing the gauge.\n        borderwidth\n            Sets the width (in px) of the border enclosing the\n            gauge.\n        shape\n            Set the shape of the gauge\n        steps\n            A tuple of\n            :class:`plotly.graph_objects.indicator.gauge.Step`\n            instances or dicts with compatible properties\n        stepdefaults\n            When used in a template (as\n            layout.template.data.indicator.gauge.stepdefaults),\n            sets the default property values to use for elements of\n            indicator.gauge.steps\n        threshold\n            :class:`plotly.graph_objects.indicator.gauge.Threshold`\n            instance or dict with compatible properties\n\n        Returns\n        -------\n        Gauge\n        "
        super(Gauge, self).__init__('gauge')
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
            raise ValueError('The first argument to the plotly.graph_objs.indicator.Gauge\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.indicator.Gauge`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('axis', None)
        _v = axis if axis is not None else _v
        if _v is not None:
            self['axis'] = _v
        _v = arg.pop('bar', None)
        _v = bar if bar is not None else _v
        if _v is not None:
            self['bar'] = _v
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
        _v = arg.pop('shape', None)
        _v = shape if shape is not None else _v
        if _v is not None:
            self['shape'] = _v
        _v = arg.pop('steps', None)
        _v = steps if steps is not None else _v
        if _v is not None:
            self['steps'] = _v
        _v = arg.pop('stepdefaults', None)
        _v = stepdefaults if stepdefaults is not None else _v
        if _v is not None:
            self['stepdefaults'] = _v
        _v = arg.pop('threshold', None)
        _v = threshold if threshold is not None else _v
        if _v is not None:
            self['threshold'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
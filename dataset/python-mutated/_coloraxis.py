from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Coloraxis(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.coloraxis'
    _valid_props = {'autocolorscale', 'cauto', 'cmax', 'cmid', 'cmin', 'colorbar', 'colorscale', 'reversescale', 'showscale'}

    @property
    def autocolorscale(self):
        if False:
            return 10
        "\n        Determines whether the colorscale is a default palette\n        (`autocolorscale: true`) or the palette determined by\n        `colorscale`. In case `colorscale` is unspecified or\n        `autocolorscale` is true, the default palette will be chosen\n        according to whether numbers in the `color` array are all\n        positive, all negative or mixed.\n\n        The 'autocolorscale' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['autocolorscale']

    @autocolorscale.setter
    def autocolorscale(self, val):
        if False:
            while True:
                i = 10
        self['autocolorscale'] = val

    @property
    def cauto(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not the color domain is computed with\n        respect to the input data (here corresponding trace color\n        array(s)) or the bounds set in `cmin` and `cmax` Defaults to\n        `false` when `cmin` and `cmax` are set by the user.\n\n        The 'cauto' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['cauto']

    @cauto.setter
    def cauto(self, val):
        if False:
            i = 10
            return i + 15
        self['cauto'] = val

    @property
    def cmax(self):
        if False:
            while True:
                i = 10
        "\n        Sets the upper bound of the color domain. Value should have the\n        same units as corresponding trace color array(s) and if set,\n        `cmin` must be set as well.\n\n        The 'cmax' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['cmax']

    @cmax.setter
    def cmax(self, val):
        if False:
            print('Hello World!')
        self['cmax'] = val

    @property
    def cmid(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the mid-point of the color domain by scaling `cmin` and/or\n        `cmax` to be equidistant to this point. Value should have the\n        same units as corresponding trace color array(s). Has no effect\n        when `cauto` is `false`.\n\n        The 'cmid' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['cmid']

    @cmid.setter
    def cmid(self, val):
        if False:
            i = 10
            return i + 15
        self['cmid'] = val

    @property
    def cmin(self):
        if False:
            while True:
                i = 10
        "\n        Sets the lower bound of the color domain. Value should have the\n        same units as corresponding trace color array(s) and if set,\n        `cmax` must be set as well.\n\n        The 'cmin' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['cmin']

    @cmin.setter
    def cmin(self, val):
        if False:
            return 10
        self['cmin'] = val

    @property
    def colorbar(self):
        if False:
            print('Hello World!')
        '\n        The \'colorbar\' property is an instance of ColorBar\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.coloraxis.ColorBar`\n          - A dict of string/value properties that will be passed\n            to the ColorBar constructor\n\n            Supported dict properties:\n\n                bgcolor\n                    Sets the color of padded area.\n                bordercolor\n                    Sets the axis line color.\n                borderwidth\n                    Sets the width (in px) or the border enclosing\n                    this color bar.\n                dtick\n                    Sets the step in-between ticks on this axis.\n                    Use with `tick0`. Must be a positive number, or\n                    special strings available to "log" and "date"\n                    axes. If the axis `type` is "log", then ticks\n                    are set every 10^(n*dtick) where n is the tick\n                    number. For example, to set a tick mark at 1,\n                    10, 100, 1000, ... set dtick to 1. To set tick\n                    marks at 1, 100, 10000, ... set dtick to 2. To\n                    set tick marks at 1, 5, 25, 125, 625, 3125, ...\n                    set dtick to log_10(5), or 0.69897000433. "log"\n                    has several special values; "L<f>", where `f`\n                    is a positive number, gives ticks linearly\n                    spaced in value (but not position). For example\n                    `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n                    at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10\n                    plus small digits between, use "D1" (all\n                    digits) or "D2" (only 2 and 5). `tick0` is\n                    ignored for "D1" and "D2". If the axis `type`\n                    is "date", then you must convert the time to\n                    milliseconds. For example, to set the interval\n                    between ticks to one day, set `dtick` to\n                    86400000.0. "date" also has special values\n                    "M<n>" gives ticks spaced by a number of\n                    months. `n` must be a positive integer. To set\n                    ticks on the 15th of every third month, set\n                    `tick0` to "2000-01-15" and `dtick` to "M3". To\n                    set ticks every 4 years, set `dtick` to "M48"\n                exponentformat\n                    Determines a formatting rule for the tick\n                    exponents. For example, consider the number\n                    1,000,000,000. If "none", it appears as\n                    1,000,000,000. If "e", 1e+9. If "E", 1E+9. If\n                    "power", 1x10^9 (with 9 in a super script). If\n                    "SI", 1G. If "B", 1B.\n                labelalias\n                    Replacement text for specific tick or hover\n                    labels. For example using {US: \'USA\', CA:\n                    \'Canada\'} changes US to USA and CA to Canada.\n                    The labels we would have shown must match the\n                    keys exactly, after adding any tickprefix or\n                    ticksuffix. For negative numbers the minus sign\n                    symbol used (U+2212) is wider than the regular\n                    ascii dash. That means you need to use −1\n                    instead of -1. labelalias can be used with any\n                    axis type, and both keys (if needed) and values\n                    (if desired) can include html-like tags or\n                    MathJax.\n                len\n                    Sets the length of the color bar This measure\n                    excludes the padding of both ends. That is, the\n                    color bar length is this length minus the\n                    padding on both ends.\n                lenmode\n                    Determines whether this color bar\'s length\n                    (i.e. the measure in the color variation\n                    direction) is set in units of plot "fraction"\n                    or in *pixels. Use `len` to set the value.\n                minexponent\n                    Hide SI prefix for 10^n if |n| is below this\n                    number. This only has an effect when\n                    `tickformat` is "SI" or "B".\n                nticks\n                    Specifies the maximum number of ticks for the\n                    particular axis. The actual number of ticks\n                    will be chosen automatically to be less than or\n                    equal to `nticks`. Has an effect only if\n                    `tickmode` is set to "auto".\n                orientation\n                    Sets the orientation of the colorbar.\n                outlinecolor\n                    Sets the axis line color.\n                outlinewidth\n                    Sets the width (in px) of the axis line.\n                separatethousands\n                    If "true", even 4-digit integers are separated\n                showexponent\n                    If "all", all exponents are shown besides their\n                    significands. If "first", only the exponent of\n                    the first tick is shown. If "last", only the\n                    exponent of the last tick is shown. If "none",\n                    no exponents appear.\n                showticklabels\n                    Determines whether or not the tick labels are\n                    drawn.\n                showtickprefix\n                    If "all", all tick labels are displayed with a\n                    prefix. If "first", only the first tick is\n                    displayed with a prefix. If "last", only the\n                    last tick is displayed with a suffix. If\n                    "none", tick prefixes are hidden.\n                showticksuffix\n                    Same as `showtickprefix` but for tick suffixes.\n                thickness\n                    Sets the thickness of the color bar This\n                    measure excludes the size of the padding, ticks\n                    and labels.\n                thicknessmode\n                    Determines whether this color bar\'s thickness\n                    (i.e. the measure in the constant color\n                    direction) is set in units of plot "fraction"\n                    or in "pixels". Use `thickness` to set the\n                    value.\n                tick0\n                    Sets the placement of the first tick on this\n                    axis. Use with `dtick`. If the axis `type` is\n                    "log", then you must take the log of your\n                    starting tick (e.g. to set the starting tick to\n                    100, set the `tick0` to 2) except when\n                    `dtick`=*L<f>* (see `dtick` for more info). If\n                    the axis `type` is "date", it should be a date\n                    string, like date data. If the axis `type` is\n                    "category", it should be a number, using the\n                    scale where each category is assigned a serial\n                    number from zero in the order it appears.\n                tickangle\n                    Sets the angle of the tick labels with respect\n                    to the horizontal. For example, a `tickangle`\n                    of -90 draws the tick labels vertically.\n                tickcolor\n                    Sets the tick color.\n                tickfont\n                    Sets the color bar\'s tick label font\n                tickformat\n                    Sets the tick label formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format. And for dates see:\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format. We add two\n                    items to d3\'s date formatter: "%h" for half of\n                    the year as a decimal number as well as "%{n}f"\n                    for fractional seconds with n digits. For\n                    example, *2016-10-13 09:15:23.456* with\n                    tickformat "%H~%M~%S.%2f" would display\n                    "09~15~23.46"\n                tickformatstops\n                    A tuple of :class:`plotly.graph_objects.layout.\n                    coloraxis.colorbar.Tickformatstop` instances or\n                    dicts with compatible properties\n                tickformatstopdefaults\n                    When used in a template (as layout.template.lay\n                    out.coloraxis.colorbar.tickformatstopdefaults),\n                    sets the default property values to use for\n                    elements of\n                    layout.coloraxis.colorbar.tickformatstops\n                ticklabeloverflow\n                    Determines how we handle tick labels that would\n                    overflow either the graph div or the domain of\n                    the axis. The default value for inside tick\n                    labels is *hide past domain*. In other cases\n                    the default is *hide past div*.\n                ticklabelposition\n                    Determines where tick labels are drawn relative\n                    to the ticks. Left and right options are used\n                    when `orientation` is "h", top and bottom when\n                    `orientation` is "v".\n                ticklabelstep\n                    Sets the spacing between tick labels as\n                    compared to the spacing between ticks. A value\n                    of 1 (default) means each tick gets a label. A\n                    value of 2 means shows every 2nd label. A\n                    larger value n means only every nth tick is\n                    labeled. `tick0` determines which labels are\n                    shown. Not implemented for axes with `type`\n                    "log" or "multicategory", or when `tickmode` is\n                    "array".\n                ticklen\n                    Sets the tick length (in px).\n                tickmode\n                    Sets the tick mode for this axis. If "auto",\n                    the number of ticks is set via `nticks`. If\n                    "linear", the placement of the ticks is\n                    determined by a starting position `tick0` and a\n                    tick step `dtick` ("linear" is the default\n                    value if `tick0` and `dtick` are provided). If\n                    "array", the placement of the ticks is set via\n                    `tickvals` and the tick text is `ticktext`.\n                    ("array" is the default value if `tickvals` is\n                    provided).\n                tickprefix\n                    Sets a tick label prefix.\n                ticks\n                    Determines whether ticks are drawn or not. If\n                    "", this axis\' ticks are not drawn. If\n                    "outside" ("inside"), this axis\' are drawn\n                    outside (inside) the axis lines.\n                ticksuffix\n                    Sets a tick label suffix.\n                ticktext\n                    Sets the text displayed at the ticks position\n                    via `tickvals`. Only has an effect if\n                    `tickmode` is set to "array". Used with\n                    `tickvals`.\n                ticktextsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `ticktext`.\n                tickvals\n                    Sets the values at which ticks on this axis\n                    appear. Only has an effect if `tickmode` is set\n                    to "array". Used with `ticktext`.\n                tickvalssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `tickvals`.\n                tickwidth\n                    Sets the tick width (in px).\n                title\n                    :class:`plotly.graph_objects.layout.coloraxis.c\n                    olorbar.Title` instance or dict with compatible\n                    properties\n                titlefont\n                    Deprecated: Please use\n                    layout.coloraxis.colorbar.title.font instead.\n                    Sets this color bar\'s title font. Note that the\n                    title\'s font used to be set by the now\n                    deprecated `titlefont` attribute.\n                titleside\n                    Deprecated: Please use\n                    layout.coloraxis.colorbar.title.side instead.\n                    Determines the location of color bar\'s title\n                    with respect to the color bar. Defaults to\n                    "top" when `orientation` if "v" and  defaults\n                    to "right" when `orientation` if "h". Note that\n                    the title\'s location used to be set by the now\n                    deprecated `titleside` attribute.\n                x\n                    Sets the x position with respect to `xref` of\n                    the color bar (in plot fraction). When `xref`\n                    is "paper", defaults to 1.02 when `orientation`\n                    is "v" and 0.5 when `orientation` is "h". When\n                    `xref` is "container", defaults to 1 when\n                    `orientation` is "v" and 0.5 when `orientation`\n                    is "h". Must be between 0 and 1 if `xref` is\n                    "container" and between "-2" and 3 if `xref` is\n                    "paper".\n                xanchor\n                    Sets this color bar\'s horizontal position\n                    anchor. This anchor binds the `x` position to\n                    the "left", "center" or "right" of the color\n                    bar. Defaults to "left" when `orientation` is\n                    "v" and "center" when `orientation` is "h".\n                xpad\n                    Sets the amount of padding (in px) along the x\n                    direction.\n                xref\n                    Sets the container `x` refers to. "container"\n                    spans the entire `width` of the plot. "paper"\n                    refers to the width of the plotting area only.\n                y\n                    Sets the y position with respect to `yref` of\n                    the color bar (in plot fraction). When `yref`\n                    is "paper", defaults to 0.5 when `orientation`\n                    is "v" and 1.02 when `orientation` is "h". When\n                    `yref` is "container", defaults to 0.5 when\n                    `orientation` is "v" and 1 when `orientation`\n                    is "h". Must be between 0 and 1 if `yref` is\n                    "container" and between "-2" and 3 if `yref` is\n                    "paper".\n                yanchor\n                    Sets this color bar\'s vertical position anchor\n                    This anchor binds the `y` position to the\n                    "top", "middle" or "bottom" of the color bar.\n                    Defaults to "middle" when `orientation` is "v"\n                    and "bottom" when `orientation` is "h".\n                ypad\n                    Sets the amount of padding (in px) along the y\n                    direction.\n                yref\n                    Sets the container `y` refers to. "container"\n                    spans the entire `height` of the plot. "paper"\n                    refers to the height of the plotting area only.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.coloraxis.ColorBar\n        '
        return self['colorbar']

    @colorbar.setter
    def colorbar(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['colorbar'] = val

    @property
    def colorscale(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the colorscale. The colorscale must be an array containing\n        arrays mapping a normalized value to an rgb, rgba, hex, hsl,\n        hsv, or named color string. At minimum, a mapping for the\n        lowest (0) and highest (1) values are required. For example,\n        `[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]`. To control the\n        bounds of the colorscale in color space, use `cmin` and `cmax`.\n        Alternatively, `colorscale` may be a palette name string of the\n        following list: Blackbody,Bluered,Blues,Cividis,Earth,Electric,\n        Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,RdBu,Reds,Viridis,\n        YlGnBu,YlOrRd.\n\n        The 'colorscale' property is a colorscale and may be\n        specified as:\n          - A list of colors that will be spaced evenly to create the colorscale.\n            Many predefined colorscale lists are included in the sequential, diverging,\n            and cyclical modules in the plotly.colors package.\n          - A list of 2-element lists where the first element is the\n            normalized color level value (starting at 0 and ending at 1),\n            and the second item is a valid color string.\n            (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])\n          - One of the following named colorscales:\n                ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',\n                 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',\n                 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',\n                 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',\n                 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',\n                 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',\n                 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',\n                 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',\n                 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',\n                 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',\n                 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',\n                 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',\n                 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',\n                 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',\n                 'ylorrd'].\n            Appending '_r' to a named colorscale reverses it.\n\n        Returns\n        -------\n        str\n        "
        return self['colorscale']

    @colorscale.setter
    def colorscale(self, val):
        if False:
            while True:
                i = 10
        self['colorscale'] = val

    @property
    def reversescale(self):
        if False:
            print('Hello World!')
        "\n        Reverses the color mapping if true. If true, `cmin` will\n        correspond to the last color in the array and `cmax` will\n        correspond to the first color.\n\n        The 'reversescale' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['reversescale']

    @reversescale.setter
    def reversescale(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['reversescale'] = val

    @property
    def showscale(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not a colorbar is displayed for this\n        trace.\n\n        The 'showscale' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showscale']

    @showscale.setter
    def showscale(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showscale'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return "        autocolorscale\n            Determines whether the colorscale is a default palette\n            (`autocolorscale: true`) or the palette determined by\n            `colorscale`. In case `colorscale` is unspecified or\n            `autocolorscale` is true, the default palette will be\n            chosen according to whether numbers in the `color`\n            array are all positive, all negative or mixed.\n        cauto\n            Determines whether or not the color domain is computed\n            with respect to the input data (here corresponding\n            trace color array(s)) or the bounds set in `cmin` and\n            `cmax` Defaults to `false` when `cmin` and `cmax` are\n            set by the user.\n        cmax\n            Sets the upper bound of the color domain. Value should\n            have the same units as corresponding trace color\n            array(s) and if set, `cmin` must be set as well.\n        cmid\n            Sets the mid-point of the color domain by scaling\n            `cmin` and/or `cmax` to be equidistant to this point.\n            Value should have the same units as corresponding trace\n            color array(s). Has no effect when `cauto` is `false`.\n        cmin\n            Sets the lower bound of the color domain. Value should\n            have the same units as corresponding trace color\n            array(s) and if set, `cmax` must be set as well.\n        colorbar\n            :class:`plotly.graph_objects.layout.coloraxis.ColorBar`\n            instance or dict with compatible properties\n        colorscale\n            Sets the colorscale. The colorscale must be an array\n            containing arrays mapping a normalized value to an rgb,\n            rgba, hex, hsl, hsv, or named color string. At minimum,\n            a mapping for the lowest (0) and highest (1) values are\n            required. For example, `[[0, 'rgb(0,0,255)'], [1,\n            'rgb(255,0,0)']]`. To control the bounds of the\n            colorscale in color space, use `cmin` and `cmax`.\n            Alternatively, `colorscale` may be a palette name\n            string of the following list: Blackbody,Bluered,Blues,C\n            ividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portl\n            and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.\n        reversescale\n            Reverses the color mapping if true. If true, `cmin`\n            will correspond to the last color in the array and\n            `cmax` will correspond to the first color.\n        showscale\n            Determines whether or not a colorbar is displayed for\n            this trace.\n        "

    def __init__(self, arg=None, autocolorscale=None, cauto=None, cmax=None, cmid=None, cmin=None, colorbar=None, colorscale=None, reversescale=None, showscale=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Construct a new Coloraxis object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Coloraxis`\n        autocolorscale\n            Determines whether the colorscale is a default palette\n            (`autocolorscale: true`) or the palette determined by\n            `colorscale`. In case `colorscale` is unspecified or\n            `autocolorscale` is true, the default palette will be\n            chosen according to whether numbers in the `color`\n            array are all positive, all negative or mixed.\n        cauto\n            Determines whether or not the color domain is computed\n            with respect to the input data (here corresponding\n            trace color array(s)) or the bounds set in `cmin` and\n            `cmax` Defaults to `false` when `cmin` and `cmax` are\n            set by the user.\n        cmax\n            Sets the upper bound of the color domain. Value should\n            have the same units as corresponding trace color\n            array(s) and if set, `cmin` must be set as well.\n        cmid\n            Sets the mid-point of the color domain by scaling\n            `cmin` and/or `cmax` to be equidistant to this point.\n            Value should have the same units as corresponding trace\n            color array(s). Has no effect when `cauto` is `false`.\n        cmin\n            Sets the lower bound of the color domain. Value should\n            have the same units as corresponding trace color\n            array(s) and if set, `cmax` must be set as well.\n        colorbar\n            :class:`plotly.graph_objects.layout.coloraxis.ColorBar`\n            instance or dict with compatible properties\n        colorscale\n            Sets the colorscale. The colorscale must be an array\n            containing arrays mapping a normalized value to an rgb,\n            rgba, hex, hsl, hsv, or named color string. At minimum,\n            a mapping for the lowest (0) and highest (1) values are\n            required. For example, `[[0, 'rgb(0,0,255)'], [1,\n            'rgb(255,0,0)']]`. To control the bounds of the\n            colorscale in color space, use `cmin` and `cmax`.\n            Alternatively, `colorscale` may be a palette name\n            string of the following list: Blackbody,Bluered,Blues,C\n            ividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portl\n            and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.\n        reversescale\n            Reverses the color mapping if true. If true, `cmin`\n            will correspond to the last color in the array and\n            `cmax` will correspond to the first color.\n        showscale\n            Determines whether or not a colorbar is displayed for\n            this trace.\n\n        Returns\n        -------\n        Coloraxis\n        "
        super(Coloraxis, self).__init__('coloraxis')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Coloraxis\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Coloraxis`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('autocolorscale', None)
        _v = autocolorscale if autocolorscale is not None else _v
        if _v is not None:
            self['autocolorscale'] = _v
        _v = arg.pop('cauto', None)
        _v = cauto if cauto is not None else _v
        if _v is not None:
            self['cauto'] = _v
        _v = arg.pop('cmax', None)
        _v = cmax if cmax is not None else _v
        if _v is not None:
            self['cmax'] = _v
        _v = arg.pop('cmid', None)
        _v = cmid if cmid is not None else _v
        if _v is not None:
            self['cmid'] = _v
        _v = arg.pop('cmin', None)
        _v = cmin if cmin is not None else _v
        if _v is not None:
            self['cmin'] = _v
        _v = arg.pop('colorbar', None)
        _v = colorbar if colorbar is not None else _v
        if _v is not None:
            self['colorbar'] = _v
        _v = arg.pop('colorscale', None)
        _v = colorscale if colorscale is not None else _v
        if _v is not None:
            self['colorscale'] = _v
        _v = arg.pop('reversescale', None)
        _v = reversescale if reversescale is not None else _v
        if _v is not None:
            self['reversescale'] = _v
        _v = arg.pop('showscale', None)
        _v = showscale if showscale is not None else _v
        if _v is not None:
            self['showscale'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
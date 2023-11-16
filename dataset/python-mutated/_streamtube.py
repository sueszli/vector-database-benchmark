from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy

class Streamtube(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'streamtube'
    _valid_props = {'autocolorscale', 'cauto', 'cmax', 'cmid', 'cmin', 'coloraxis', 'colorbar', 'colorscale', 'customdata', 'customdatasrc', 'hoverinfo', 'hoverinfosrc', 'hoverlabel', 'hovertemplate', 'hovertemplatesrc', 'hovertext', 'ids', 'idssrc', 'legend', 'legendgroup', 'legendgrouptitle', 'legendrank', 'legendwidth', 'lighting', 'lightposition', 'maxdisplayed', 'meta', 'metasrc', 'name', 'opacity', 'reversescale', 'scene', 'showlegend', 'showscale', 'sizeref', 'starts', 'stream', 'text', 'type', 'u', 'uhoverformat', 'uid', 'uirevision', 'usrc', 'v', 'vhoverformat', 'visible', 'vsrc', 'w', 'whoverformat', 'wsrc', 'x', 'xhoverformat', 'xsrc', 'y', 'yhoverformat', 'ysrc', 'z', 'zhoverformat', 'zsrc'}

    @property
    def autocolorscale(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether the colorscale is a default palette\n        (`autocolorscale: true`) or the palette determined by\n        `colorscale`. In case `colorscale` is unspecified or\n        `autocolorscale` is true, the default palette will be chosen\n        according to whether numbers in the `color` array are all\n        positive, all negative or mixed.\n\n        The 'autocolorscale' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['autocolorscale']

    @autocolorscale.setter
    def autocolorscale(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['autocolorscale'] = val

    @property
    def cauto(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not the color domain is computed with\n        respect to the input data (here u/v/w norm) or the bounds set\n        in `cmin` and `cmax` Defaults to `false` when `cmin` and `cmax`\n        are set by the user.\n\n        The 'cauto' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['cauto']

    @cauto.setter
    def cauto(self, val):
        if False:
            return 10
        self['cauto'] = val

    @property
    def cmax(self):
        if False:
            return 10
        "\n        Sets the upper bound of the color domain. Value should have the\n        same units as u/v/w norm and if set, `cmin` must be set as\n        well.\n\n        The 'cmax' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['cmax']

    @cmax.setter
    def cmax(self, val):
        if False:
            i = 10
            return i + 15
        self['cmax'] = val

    @property
    def cmid(self):
        if False:
            print('Hello World!')
        "\n        Sets the mid-point of the color domain by scaling `cmin` and/or\n        `cmax` to be equidistant to this point. Value should have the\n        same units as u/v/w norm. Has no effect when `cauto` is\n        `false`.\n\n        The 'cmid' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['cmid']

    @cmid.setter
    def cmid(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['cmid'] = val

    @property
    def cmin(self):
        if False:
            while True:
                i = 10
        "\n        Sets the lower bound of the color domain. Value should have the\n        same units as u/v/w norm and if set, `cmax` must be set as\n        well.\n\n        The 'cmin' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['cmin']

    @cmin.setter
    def cmin(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['cmin'] = val

    @property
    def coloraxis(self):
        if False:
            return 10
        '\n        Sets a reference to a shared color axis. References to these\n        shared color axes are "coloraxis", "coloraxis2", "coloraxis3",\n        etc. Settings for these shared color axes are set in the\n        layout, under `layout.coloraxis`, `layout.coloraxis2`, etc.\n        Note that multiple color scales can be linked to the same color\n        axis.\n\n        The \'coloraxis\' property is an identifier of a particular\n        subplot, of type \'coloraxis\', that may be specified as the string \'coloraxis\'\n        optionally followed by an integer >= 1\n        (e.g. \'coloraxis\', \'coloraxis1\', \'coloraxis2\', \'coloraxis3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['coloraxis']

    @coloraxis.setter
    def coloraxis(self, val):
        if False:
            i = 10
            return i + 15
        self['coloraxis'] = val

    @property
    def colorbar(self):
        if False:
            return 10
        '\n        The \'colorbar\' property is an instance of ColorBar\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.streamtube.ColorBar`\n          - A dict of string/value properties that will be passed\n            to the ColorBar constructor\n\n            Supported dict properties:\n\n                bgcolor\n                    Sets the color of padded area.\n                bordercolor\n                    Sets the axis line color.\n                borderwidth\n                    Sets the width (in px) or the border enclosing\n                    this color bar.\n                dtick\n                    Sets the step in-between ticks on this axis.\n                    Use with `tick0`. Must be a positive number, or\n                    special strings available to "log" and "date"\n                    axes. If the axis `type` is "log", then ticks\n                    are set every 10^(n*dtick) where n is the tick\n                    number. For example, to set a tick mark at 1,\n                    10, 100, 1000, ... set dtick to 1. To set tick\n                    marks at 1, 100, 10000, ... set dtick to 2. To\n                    set tick marks at 1, 5, 25, 125, 625, 3125, ...\n                    set dtick to log_10(5), or 0.69897000433. "log"\n                    has several special values; "L<f>", where `f`\n                    is a positive number, gives ticks linearly\n                    spaced in value (but not position). For example\n                    `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n                    at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10\n                    plus small digits between, use "D1" (all\n                    digits) or "D2" (only 2 and 5). `tick0` is\n                    ignored for "D1" and "D2". If the axis `type`\n                    is "date", then you must convert the time to\n                    milliseconds. For example, to set the interval\n                    between ticks to one day, set `dtick` to\n                    86400000.0. "date" also has special values\n                    "M<n>" gives ticks spaced by a number of\n                    months. `n` must be a positive integer. To set\n                    ticks on the 15th of every third month, set\n                    `tick0` to "2000-01-15" and `dtick` to "M3". To\n                    set ticks every 4 years, set `dtick` to "M48"\n                exponentformat\n                    Determines a formatting rule for the tick\n                    exponents. For example, consider the number\n                    1,000,000,000. If "none", it appears as\n                    1,000,000,000. If "e", 1e+9. If "E", 1E+9. If\n                    "power", 1x10^9 (with 9 in a super script). If\n                    "SI", 1G. If "B", 1B.\n                labelalias\n                    Replacement text for specific tick or hover\n                    labels. For example using {US: \'USA\', CA:\n                    \'Canada\'} changes US to USA and CA to Canada.\n                    The labels we would have shown must match the\n                    keys exactly, after adding any tickprefix or\n                    ticksuffix. For negative numbers the minus sign\n                    symbol used (U+2212) is wider than the regular\n                    ascii dash. That means you need to use −1\n                    instead of -1. labelalias can be used with any\n                    axis type, and both keys (if needed) and values\n                    (if desired) can include html-like tags or\n                    MathJax.\n                len\n                    Sets the length of the color bar This measure\n                    excludes the padding of both ends. That is, the\n                    color bar length is this length minus the\n                    padding on both ends.\n                lenmode\n                    Determines whether this color bar\'s length\n                    (i.e. the measure in the color variation\n                    direction) is set in units of plot "fraction"\n                    or in *pixels. Use `len` to set the value.\n                minexponent\n                    Hide SI prefix for 10^n if |n| is below this\n                    number. This only has an effect when\n                    `tickformat` is "SI" or "B".\n                nticks\n                    Specifies the maximum number of ticks for the\n                    particular axis. The actual number of ticks\n                    will be chosen automatically to be less than or\n                    equal to `nticks`. Has an effect only if\n                    `tickmode` is set to "auto".\n                orientation\n                    Sets the orientation of the colorbar.\n                outlinecolor\n                    Sets the axis line color.\n                outlinewidth\n                    Sets the width (in px) of the axis line.\n                separatethousands\n                    If "true", even 4-digit integers are separated\n                showexponent\n                    If "all", all exponents are shown besides their\n                    significands. If "first", only the exponent of\n                    the first tick is shown. If "last", only the\n                    exponent of the last tick is shown. If "none",\n                    no exponents appear.\n                showticklabels\n                    Determines whether or not the tick labels are\n                    drawn.\n                showtickprefix\n                    If "all", all tick labels are displayed with a\n                    prefix. If "first", only the first tick is\n                    displayed with a prefix. If "last", only the\n                    last tick is displayed with a suffix. If\n                    "none", tick prefixes are hidden.\n                showticksuffix\n                    Same as `showtickprefix` but for tick suffixes.\n                thickness\n                    Sets the thickness of the color bar This\n                    measure excludes the size of the padding, ticks\n                    and labels.\n                thicknessmode\n                    Determines whether this color bar\'s thickness\n                    (i.e. the measure in the constant color\n                    direction) is set in units of plot "fraction"\n                    or in "pixels". Use `thickness` to set the\n                    value.\n                tick0\n                    Sets the placement of the first tick on this\n                    axis. Use with `dtick`. If the axis `type` is\n                    "log", then you must take the log of your\n                    starting tick (e.g. to set the starting tick to\n                    100, set the `tick0` to 2) except when\n                    `dtick`=*L<f>* (see `dtick` for more info). If\n                    the axis `type` is "date", it should be a date\n                    string, like date data. If the axis `type` is\n                    "category", it should be a number, using the\n                    scale where each category is assigned a serial\n                    number from zero in the order it appears.\n                tickangle\n                    Sets the angle of the tick labels with respect\n                    to the horizontal. For example, a `tickangle`\n                    of -90 draws the tick labels vertically.\n                tickcolor\n                    Sets the tick color.\n                tickfont\n                    Sets the color bar\'s tick label font\n                tickformat\n                    Sets the tick label formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format. And for dates see:\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format. We add two\n                    items to d3\'s date formatter: "%h" for half of\n                    the year as a decimal number as well as "%{n}f"\n                    for fractional seconds with n digits. For\n                    example, *2016-10-13 09:15:23.456* with\n                    tickformat "%H~%M~%S.%2f" would display\n                    "09~15~23.46"\n                tickformatstops\n                    A tuple of :class:`plotly.graph_objects.streamt\n                    ube.colorbar.Tickformatstop` instances or dicts\n                    with compatible properties\n                tickformatstopdefaults\n                    When used in a template (as layout.template.dat\n                    a.streamtube.colorbar.tickformatstopdefaults),\n                    sets the default property values to use for\n                    elements of streamtube.colorbar.tickformatstops\n                ticklabeloverflow\n                    Determines how we handle tick labels that would\n                    overflow either the graph div or the domain of\n                    the axis. The default value for inside tick\n                    labels is *hide past domain*. In other cases\n                    the default is *hide past div*.\n                ticklabelposition\n                    Determines where tick labels are drawn relative\n                    to the ticks. Left and right options are used\n                    when `orientation` is "h", top and bottom when\n                    `orientation` is "v".\n                ticklabelstep\n                    Sets the spacing between tick labels as\n                    compared to the spacing between ticks. A value\n                    of 1 (default) means each tick gets a label. A\n                    value of 2 means shows every 2nd label. A\n                    larger value n means only every nth tick is\n                    labeled. `tick0` determines which labels are\n                    shown. Not implemented for axes with `type`\n                    "log" or "multicategory", or when `tickmode` is\n                    "array".\n                ticklen\n                    Sets the tick length (in px).\n                tickmode\n                    Sets the tick mode for this axis. If "auto",\n                    the number of ticks is set via `nticks`. If\n                    "linear", the placement of the ticks is\n                    determined by a starting position `tick0` and a\n                    tick step `dtick` ("linear" is the default\n                    value if `tick0` and `dtick` are provided). If\n                    "array", the placement of the ticks is set via\n                    `tickvals` and the tick text is `ticktext`.\n                    ("array" is the default value if `tickvals` is\n                    provided).\n                tickprefix\n                    Sets a tick label prefix.\n                ticks\n                    Determines whether ticks are drawn or not. If\n                    "", this axis\' ticks are not drawn. If\n                    "outside" ("inside"), this axis\' are drawn\n                    outside (inside) the axis lines.\n                ticksuffix\n                    Sets a tick label suffix.\n                ticktext\n                    Sets the text displayed at the ticks position\n                    via `tickvals`. Only has an effect if\n                    `tickmode` is set to "array". Used with\n                    `tickvals`.\n                ticktextsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `ticktext`.\n                tickvals\n                    Sets the values at which ticks on this axis\n                    appear. Only has an effect if `tickmode` is set\n                    to "array". Used with `ticktext`.\n                tickvalssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `tickvals`.\n                tickwidth\n                    Sets the tick width (in px).\n                title\n                    :class:`plotly.graph_objects.streamtube.colorba\n                    r.Title` instance or dict with compatible\n                    properties\n                titlefont\n                    Deprecated: Please use\n                    streamtube.colorbar.title.font instead. Sets\n                    this color bar\'s title font. Note that the\n                    title\'s font used to be set by the now\n                    deprecated `titlefont` attribute.\n                titleside\n                    Deprecated: Please use\n                    streamtube.colorbar.title.side instead.\n                    Determines the location of color bar\'s title\n                    with respect to the color bar. Defaults to\n                    "top" when `orientation` if "v" and  defaults\n                    to "right" when `orientation` if "h". Note that\n                    the title\'s location used to be set by the now\n                    deprecated `titleside` attribute.\n                x\n                    Sets the x position with respect to `xref` of\n                    the color bar (in plot fraction). When `xref`\n                    is "paper", defaults to 1.02 when `orientation`\n                    is "v" and 0.5 when `orientation` is "h". When\n                    `xref` is "container", defaults to 1 when\n                    `orientation` is "v" and 0.5 when `orientation`\n                    is "h". Must be between 0 and 1 if `xref` is\n                    "container" and between "-2" and 3 if `xref` is\n                    "paper".\n                xanchor\n                    Sets this color bar\'s horizontal position\n                    anchor. This anchor binds the `x` position to\n                    the "left", "center" or "right" of the color\n                    bar. Defaults to "left" when `orientation` is\n                    "v" and "center" when `orientation` is "h".\n                xpad\n                    Sets the amount of padding (in px) along the x\n                    direction.\n                xref\n                    Sets the container `x` refers to. "container"\n                    spans the entire `width` of the plot. "paper"\n                    refers to the width of the plotting area only.\n                y\n                    Sets the y position with respect to `yref` of\n                    the color bar (in plot fraction). When `yref`\n                    is "paper", defaults to 0.5 when `orientation`\n                    is "v" and 1.02 when `orientation` is "h". When\n                    `yref` is "container", defaults to 0.5 when\n                    `orientation` is "v" and 1 when `orientation`\n                    is "h". Must be between 0 and 1 if `yref` is\n                    "container" and between "-2" and 3 if `yref` is\n                    "paper".\n                yanchor\n                    Sets this color bar\'s vertical position anchor\n                    This anchor binds the `y` position to the\n                    "top", "middle" or "bottom" of the color bar.\n                    Defaults to "middle" when `orientation` is "v"\n                    and "bottom" when `orientation` is "h".\n                ypad\n                    Sets the amount of padding (in px) along the y\n                    direction.\n                yref\n                    Sets the container `y` refers to. "container"\n                    spans the entire `height` of the plot. "paper"\n                    refers to the height of the plotting area only.\n\n        Returns\n        -------\n        plotly.graph_objs.streamtube.ColorBar\n        '
        return self['colorbar']

    @colorbar.setter
    def colorbar(self, val):
        if False:
            while True:
                i = 10
        self['colorbar'] = val

    @property
    def colorscale(self):
        if False:
            return 10
        "\n        Sets the colorscale. The colorscale must be an array containing\n        arrays mapping a normalized value to an rgb, rgba, hex, hsl,\n        hsv, or named color string. At minimum, a mapping for the\n        lowest (0) and highest (1) values are required. For example,\n        `[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]`. To control the\n        bounds of the colorscale in color space, use `cmin` and `cmax`.\n        Alternatively, `colorscale` may be a palette name string of the\n        following list: Blackbody,Bluered,Blues,Cividis,Earth,Electric,\n        Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,RdBu,Reds,Viridis,\n        YlGnBu,YlOrRd.\n\n        The 'colorscale' property is a colorscale and may be\n        specified as:\n          - A list of colors that will be spaced evenly to create the colorscale.\n            Many predefined colorscale lists are included in the sequential, diverging,\n            and cyclical modules in the plotly.colors package.\n          - A list of 2-element lists where the first element is the\n            normalized color level value (starting at 0 and ending at 1),\n            and the second item is a valid color string.\n            (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])\n          - One of the following named colorscales:\n                ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',\n                 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',\n                 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',\n                 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',\n                 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',\n                 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',\n                 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',\n                 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',\n                 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',\n                 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',\n                 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',\n                 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',\n                 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',\n                 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',\n                 'ylorrd'].\n            Appending '_r' to a named colorscale reverses it.\n\n        Returns\n        -------\n        str\n        "
        return self['colorscale']

    @colorscale.setter
    def colorscale(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['colorscale'] = val

    @property
    def customdata(self):
        if False:
            i = 10
            return i + 15
        '\n        Assigns extra data each datum. This may be useful when\n        listening to hover, click and selection events. Note that,\n        "scatter" traces also appends customdata items in the markers\n        DOM elements\n\n        The \'customdata\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['customdata']

    @customdata.setter
    def customdata(self, val):
        if False:
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
        self['customdatasrc'] = val

    @property
    def hoverinfo(self):
        if False:
            return 10
        "\n        Determines which trace information appear on hover. If `none`\n        or `skip` are set, no information is displayed upon hovering.\n        But, if `none` is set, click and hover events are still fired.\n\n        The 'hoverinfo' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['x', 'y', 'z', 'u', 'v', 'w', 'norm', 'divergence', 'text', 'name'] joined with '+' characters\n            (e.g. 'x+y')\n            OR exactly one of ['all', 'none', 'skip'] (e.g. 'skip')\n          - A list or array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['hoverinfo']

    @hoverinfo.setter
    def hoverinfo(self, val):
        if False:
            i = 10
            return i + 15
        self['hoverinfo'] = val

    @property
    def hoverinfosrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for\n        `hoverinfo`.\n\n        The 'hoverinfosrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['hoverinfosrc']

    @hoverinfosrc.setter
    def hoverinfosrc(self, val):
        if False:
            i = 10
            return i + 15
        self['hoverinfosrc'] = val

    @property
    def hoverlabel(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'hoverlabel' property is an instance of Hoverlabel\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.streamtube.Hoverlabel`\n          - A dict of string/value properties that will be passed\n            to the Hoverlabel constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the text\n                    content within hover label box. Has an effect\n                    only if the hover label text spans more two or\n                    more lines\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                bgcolor\n                    Sets the background color of the hover labels\n                    for this trace\n                bgcolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bgcolor`.\n                bordercolor\n                    Sets the border color of the hover labels for\n                    this trace.\n                bordercolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bordercolor`.\n                font\n                    Sets the font used in hover labels.\n                namelength\n                    Sets the default length (in number of\n                    characters) of the trace name in the hover\n                    labels for all traces. -1 shows the whole name\n                    regardless of length. 0-3 shows the first 0-3\n                    characters, and an integer >3 will show the\n                    whole name if it is less than that many\n                    characters, but if it is longer, will truncate\n                    to `namelength - 3` characters and add an\n                    ellipsis.\n                namelengthsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `namelength`.\n\n        Returns\n        -------\n        plotly.graph_objs.streamtube.Hoverlabel\n        "
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        if False:
            return 10
        self['hoverlabel'] = val

    @property
    def hovertemplate(self):
        if False:
            return 10
        '\n        Template string used for rendering the information that appear\n        on hover box. Note that this will override `hoverinfo`.\n        Variables are inserted using %{variable}, for example "y: %{y}"\n        as well as %{xother}, {%_xother}, {%_xother_}, {%xother_}. When\n        showing info for several points, "xother" will be added to\n        those with different x positions from the first point. An\n        underscore before or after "(x|y)other" will add a space on\n        that side, only when this field is shown. Numbers are formatted\n        using d3-format\'s syntax %{variable:d3-format}, for example\n        "Price: %{y:$.2f}".\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for\n        details on the formatting syntax. Dates are formatted using\n        d3-time-format\'s syntax %{variable|d3-time-format}, for example\n        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format for details on the date\n        formatting syntax. The variables available in `hovertemplate`\n        are the ones emitted as event data described at this link\n        https://plotly.com/javascript/plotlyjs-events/#event-data.\n        Additionally, every attributes that can be specified per-point\n        (the ones that are `arrayOk: true`) are available. Finally, the\n        template string has access to variables `tubex`, `tubey`,\n        `tubez`, `tubeu`, `tubev`, `tubew`, `norm` and `divergence`.\n        Anything contained in tag `<extra>` is displayed in the\n        secondary box, for example "<extra>{fullData.name}</extra>". To\n        hide the secondary box completely, use an empty tag\n        `<extra></extra>`.\n\n        The \'hovertemplate\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['hovertemplate']

    @hovertemplate.setter
    def hovertemplate(self, val):
        if False:
            return 10
        self['hovertemplate'] = val

    @property
    def hovertemplatesrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for\n        `hovertemplate`.\n\n        The 'hovertemplatesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['hovertemplatesrc']

    @hovertemplatesrc.setter
    def hovertemplatesrc(self, val):
        if False:
            print('Hello World!')
        self['hovertemplatesrc'] = val

    @property
    def hovertext(self):
        if False:
            i = 10
            return i + 15
        "\n        Same as `text`.\n\n        The 'hovertext' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['hovertext']

    @hovertext.setter
    def hovertext(self, val):
        if False:
            while True:
                i = 10
        self['hovertext'] = val

    @property
    def ids(self):
        if False:
            return 10
        "\n        Assigns id labels to each datum. These ids for object constancy\n        of data points during animation. Should be an array of strings,\n        not numbers or any other type.\n\n        The 'ids' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['ids']

    @ids.setter
    def ids(self, val):
        if False:
            return 10
        self['ids'] = val

    @property
    def idssrc(self):
        if False:
            return 10
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
            return 10
        '\n        Sets the reference to a legend to show this trace in.\n        References to these legends are "legend", "legend2", "legend3",\n        etc. Settings for these legends are set in the layout, under\n        `layout.legend`, `layout.legend2`, etc.\n\n        The \'legend\' property is an identifier of a particular\n        subplot, of type \'legend\', that may be specified as the string \'legend\'\n        optionally followed by an integer >= 1\n        (e.g. \'legend\', \'legend1\', \'legend2\', \'legend3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['legend']

    @legend.setter
    def legend(self, val):
        if False:
            return 10
        self['legend'] = val

    @property
    def legendgroup(self):
        if False:
            while True:
                i = 10
        "\n        Sets the legend group for this trace. Traces and shapes part of\n        the same legend group hide/show at the same time when toggling\n        legend items.\n\n        The 'legendgroup' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['legendgroup']

    @legendgroup.setter
    def legendgroup(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['legendgroup'] = val

    @property
    def legendgrouptitle(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'legendgrouptitle' property is an instance of Legendgrouptitle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.streamtube.Legendgrouptitle`\n          - A dict of string/value properties that will be passed\n            to the Legendgrouptitle constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend group's title font.\n                text\n                    Sets the title of the legend group.\n\n        Returns\n        -------\n        plotly.graph_objs.streamtube.Legendgrouptitle\n        "
        return self['legendgrouptitle']

    @legendgrouptitle.setter
    def legendgrouptitle(self, val):
        if False:
            while True:
                i = 10
        self['legendgrouptitle'] = val

    @property
    def legendrank(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the legend rank for this trace. Items and groups with\n        smaller ranks are presented on top/left side while with\n        "reversed" `legend.traceorder` they are on bottom/right side.\n        The default legendrank is 1000, so that you can use ranks less\n        than 1000 to place certain items before all unranked items, and\n        ranks greater than 1000 to go after all unranked items. When\n        having unranked or equal rank items shapes would be displayed\n        after traces i.e. according to their order in data and layout.\n\n        The \'legendrank\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['legendrank']

    @legendrank.setter
    def legendrank(self, val):
        if False:
            print('Hello World!')
        self['legendrank'] = val

    @property
    def legendwidth(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the width (in px or fraction) of the legend for this\n        trace.\n\n        The 'legendwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['legendwidth']

    @legendwidth.setter
    def legendwidth(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['legendwidth'] = val

    @property
    def lighting(self):
        if False:
            return 10
        "\n        The 'lighting' property is an instance of Lighting\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.streamtube.Lighting`\n          - A dict of string/value properties that will be passed\n            to the Lighting constructor\n\n            Supported dict properties:\n\n                ambient\n                    Ambient light increases overall color\n                    visibility but can wash out the image.\n                diffuse\n                    Represents the extent that incident rays are\n                    reflected in a range of angles.\n                facenormalsepsilon\n                    Epsilon for face normals calculation avoids\n                    math issues arising from degenerate geometry.\n                fresnel\n                    Represents the reflectance as a dependency of\n                    the viewing angle; e.g. paper is reflective\n                    when viewing it from the edge of the paper\n                    (almost 90 degrees), causing shine.\n                roughness\n                    Alters specular reflection; the rougher the\n                    surface, the wider and less contrasty the\n                    shine.\n                specular\n                    Represents the level that incident rays are\n                    reflected in a single direction, causing shine.\n                vertexnormalsepsilon\n                    Epsilon for vertex normals calculation avoids\n                    math issues arising from degenerate geometry.\n\n        Returns\n        -------\n        plotly.graph_objs.streamtube.Lighting\n        "
        return self['lighting']

    @lighting.setter
    def lighting(self, val):
        if False:
            while True:
                i = 10
        self['lighting'] = val

    @property
    def lightposition(self):
        if False:
            while True:
                i = 10
        "\n        The 'lightposition' property is an instance of Lightposition\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.streamtube.Lightposition`\n          - A dict of string/value properties that will be passed\n            to the Lightposition constructor\n\n            Supported dict properties:\n\n                x\n                    Numeric vector, representing the X coordinate\n                    for each vertex.\n                y\n                    Numeric vector, representing the Y coordinate\n                    for each vertex.\n                z\n                    Numeric vector, representing the Z coordinate\n                    for each vertex.\n\n        Returns\n        -------\n        plotly.graph_objs.streamtube.Lightposition\n        "
        return self['lightposition']

    @lightposition.setter
    def lightposition(self, val):
        if False:
            while True:
                i = 10
        self['lightposition'] = val

    @property
    def maxdisplayed(self):
        if False:
            return 10
        "\n        The maximum number of displayed segments in a streamtube.\n\n        The 'maxdisplayed' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['maxdisplayed']

    @maxdisplayed.setter
    def maxdisplayed(self, val):
        if False:
            return 10
        self['maxdisplayed'] = val

    @property
    def meta(self):
        if False:
            return 10
        "\n        Assigns extra meta information associated with this trace that\n        can be used in various text attributes. Attributes such as\n        trace `name`, graph, axis and colorbar `title.text`, annotation\n        `text` `rangeselector`, `updatemenues` and `sliders` `label`\n        text all support `meta`. To access the trace `meta` values in\n        an attribute in the same trace, simply use `%{meta[i]}` where\n        `i` is the index or key of the `meta` item in question. To\n        access trace `meta` in layout attributes, use\n        `%{data[n[.meta[i]}` where `i` is the index or key of the\n        `meta` and `n` is the trace index.\n\n        The 'meta' property accepts values of any type\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['meta']

    @meta.setter
    def meta(self, val):
        if False:
            return 10
        self['meta'] = val

    @property
    def metasrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `meta`.\n\n        The 'metasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['metasrc']

    @metasrc.setter
    def metasrc(self, val):
        if False:
            while True:
                i = 10
        self['metasrc'] = val

    @property
    def name(self):
        if False:
            print('Hello World!')
        "\n        Sets the trace name. The trace name appears as the legend item\n        and on hover.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            return 10
        self['name'] = val

    @property
    def opacity(self):
        if False:
            return 10
        "\n        Sets the opacity of the surface. Please note that in the case\n        of using high `opacity` values for example a value greater than\n        or equal to 0.5 on two surfaces (and 0.25 with four surfaces),\n        an overlay of multiple transparent surfaces may not perfectly\n        be sorted in depth by the webgl API. This behavior may be\n        improved in the near future and is subject to change.\n\n        The 'opacity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            return 10
        self['opacity'] = val

    @property
    def reversescale(self):
        if False:
            return 10
        "\n        Reverses the color mapping if true. If true, `cmin` will\n        correspond to the last color in the array and `cmax` will\n        correspond to the first color.\n\n        The 'reversescale' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['reversescale']

    @reversescale.setter
    def reversescale(self, val):
        if False:
            return 10
        self['reversescale'] = val

    @property
    def scene(self):
        if False:
            while True:
                i = 10
        '\n        Sets a reference between this trace\'s 3D coordinate system and\n        a 3D scene. If "scene" (the default value), the (x,y,z)\n        coordinates refer to `layout.scene`. If "scene2", the (x,y,z)\n        coordinates refer to `layout.scene2`, and so on.\n\n        The \'scene\' property is an identifier of a particular\n        subplot, of type \'scene\', that may be specified as the string \'scene\'\n        optionally followed by an integer >= 1\n        (e.g. \'scene\', \'scene1\', \'scene2\', \'scene3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['scene']

    @scene.setter
    def scene(self, val):
        if False:
            print('Hello World!')
        self['scene'] = val

    @property
    def showlegend(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not an item corresponding to this trace\n        is shown in the legend.\n\n        The 'showlegend' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showlegend']

    @showlegend.setter
    def showlegend(self, val):
        if False:
            print('Hello World!')
        self['showlegend'] = val

    @property
    def showscale(self):
        if False:
            return 10
        "\n        Determines whether or not a colorbar is displayed for this\n        trace.\n\n        The 'showscale' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showscale']

    @showscale.setter
    def showscale(self, val):
        if False:
            i = 10
            return i + 15
        self['showscale'] = val

    @property
    def sizeref(self):
        if False:
            while True:
                i = 10
        "\n        The scaling factor for the streamtubes. The default is 1, which\n        avoids two max divergence tubes from touching at adjacent\n        starting positions.\n\n        The 'sizeref' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['sizeref']

    @sizeref.setter
    def sizeref(self, val):
        if False:
            print('Hello World!')
        self['sizeref'] = val

    @property
    def starts(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'starts' property is an instance of Starts\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.streamtube.Starts`\n          - A dict of string/value properties that will be passed\n            to the Starts constructor\n\n            Supported dict properties:\n\n                x\n                    Sets the x components of the starting position\n                    of the streamtubes\n                xsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `x`.\n                y\n                    Sets the y components of the starting position\n                    of the streamtubes\n                ysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `y`.\n                z\n                    Sets the z components of the starting position\n                    of the streamtubes\n                zsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `z`.\n\n        Returns\n        -------\n        plotly.graph_objs.streamtube.Starts\n        "
        return self['starts']

    @starts.setter
    def starts(self, val):
        if False:
            while True:
                i = 10
        self['starts'] = val

    @property
    def stream(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'stream' property is an instance of Stream\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.streamtube.Stream`\n          - A dict of string/value properties that will be passed\n            to the Stream constructor\n\n            Supported dict properties:\n\n                maxpoints\n                    Sets the maximum number of points to keep on\n                    the plots from an incoming stream. If\n                    `maxpoints` is set to 50, only the newest 50\n                    points will be displayed on the plot.\n                token\n                    The stream id number links a data trace on a\n                    plot with a stream. See https://chart-\n                    studio.plotly.com/settings for more details.\n\n        Returns\n        -------\n        plotly.graph_objs.streamtube.Stream\n        "
        return self['stream']

    @stream.setter
    def stream(self, val):
        if False:
            while True:
                i = 10
        self['stream'] = val

    @property
    def text(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets a text element associated with this trace. If trace\n        `hoverinfo` contains a "text" flag, this text element will be\n        seen in all hover labels. Note that streamtube traces do not\n        support array `text` values.\n\n        The \'text\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['text']

    @text.setter
    def text(self, val):
        if False:
            return 10
        self['text'] = val

    @property
    def u(self):
        if False:
            while True:
                i = 10
        "\n        Sets the x components of the vector field.\n\n        The 'u' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['u']

    @u.setter
    def u(self, val):
        if False:
            return 10
        self['u'] = val

    @property
    def uhoverformat(self):
        if False:
            print('Hello World!')
        "\n        Sets the hover text formatting rulefor `u`  using d3 formatting\n        mini-languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format.By\n        default the values are formatted using generic number format.\n\n        The 'uhoverformat' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['uhoverformat']

    @uhoverformat.setter
    def uhoverformat(self, val):
        if False:
            print('Hello World!')
        self['uhoverformat'] = val

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
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        self['uirevision'] = val

    @property
    def usrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `u`.\n\n        The 'usrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['usrc']

    @usrc.setter
    def usrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['usrc'] = val

    @property
    def v(self):
        if False:
            while True:
                i = 10
        "\n        Sets the y components of the vector field.\n\n        The 'v' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['v']

    @v.setter
    def v(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['v'] = val

    @property
    def vhoverformat(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the hover text formatting rulefor `v`  using d3 formatting\n        mini-languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format.By\n        default the values are formatted using generic number format.\n\n        The 'vhoverformat' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['vhoverformat']

    @vhoverformat.setter
    def vhoverformat(self, val):
        if False:
            print('Hello World!')
        self['vhoverformat'] = val

    @property
    def visible(self):
        if False:
            while True:
                i = 10
        '\n        Determines whether or not this trace is visible. If\n        "legendonly", the trace is not drawn, but can appear as a\n        legend item (provided that the legend itself is visible).\n\n        The \'visible\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, \'legendonly\']\n\n        Returns\n        -------\n        Any\n        '
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            print('Hello World!')
        self['visible'] = val

    @property
    def vsrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `v`.\n\n        The 'vsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['vsrc']

    @vsrc.setter
    def vsrc(self, val):
        if False:
            print('Hello World!')
        self['vsrc'] = val

    @property
    def w(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the z components of the vector field.\n\n        The 'w' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['w']

    @w.setter
    def w(self, val):
        if False:
            i = 10
            return i + 15
        self['w'] = val

    @property
    def whoverformat(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the hover text formatting rulefor `w`  using d3 formatting\n        mini-languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format.By\n        default the values are formatted using generic number format.\n\n        The 'whoverformat' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['whoverformat']

    @whoverformat.setter
    def whoverformat(self, val):
        if False:
            i = 10
            return i + 15
        self['whoverformat'] = val

    @property
    def wsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `w`.\n\n        The 'wsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['wsrc']

    @wsrc.setter
    def wsrc(self, val):
        if False:
            while True:
                i = 10
        self['wsrc'] = val

    @property
    def x(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the x coordinates of the vector field.\n\n        The 'x' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['x'] = val

    @property
    def xhoverformat(self):
        if False:
            return 10
        '\n        Sets the hover text formatting rulefor `x`  using d3 formatting\n        mini-languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display *09~15~23.46*By default the values\n        are formatted using `xaxis.hoverformat`.\n\n        The \'xhoverformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['xhoverformat']

    @xhoverformat.setter
    def xhoverformat(self, val):
        if False:
            i = 10
            return i + 15
        self['xhoverformat'] = val

    @property
    def xsrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `x`.\n\n        The 'xsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['xsrc']

    @xsrc.setter
    def xsrc(self, val):
        if False:
            print('Hello World!')
        self['xsrc'] = val

    @property
    def y(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the y coordinates of the vector field.\n\n        The 'y' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            i = 10
            return i + 15
        self['y'] = val

    @property
    def yhoverformat(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the hover text formatting rulefor `y`  using d3 formatting\n        mini-languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display *09~15~23.46*By default the values\n        are formatted using `yaxis.hoverformat`.\n\n        The \'yhoverformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['yhoverformat']

    @yhoverformat.setter
    def yhoverformat(self, val):
        if False:
            while True:
                i = 10
        self['yhoverformat'] = val

    @property
    def ysrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `y`.\n\n        The 'ysrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['ysrc']

    @ysrc.setter
    def ysrc(self, val):
        if False:
            print('Hello World!')
        self['ysrc'] = val

    @property
    def z(self):
        if False:
            return 10
        "\n        Sets the z coordinates of the vector field.\n\n        The 'z' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['z']

    @z.setter
    def z(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['z'] = val

    @property
    def zhoverformat(self):
        if False:
            while True:
                i = 10
        '\n        Sets the hover text formatting rulefor `z`  using d3 formatting\n        mini-languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display *09~15~23.46*By default the values\n        are formatted using `zaxis.hoverformat`.\n\n        The \'zhoverformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['zhoverformat']

    @zhoverformat.setter
    def zhoverformat(self, val):
        if False:
            while True:
                i = 10
        self['zhoverformat'] = val

    @property
    def zsrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `z`.\n\n        The 'zsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['zsrc']

    @zsrc.setter
    def zsrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['zsrc'] = val

    @property
    def type(self):
        if False:
            print('Hello World!')
        return self._props['type']

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        autocolorscale\n            Determines whether the colorscale is a default palette\n            (`autocolorscale: true`) or the palette determined by\n            `colorscale`. In case `colorscale` is unspecified or\n            `autocolorscale` is true, the default palette will be\n            chosen according to whether numbers in the `color`\n            array are all positive, all negative or mixed.\n        cauto\n            Determines whether or not the color domain is computed\n            with respect to the input data (here u/v/w norm) or the\n            bounds set in `cmin` and `cmax` Defaults to `false`\n            when `cmin` and `cmax` are set by the user.\n        cmax\n            Sets the upper bound of the color domain. Value should\n            have the same units as u/v/w norm and if set, `cmin`\n            must be set as well.\n        cmid\n            Sets the mid-point of the color domain by scaling\n            `cmin` and/or `cmax` to be equidistant to this point.\n            Value should have the same units as u/v/w norm. Has no\n            effect when `cauto` is `false`.\n        cmin\n            Sets the lower bound of the color domain. Value should\n            have the same units as u/v/w norm and if set, `cmax`\n            must be set as well.\n        coloraxis\n            Sets a reference to a shared color axis. References to\n            these shared color axes are "coloraxis", "coloraxis2",\n            "coloraxis3", etc. Settings for these shared color axes\n            are set in the layout, under `layout.coloraxis`,\n            `layout.coloraxis2`, etc. Note that multiple color\n            scales can be linked to the same color axis.\n        colorbar\n            :class:`plotly.graph_objects.streamtube.ColorBar`\n            instance or dict with compatible properties\n        colorscale\n            Sets the colorscale. The colorscale must be an array\n            containing arrays mapping a normalized value to an rgb,\n            rgba, hex, hsl, hsv, or named color string. At minimum,\n            a mapping for the lowest (0) and highest (1) values are\n            required. For example, `[[0, \'rgb(0,0,255)\'], [1,\n            \'rgb(255,0,0)\']]`. To control the bounds of the\n            colorscale in color space, use `cmin` and `cmax`.\n            Alternatively, `colorscale` may be a palette name\n            string of the following list: Blackbody,Bluered,Blues,C\n            ividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portl\n            and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.streamtube.Hoverlabel`\n            instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `tubex`, `tubey`, `tubez`, `tubeu`,\n            `tubev`, `tubew`, `norm` and `divergence`. Anything\n            contained in tag `<extra>` is displayed in the\n            secondary box, for example\n            "<extra>{fullData.name}</extra>". To hide the secondary\n            box completely, use an empty tag `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Same as `text`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.streamtube.Legendgrouptitl\n            e` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        lighting\n            :class:`plotly.graph_objects.streamtube.Lighting`\n            instance or dict with compatible properties\n        lightposition\n            :class:`plotly.graph_objects.streamtube.Lightposition`\n            instance or dict with compatible properties\n        maxdisplayed\n            The maximum number of displayed segments in a\n            streamtube.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        opacity\n            Sets the opacity of the surface. Please note that in\n            the case of using high `opacity` values for example a\n            value greater than or equal to 0.5 on two surfaces (and\n            0.25 with four surfaces), an overlay of multiple\n            transparent surfaces may not perfectly be sorted in\n            depth by the webgl API. This behavior may be improved\n            in the near future and is subject to change.\n        reversescale\n            Reverses the color mapping if true. If true, `cmin`\n            will correspond to the last color in the array and\n            `cmax` will correspond to the first color.\n        scene\n            Sets a reference between this trace\'s 3D coordinate\n            system and a 3D scene. If "scene" (the default value),\n            the (x,y,z) coordinates refer to `layout.scene`. If\n            "scene2", the (x,y,z) coordinates refer to\n            `layout.scene2`, and so on.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        showscale\n            Determines whether or not a colorbar is displayed for\n            this trace.\n        sizeref\n            The scaling factor for the streamtubes. The default is\n            1, which avoids two max divergence tubes from touching\n            at adjacent starting positions.\n        starts\n            :class:`plotly.graph_objects.streamtube.Starts`\n            instance or dict with compatible properties\n        stream\n            :class:`plotly.graph_objects.streamtube.Stream`\n            instance or dict with compatible properties\n        text\n            Sets a text element associated with this trace. If\n            trace `hoverinfo` contains a "text" flag, this text\n            element will be seen in all hover labels. Note that\n            streamtube traces do not support array `text` values.\n        u\n            Sets the x components of the vector field.\n        uhoverformat\n            Sets the hover text formatting rulefor `u`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see: https://github.com/d\n            3/d3-format/tree/v1.4.5#d3-format.By default the values\n            are formatted using generic number format.\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        usrc\n            Sets the source reference on Chart Studio Cloud for\n            `u`.\n        v\n            Sets the y components of the vector field.\n        vhoverformat\n            Sets the hover text formatting rulefor `v`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see: https://github.com/d\n            3/d3-format/tree/v1.4.5#d3-format.By default the values\n            are formatted using generic number format.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        vsrc\n            Sets the source reference on Chart Studio Cloud for\n            `v`.\n        w\n            Sets the z components of the vector field.\n        whoverformat\n            Sets the hover text formatting rulefor `w`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see: https://github.com/d\n            3/d3-format/tree/v1.4.5#d3-format.By default the values\n            are formatted using generic number format.\n        wsrc\n            Sets the source reference on Chart Studio Cloud for\n            `w`.\n        x\n            Sets the x coordinates of the vector field.\n        xhoverformat\n            Sets the hover text formatting rulefor `x`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `xaxis.hoverformat`.\n        xsrc\n            Sets the source reference on Chart Studio Cloud for\n            `x`.\n        y\n            Sets the y coordinates of the vector field.\n        yhoverformat\n            Sets the hover text formatting rulefor `y`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `yaxis.hoverformat`.\n        ysrc\n            Sets the source reference on Chart Studio Cloud for\n            `y`.\n        z\n            Sets the z coordinates of the vector field.\n        zhoverformat\n            Sets the hover text formatting rulefor `z`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `zaxis.hoverformat`.\n        zsrc\n            Sets the source reference on Chart Studio Cloud for\n            `z`.\n        '

    def __init__(self, arg=None, autocolorscale=None, cauto=None, cmax=None, cmid=None, cmin=None, coloraxis=None, colorbar=None, colorscale=None, customdata=None, customdatasrc=None, hoverinfo=None, hoverinfosrc=None, hoverlabel=None, hovertemplate=None, hovertemplatesrc=None, hovertext=None, ids=None, idssrc=None, legend=None, legendgroup=None, legendgrouptitle=None, legendrank=None, legendwidth=None, lighting=None, lightposition=None, maxdisplayed=None, meta=None, metasrc=None, name=None, opacity=None, reversescale=None, scene=None, showlegend=None, showscale=None, sizeref=None, starts=None, stream=None, text=None, u=None, uhoverformat=None, uid=None, uirevision=None, usrc=None, v=None, vhoverformat=None, visible=None, vsrc=None, w=None, whoverformat=None, wsrc=None, x=None, xhoverformat=None, xsrc=None, y=None, yhoverformat=None, ysrc=None, z=None, zhoverformat=None, zsrc=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Streamtube object\n\n        Use a streamtube trace to visualize flow in a vector field.\n        Specify a vector field using 6 1D arrays of equal length, 3\n        position arrays `x`, `y` and `z` and 3 vector component arrays\n        `u`, `v`, and `w`.  By default, the tubes\' starting positions\n        will be cut from the vector field\'s x-z plane at its minimum y\n        value. To specify your own starting position, use attributes\n        `starts.x`, `starts.y` and `starts.z`. The color is encoded by\n        the norm of (u, v, w), and the local radius by the divergence\n        of (u, v, w).\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.Streamtube`\n        autocolorscale\n            Determines whether the colorscale is a default palette\n            (`autocolorscale: true`) or the palette determined by\n            `colorscale`. In case `colorscale` is unspecified or\n            `autocolorscale` is true, the default palette will be\n            chosen according to whether numbers in the `color`\n            array are all positive, all negative or mixed.\n        cauto\n            Determines whether or not the color domain is computed\n            with respect to the input data (here u/v/w norm) or the\n            bounds set in `cmin` and `cmax` Defaults to `false`\n            when `cmin` and `cmax` are set by the user.\n        cmax\n            Sets the upper bound of the color domain. Value should\n            have the same units as u/v/w norm and if set, `cmin`\n            must be set as well.\n        cmid\n            Sets the mid-point of the color domain by scaling\n            `cmin` and/or `cmax` to be equidistant to this point.\n            Value should have the same units as u/v/w norm. Has no\n            effect when `cauto` is `false`.\n        cmin\n            Sets the lower bound of the color domain. Value should\n            have the same units as u/v/w norm and if set, `cmax`\n            must be set as well.\n        coloraxis\n            Sets a reference to a shared color axis. References to\n            these shared color axes are "coloraxis", "coloraxis2",\n            "coloraxis3", etc. Settings for these shared color axes\n            are set in the layout, under `layout.coloraxis`,\n            `layout.coloraxis2`, etc. Note that multiple color\n            scales can be linked to the same color axis.\n        colorbar\n            :class:`plotly.graph_objects.streamtube.ColorBar`\n            instance or dict with compatible properties\n        colorscale\n            Sets the colorscale. The colorscale must be an array\n            containing arrays mapping a normalized value to an rgb,\n            rgba, hex, hsl, hsv, or named color string. At minimum,\n            a mapping for the lowest (0) and highest (1) values are\n            required. For example, `[[0, \'rgb(0,0,255)\'], [1,\n            \'rgb(255,0,0)\']]`. To control the bounds of the\n            colorscale in color space, use `cmin` and `cmax`.\n            Alternatively, `colorscale` may be a palette name\n            string of the following list: Blackbody,Bluered,Blues,C\n            ividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portl\n            and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.streamtube.Hoverlabel`\n            instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `tubex`, `tubey`, `tubez`, `tubeu`,\n            `tubev`, `tubew`, `norm` and `divergence`. Anything\n            contained in tag `<extra>` is displayed in the\n            secondary box, for example\n            "<extra>{fullData.name}</extra>". To hide the secondary\n            box completely, use an empty tag `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Same as `text`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.streamtube.Legendgrouptitl\n            e` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        lighting\n            :class:`plotly.graph_objects.streamtube.Lighting`\n            instance or dict with compatible properties\n        lightposition\n            :class:`plotly.graph_objects.streamtube.Lightposition`\n            instance or dict with compatible properties\n        maxdisplayed\n            The maximum number of displayed segments in a\n            streamtube.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        opacity\n            Sets the opacity of the surface. Please note that in\n            the case of using high `opacity` values for example a\n            value greater than or equal to 0.5 on two surfaces (and\n            0.25 with four surfaces), an overlay of multiple\n            transparent surfaces may not perfectly be sorted in\n            depth by the webgl API. This behavior may be improved\n            in the near future and is subject to change.\n        reversescale\n            Reverses the color mapping if true. If true, `cmin`\n            will correspond to the last color in the array and\n            `cmax` will correspond to the first color.\n        scene\n            Sets a reference between this trace\'s 3D coordinate\n            system and a 3D scene. If "scene" (the default value),\n            the (x,y,z) coordinates refer to `layout.scene`. If\n            "scene2", the (x,y,z) coordinates refer to\n            `layout.scene2`, and so on.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        showscale\n            Determines whether or not a colorbar is displayed for\n            this trace.\n        sizeref\n            The scaling factor for the streamtubes. The default is\n            1, which avoids two max divergence tubes from touching\n            at adjacent starting positions.\n        starts\n            :class:`plotly.graph_objects.streamtube.Starts`\n            instance or dict with compatible properties\n        stream\n            :class:`plotly.graph_objects.streamtube.Stream`\n            instance or dict with compatible properties\n        text\n            Sets a text element associated with this trace. If\n            trace `hoverinfo` contains a "text" flag, this text\n            element will be seen in all hover labels. Note that\n            streamtube traces do not support array `text` values.\n        u\n            Sets the x components of the vector field.\n        uhoverformat\n            Sets the hover text formatting rulefor `u`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see: https://github.com/d\n            3/d3-format/tree/v1.4.5#d3-format.By default the values\n            are formatted using generic number format.\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        usrc\n            Sets the source reference on Chart Studio Cloud for\n            `u`.\n        v\n            Sets the y components of the vector field.\n        vhoverformat\n            Sets the hover text formatting rulefor `v`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see: https://github.com/d\n            3/d3-format/tree/v1.4.5#d3-format.By default the values\n            are formatted using generic number format.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        vsrc\n            Sets the source reference on Chart Studio Cloud for\n            `v`.\n        w\n            Sets the z components of the vector field.\n        whoverformat\n            Sets the hover text formatting rulefor `w`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see: https://github.com/d\n            3/d3-format/tree/v1.4.5#d3-format.By default the values\n            are formatted using generic number format.\n        wsrc\n            Sets the source reference on Chart Studio Cloud for\n            `w`.\n        x\n            Sets the x coordinates of the vector field.\n        xhoverformat\n            Sets the hover text formatting rulefor `x`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `xaxis.hoverformat`.\n        xsrc\n            Sets the source reference on Chart Studio Cloud for\n            `x`.\n        y\n            Sets the y coordinates of the vector field.\n        yhoverformat\n            Sets the hover text formatting rulefor `y`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `yaxis.hoverformat`.\n        ysrc\n            Sets the source reference on Chart Studio Cloud for\n            `y`.\n        z\n            Sets the z coordinates of the vector field.\n        zhoverformat\n            Sets the hover text formatting rulefor `z`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `zaxis.hoverformat`.\n        zsrc\n            Sets the source reference on Chart Studio Cloud for\n            `z`.\n\n        Returns\n        -------\n        Streamtube\n        '
        super(Streamtube, self).__init__('streamtube')
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
            raise ValueError('The first argument to the plotly.graph_objs.Streamtube\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Streamtube`')
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
        _v = arg.pop('coloraxis', None)
        _v = coloraxis if coloraxis is not None else _v
        if _v is not None:
            self['coloraxis'] = _v
        _v = arg.pop('colorbar', None)
        _v = colorbar if colorbar is not None else _v
        if _v is not None:
            self['colorbar'] = _v
        _v = arg.pop('colorscale', None)
        _v = colorscale if colorscale is not None else _v
        if _v is not None:
            self['colorscale'] = _v
        _v = arg.pop('customdata', None)
        _v = customdata if customdata is not None else _v
        if _v is not None:
            self['customdata'] = _v
        _v = arg.pop('customdatasrc', None)
        _v = customdatasrc if customdatasrc is not None else _v
        if _v is not None:
            self['customdatasrc'] = _v
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
        _v = arg.pop('lighting', None)
        _v = lighting if lighting is not None else _v
        if _v is not None:
            self['lighting'] = _v
        _v = arg.pop('lightposition', None)
        _v = lightposition if lightposition is not None else _v
        if _v is not None:
            self['lightposition'] = _v
        _v = arg.pop('maxdisplayed', None)
        _v = maxdisplayed if maxdisplayed is not None else _v
        if _v is not None:
            self['maxdisplayed'] = _v
        _v = arg.pop('meta', None)
        _v = meta if meta is not None else _v
        if _v is not None:
            self['meta'] = _v
        _v = arg.pop('metasrc', None)
        _v = metasrc if metasrc is not None else _v
        if _v is not None:
            self['metasrc'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        _v = arg.pop('reversescale', None)
        _v = reversescale if reversescale is not None else _v
        if _v is not None:
            self['reversescale'] = _v
        _v = arg.pop('scene', None)
        _v = scene if scene is not None else _v
        if _v is not None:
            self['scene'] = _v
        _v = arg.pop('showlegend', None)
        _v = showlegend if showlegend is not None else _v
        if _v is not None:
            self['showlegend'] = _v
        _v = arg.pop('showscale', None)
        _v = showscale if showscale is not None else _v
        if _v is not None:
            self['showscale'] = _v
        _v = arg.pop('sizeref', None)
        _v = sizeref if sizeref is not None else _v
        if _v is not None:
            self['sizeref'] = _v
        _v = arg.pop('starts', None)
        _v = starts if starts is not None else _v
        if _v is not None:
            self['starts'] = _v
        _v = arg.pop('stream', None)
        _v = stream if stream is not None else _v
        if _v is not None:
            self['stream'] = _v
        _v = arg.pop('text', None)
        _v = text if text is not None else _v
        if _v is not None:
            self['text'] = _v
        _v = arg.pop('u', None)
        _v = u if u is not None else _v
        if _v is not None:
            self['u'] = _v
        _v = arg.pop('uhoverformat', None)
        _v = uhoverformat if uhoverformat is not None else _v
        if _v is not None:
            self['uhoverformat'] = _v
        _v = arg.pop('uid', None)
        _v = uid if uid is not None else _v
        if _v is not None:
            self['uid'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('usrc', None)
        _v = usrc if usrc is not None else _v
        if _v is not None:
            self['usrc'] = _v
        _v = arg.pop('v', None)
        _v = v if v is not None else _v
        if _v is not None:
            self['v'] = _v
        _v = arg.pop('vhoverformat', None)
        _v = vhoverformat if vhoverformat is not None else _v
        if _v is not None:
            self['vhoverformat'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        _v = arg.pop('vsrc', None)
        _v = vsrc if vsrc is not None else _v
        if _v is not None:
            self['vsrc'] = _v
        _v = arg.pop('w', None)
        _v = w if w is not None else _v
        if _v is not None:
            self['w'] = _v
        _v = arg.pop('whoverformat', None)
        _v = whoverformat if whoverformat is not None else _v
        if _v is not None:
            self['whoverformat'] = _v
        _v = arg.pop('wsrc', None)
        _v = wsrc if wsrc is not None else _v
        if _v is not None:
            self['wsrc'] = _v
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('xhoverformat', None)
        _v = xhoverformat if xhoverformat is not None else _v
        if _v is not None:
            self['xhoverformat'] = _v
        _v = arg.pop('xsrc', None)
        _v = xsrc if xsrc is not None else _v
        if _v is not None:
            self['xsrc'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('yhoverformat', None)
        _v = yhoverformat if yhoverformat is not None else _v
        if _v is not None:
            self['yhoverformat'] = _v
        _v = arg.pop('ysrc', None)
        _v = ysrc if ysrc is not None else _v
        if _v is not None:
            self['ysrc'] = _v
        _v = arg.pop('z', None)
        _v = z if z is not None else _v
        if _v is not None:
            self['z'] = _v
        _v = arg.pop('zhoverformat', None)
        _v = zhoverformat if zhoverformat is not None else _v
        if _v is not None:
            self['zhoverformat'] = _v
        _v = arg.pop('zsrc', None)
        _v = zsrc if zsrc is not None else _v
        if _v is not None:
            self['zsrc'] = _v
        self._props['type'] = 'streamtube'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
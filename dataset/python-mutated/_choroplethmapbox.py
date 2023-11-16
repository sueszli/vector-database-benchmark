from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy

class Choroplethmapbox(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'choroplethmapbox'
    _valid_props = {'autocolorscale', 'below', 'coloraxis', 'colorbar', 'colorscale', 'customdata', 'customdatasrc', 'featureidkey', 'geojson', 'hoverinfo', 'hoverinfosrc', 'hoverlabel', 'hovertemplate', 'hovertemplatesrc', 'hovertext', 'hovertextsrc', 'ids', 'idssrc', 'legend', 'legendgroup', 'legendgrouptitle', 'legendrank', 'legendwidth', 'locations', 'locationssrc', 'marker', 'meta', 'metasrc', 'name', 'reversescale', 'selected', 'selectedpoints', 'showlegend', 'showscale', 'stream', 'subplot', 'text', 'textsrc', 'type', 'uid', 'uirevision', 'unselected', 'visible', 'z', 'zauto', 'zmax', 'zmid', 'zmin', 'zsrc'}

    @property
    def autocolorscale(self):
        if False:
            print('Hello World!')
        "\n        Determines whether the colorscale is a default palette\n        (`autocolorscale: true`) or the palette determined by\n        `colorscale`. In case `colorscale` is unspecified or\n        `autocolorscale` is true, the default palette will be chosen\n        according to whether numbers in the `color` array are all\n        positive, all negative or mixed.\n\n        The 'autocolorscale' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['autocolorscale']

    @autocolorscale.setter
    def autocolorscale(self, val):
        if False:
            i = 10
            return i + 15
        self['autocolorscale'] = val

    @property
    def below(self):
        if False:
            return 10
        "\n        Determines if the choropleth polygons will be inserted before\n        the layer with the specified ID. By default, choroplethmapbox\n        traces are placed above the water layers. If set to '', the\n        layer will be inserted above every existing layer.\n\n        The 'below' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['below']

    @below.setter
    def below(self, val):
        if False:
            while True:
                i = 10
        self['below'] = val

    @property
    def coloraxis(self):
        if False:
            print('Hello World!')
        '\n        Sets a reference to a shared color axis. References to these\n        shared color axes are "coloraxis", "coloraxis2", "coloraxis3",\n        etc. Settings for these shared color axes are set in the\n        layout, under `layout.coloraxis`, `layout.coloraxis2`, etc.\n        Note that multiple color scales can be linked to the same color\n        axis.\n\n        The \'coloraxis\' property is an identifier of a particular\n        subplot, of type \'coloraxis\', that may be specified as the string \'coloraxis\'\n        optionally followed by an integer >= 1\n        (e.g. \'coloraxis\', \'coloraxis1\', \'coloraxis2\', \'coloraxis3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['coloraxis']

    @coloraxis.setter
    def coloraxis(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['coloraxis'] = val

    @property
    def colorbar(self):
        if False:
            i = 10
            return i + 15
        '\n        The \'colorbar\' property is an instance of ColorBar\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.choroplethmapbox.ColorBar`\n          - A dict of string/value properties that will be passed\n            to the ColorBar constructor\n\n            Supported dict properties:\n\n                bgcolor\n                    Sets the color of padded area.\n                bordercolor\n                    Sets the axis line color.\n                borderwidth\n                    Sets the width (in px) or the border enclosing\n                    this color bar.\n                dtick\n                    Sets the step in-between ticks on this axis.\n                    Use with `tick0`. Must be a positive number, or\n                    special strings available to "log" and "date"\n                    axes. If the axis `type` is "log", then ticks\n                    are set every 10^(n*dtick) where n is the tick\n                    number. For example, to set a tick mark at 1,\n                    10, 100, 1000, ... set dtick to 1. To set tick\n                    marks at 1, 100, 10000, ... set dtick to 2. To\n                    set tick marks at 1, 5, 25, 125, 625, 3125, ...\n                    set dtick to log_10(5), or 0.69897000433. "log"\n                    has several special values; "L<f>", where `f`\n                    is a positive number, gives ticks linearly\n                    spaced in value (but not position). For example\n                    `tick0` = 0.1, `dtick` = "L0.5" will put ticks\n                    at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10\n                    plus small digits between, use "D1" (all\n                    digits) or "D2" (only 2 and 5). `tick0` is\n                    ignored for "D1" and "D2". If the axis `type`\n                    is "date", then you must convert the time to\n                    milliseconds. For example, to set the interval\n                    between ticks to one day, set `dtick` to\n                    86400000.0. "date" also has special values\n                    "M<n>" gives ticks spaced by a number of\n                    months. `n` must be a positive integer. To set\n                    ticks on the 15th of every third month, set\n                    `tick0` to "2000-01-15" and `dtick` to "M3". To\n                    set ticks every 4 years, set `dtick` to "M48"\n                exponentformat\n                    Determines a formatting rule for the tick\n                    exponents. For example, consider the number\n                    1,000,000,000. If "none", it appears as\n                    1,000,000,000. If "e", 1e+9. If "E", 1E+9. If\n                    "power", 1x10^9 (with 9 in a super script). If\n                    "SI", 1G. If "B", 1B.\n                labelalias\n                    Replacement text for specific tick or hover\n                    labels. For example using {US: \'USA\', CA:\n                    \'Canada\'} changes US to USA and CA to Canada.\n                    The labels we would have shown must match the\n                    keys exactly, after adding any tickprefix or\n                    ticksuffix. For negative numbers the minus sign\n                    symbol used (U+2212) is wider than the regular\n                    ascii dash. That means you need to use −1\n                    instead of -1. labelalias can be used with any\n                    axis type, and both keys (if needed) and values\n                    (if desired) can include html-like tags or\n                    MathJax.\n                len\n                    Sets the length of the color bar This measure\n                    excludes the padding of both ends. That is, the\n                    color bar length is this length minus the\n                    padding on both ends.\n                lenmode\n                    Determines whether this color bar\'s length\n                    (i.e. the measure in the color variation\n                    direction) is set in units of plot "fraction"\n                    or in *pixels. Use `len` to set the value.\n                minexponent\n                    Hide SI prefix for 10^n if |n| is below this\n                    number. This only has an effect when\n                    `tickformat` is "SI" or "B".\n                nticks\n                    Specifies the maximum number of ticks for the\n                    particular axis. The actual number of ticks\n                    will be chosen automatically to be less than or\n                    equal to `nticks`. Has an effect only if\n                    `tickmode` is set to "auto".\n                orientation\n                    Sets the orientation of the colorbar.\n                outlinecolor\n                    Sets the axis line color.\n                outlinewidth\n                    Sets the width (in px) of the axis line.\n                separatethousands\n                    If "true", even 4-digit integers are separated\n                showexponent\n                    If "all", all exponents are shown besides their\n                    significands. If "first", only the exponent of\n                    the first tick is shown. If "last", only the\n                    exponent of the last tick is shown. If "none",\n                    no exponents appear.\n                showticklabels\n                    Determines whether or not the tick labels are\n                    drawn.\n                showtickprefix\n                    If "all", all tick labels are displayed with a\n                    prefix. If "first", only the first tick is\n                    displayed with a prefix. If "last", only the\n                    last tick is displayed with a suffix. If\n                    "none", tick prefixes are hidden.\n                showticksuffix\n                    Same as `showtickprefix` but for tick suffixes.\n                thickness\n                    Sets the thickness of the color bar This\n                    measure excludes the size of the padding, ticks\n                    and labels.\n                thicknessmode\n                    Determines whether this color bar\'s thickness\n                    (i.e. the measure in the constant color\n                    direction) is set in units of plot "fraction"\n                    or in "pixels". Use `thickness` to set the\n                    value.\n                tick0\n                    Sets the placement of the first tick on this\n                    axis. Use with `dtick`. If the axis `type` is\n                    "log", then you must take the log of your\n                    starting tick (e.g. to set the starting tick to\n                    100, set the `tick0` to 2) except when\n                    `dtick`=*L<f>* (see `dtick` for more info). If\n                    the axis `type` is "date", it should be a date\n                    string, like date data. If the axis `type` is\n                    "category", it should be a number, using the\n                    scale where each category is assigned a serial\n                    number from zero in the order it appears.\n                tickangle\n                    Sets the angle of the tick labels with respect\n                    to the horizontal. For example, a `tickangle`\n                    of -90 draws the tick labels vertically.\n                tickcolor\n                    Sets the tick color.\n                tickfont\n                    Sets the color bar\'s tick label font\n                tickformat\n                    Sets the tick label formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format. And for dates see:\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format. We add two\n                    items to d3\'s date formatter: "%h" for half of\n                    the year as a decimal number as well as "%{n}f"\n                    for fractional seconds with n digits. For\n                    example, *2016-10-13 09:15:23.456* with\n                    tickformat "%H~%M~%S.%2f" would display\n                    "09~15~23.46"\n                tickformatstops\n                    A tuple of :class:`plotly.graph_objects.choropl\n                    ethmapbox.colorbar.Tickformatstop` instances or\n                    dicts with compatible properties\n                tickformatstopdefaults\n                    When used in a template (as layout.template.dat\n                    a.choroplethmapbox.colorbar.tickformatstopdefau\n                    lts), sets the default property values to use\n                    for elements of\n                    choroplethmapbox.colorbar.tickformatstops\n                ticklabeloverflow\n                    Determines how we handle tick labels that would\n                    overflow either the graph div or the domain of\n                    the axis. The default value for inside tick\n                    labels is *hide past domain*. In other cases\n                    the default is *hide past div*.\n                ticklabelposition\n                    Determines where tick labels are drawn relative\n                    to the ticks. Left and right options are used\n                    when `orientation` is "h", top and bottom when\n                    `orientation` is "v".\n                ticklabelstep\n                    Sets the spacing between tick labels as\n                    compared to the spacing between ticks. A value\n                    of 1 (default) means each tick gets a label. A\n                    value of 2 means shows every 2nd label. A\n                    larger value n means only every nth tick is\n                    labeled. `tick0` determines which labels are\n                    shown. Not implemented for axes with `type`\n                    "log" or "multicategory", or when `tickmode` is\n                    "array".\n                ticklen\n                    Sets the tick length (in px).\n                tickmode\n                    Sets the tick mode for this axis. If "auto",\n                    the number of ticks is set via `nticks`. If\n                    "linear", the placement of the ticks is\n                    determined by a starting position `tick0` and a\n                    tick step `dtick` ("linear" is the default\n                    value if `tick0` and `dtick` are provided). If\n                    "array", the placement of the ticks is set via\n                    `tickvals` and the tick text is `ticktext`.\n                    ("array" is the default value if `tickvals` is\n                    provided).\n                tickprefix\n                    Sets a tick label prefix.\n                ticks\n                    Determines whether ticks are drawn or not. If\n                    "", this axis\' ticks are not drawn. If\n                    "outside" ("inside"), this axis\' are drawn\n                    outside (inside) the axis lines.\n                ticksuffix\n                    Sets a tick label suffix.\n                ticktext\n                    Sets the text displayed at the ticks position\n                    via `tickvals`. Only has an effect if\n                    `tickmode` is set to "array". Used with\n                    `tickvals`.\n                ticktextsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `ticktext`.\n                tickvals\n                    Sets the values at which ticks on this axis\n                    appear. Only has an effect if `tickmode` is set\n                    to "array". Used with `ticktext`.\n                tickvalssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `tickvals`.\n                tickwidth\n                    Sets the tick width (in px).\n                title\n                    :class:`plotly.graph_objects.choroplethmapbox.c\n                    olorbar.Title` instance or dict with compatible\n                    properties\n                titlefont\n                    Deprecated: Please use\n                    choroplethmapbox.colorbar.title.font instead.\n                    Sets this color bar\'s title font. Note that the\n                    title\'s font used to be set by the now\n                    deprecated `titlefont` attribute.\n                titleside\n                    Deprecated: Please use\n                    choroplethmapbox.colorbar.title.side instead.\n                    Determines the location of color bar\'s title\n                    with respect to the color bar. Defaults to\n                    "top" when `orientation` if "v" and  defaults\n                    to "right" when `orientation` if "h". Note that\n                    the title\'s location used to be set by the now\n                    deprecated `titleside` attribute.\n                x\n                    Sets the x position with respect to `xref` of\n                    the color bar (in plot fraction). When `xref`\n                    is "paper", defaults to 1.02 when `orientation`\n                    is "v" and 0.5 when `orientation` is "h". When\n                    `xref` is "container", defaults to 1 when\n                    `orientation` is "v" and 0.5 when `orientation`\n                    is "h". Must be between 0 and 1 if `xref` is\n                    "container" and between "-2" and 3 if `xref` is\n                    "paper".\n                xanchor\n                    Sets this color bar\'s horizontal position\n                    anchor. This anchor binds the `x` position to\n                    the "left", "center" or "right" of the color\n                    bar. Defaults to "left" when `orientation` is\n                    "v" and "center" when `orientation` is "h".\n                xpad\n                    Sets the amount of padding (in px) along the x\n                    direction.\n                xref\n                    Sets the container `x` refers to. "container"\n                    spans the entire `width` of the plot. "paper"\n                    refers to the width of the plotting area only.\n                y\n                    Sets the y position with respect to `yref` of\n                    the color bar (in plot fraction). When `yref`\n                    is "paper", defaults to 0.5 when `orientation`\n                    is "v" and 1.02 when `orientation` is "h". When\n                    `yref` is "container", defaults to 0.5 when\n                    `orientation` is "v" and 1 when `orientation`\n                    is "h". Must be between 0 and 1 if `yref` is\n                    "container" and between "-2" and 3 if `yref` is\n                    "paper".\n                yanchor\n                    Sets this color bar\'s vertical position anchor\n                    This anchor binds the `y` position to the\n                    "top", "middle" or "bottom" of the color bar.\n                    Defaults to "middle" when `orientation` is "v"\n                    and "bottom" when `orientation` is "h".\n                ypad\n                    Sets the amount of padding (in px) along the y\n                    direction.\n                yref\n                    Sets the container `y` refers to. "container"\n                    spans the entire `height` of the plot. "paper"\n                    refers to the height of the plotting area only.\n\n        Returns\n        -------\n        plotly.graph_objs.choroplethmapbox.ColorBar\n        '
        return self['colorbar']

    @colorbar.setter
    def colorbar(self, val):
        if False:
            print('Hello World!')
        self['colorbar'] = val

    @property
    def colorscale(self):
        if False:
            print('Hello World!')
        "\n        Sets the colorscale. The colorscale must be an array containing\n        arrays mapping a normalized value to an rgb, rgba, hex, hsl,\n        hsv, or named color string. At minimum, a mapping for the\n        lowest (0) and highest (1) values are required. For example,\n        `[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]`. To control the\n        bounds of the colorscale in color space, use `zmin` and `zmax`.\n        Alternatively, `colorscale` may be a palette name string of the\n        following list: Blackbody,Bluered,Blues,Cividis,Earth,Electric,\n        Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,RdBu,Reds,Viridis,\n        YlGnBu,YlOrRd.\n\n        The 'colorscale' property is a colorscale and may be\n        specified as:\n          - A list of colors that will be spaced evenly to create the colorscale.\n            Many predefined colorscale lists are included in the sequential, diverging,\n            and cyclical modules in the plotly.colors package.\n          - A list of 2-element lists where the first element is the\n            normalized color level value (starting at 0 and ending at 1),\n            and the second item is a valid color string.\n            (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])\n          - One of the following named colorscales:\n                ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',\n                 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',\n                 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',\n                 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',\n                 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',\n                 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',\n                 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',\n                 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',\n                 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',\n                 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',\n                 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',\n                 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',\n                 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',\n                 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',\n                 'ylorrd'].\n            Appending '_r' to a named colorscale reverses it.\n\n        Returns\n        -------\n        str\n        "
        return self['colorscale']

    @colorscale.setter
    def colorscale(self, val):
        if False:
            return 10
        self['colorscale'] = val

    @property
    def customdata(self):
        if False:
            while True:
                i = 10
        '\n        Assigns extra data each datum. This may be useful when\n        listening to hover, click and selection events. Note that,\n        "scatter" traces also appends customdata items in the markers\n        DOM elements\n\n        The \'customdata\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['customdata']

    @customdata.setter
    def customdata(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['customdata'] = val

    @property
    def customdatasrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for\n        `customdata`.\n\n        The 'customdatasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['customdatasrc']

    @customdatasrc.setter
    def customdatasrc(self, val):
        if False:
            return 10
        self['customdatasrc'] = val

    @property
    def featureidkey(self):
        if False:
            return 10
        '\n        Sets the key in GeoJSON features which is used as id to match\n        the items included in the `locations` array. Support nested\n        property, for example "properties.name".\n\n        The \'featureidkey\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['featureidkey']

    @featureidkey.setter
    def featureidkey(self, val):
        if False:
            print('Hello World!')
        self['featureidkey'] = val

    @property
    def geojson(self):
        if False:
            while True:
                i = 10
        '\n        Sets the GeoJSON data associated with this trace. It can be set\n        as a valid GeoJSON object or as a URL string. Note that we only\n        accept GeoJSONs of type "FeatureCollection" or "Feature" with\n        geometries of type "Polygon" or "MultiPolygon".\n\n        The \'geojson\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['geojson']

    @geojson.setter
    def geojson(self, val):
        if False:
            return 10
        self['geojson'] = val

    @property
    def hoverinfo(self):
        if False:
            return 10
        "\n        Determines which trace information appear on hover. If `none`\n        or `skip` are set, no information is displayed upon hovering.\n        But, if `none` is set, click and hover events are still fired.\n\n        The 'hoverinfo' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['location', 'z', 'text', 'name'] joined with '+' characters\n            (e.g. 'location+z')\n            OR exactly one of ['all', 'none', 'skip'] (e.g. 'skip')\n          - A list or array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
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
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `hoverinfo`.\n\n        The 'hoverinfosrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['hoverinfosrc']

    @hoverinfosrc.setter
    def hoverinfosrc(self, val):
        if False:
            print('Hello World!')
        self['hoverinfosrc'] = val

    @property
    def hoverlabel(self):
        if False:
            return 10
        "\n        The 'hoverlabel' property is an instance of Hoverlabel\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.choroplethmapbox.Hoverlabel`\n          - A dict of string/value properties that will be passed\n            to the Hoverlabel constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the text\n                    content within hover label box. Has an effect\n                    only if the hover label text spans more two or\n                    more lines\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                bgcolor\n                    Sets the background color of the hover labels\n                    for this trace\n                bgcolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bgcolor`.\n                bordercolor\n                    Sets the border color of the hover labels for\n                    this trace.\n                bordercolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bordercolor`.\n                font\n                    Sets the font used in hover labels.\n                namelength\n                    Sets the default length (in number of\n                    characters) of the trace name in the hover\n                    labels for all traces. -1 shows the whole name\n                    regardless of length. 0-3 shows the first 0-3\n                    characters, and an integer >3 will show the\n                    whole name if it is less than that many\n                    characters, but if it is longer, will truncate\n                    to `namelength - 3` characters and add an\n                    ellipsis.\n                namelengthsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `namelength`.\n\n        Returns\n        -------\n        plotly.graph_objs.choroplethmapbox.Hoverlabel\n        "
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        if False:
            while True:
                i = 10
        self['hoverlabel'] = val

    @property
    def hovertemplate(self):
        if False:
            i = 10
            return i + 15
        '\n        Template string used for rendering the information that appear\n        on hover box. Note that this will override `hoverinfo`.\n        Variables are inserted using %{variable}, for example "y: %{y}"\n        as well as %{xother}, {%_xother}, {%_xother_}, {%xother_}. When\n        showing info for several points, "xother" will be added to\n        those with different x positions from the first point. An\n        underscore before or after "(x|y)other" will add a space on\n        that side, only when this field is shown. Numbers are formatted\n        using d3-format\'s syntax %{variable:d3-format}, for example\n        "Price: %{y:$.2f}".\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for\n        details on the formatting syntax. Dates are formatted using\n        d3-time-format\'s syntax %{variable|d3-time-format}, for example\n        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format for details on the date\n        formatting syntax. The variables available in `hovertemplate`\n        are the ones emitted as event data described at this link\n        https://plotly.com/javascript/plotlyjs-events/#event-data.\n        Additionally, every attributes that can be specified per-point\n        (the ones that are `arrayOk: true`) are available. Finally, the\n        template string has access to variable `properties` Anything\n        contained in tag `<extra>` is displayed in the secondary box,\n        for example "<extra>{fullData.name}</extra>". To hide the\n        secondary box completely, use an empty tag `<extra></extra>`.\n\n        The \'hovertemplate\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['hovertemplate']

    @hovertemplate.setter
    def hovertemplate(self, val):
        if False:
            return 10
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
            while True:
                i = 10
        self['hovertemplatesrc'] = val

    @property
    def hovertext(self):
        if False:
            while True:
                i = 10
        "\n        Same as `text`.\n\n        The 'hovertext' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        "
        return self['hovertext']

    @hovertext.setter
    def hovertext(self, val):
        if False:
            print('Hello World!')
        self['hovertext'] = val

    @property
    def hovertextsrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for\n        `hovertext`.\n\n        The 'hovertextsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['hovertextsrc']

    @hovertextsrc.setter
    def hovertextsrc(self, val):
        if False:
            print('Hello World!')
        self['hovertextsrc'] = val

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
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `ids`.\n\n        The 'idssrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['idssrc']

    @idssrc.setter
    def idssrc(self, val):
        if False:
            print('Hello World!')
        self['idssrc'] = val

    @property
    def legend(self):
        if False:
            print('Hello World!')
        '\n        Sets the reference to a legend to show this trace in.\n        References to these legends are "legend", "legend2", "legend3",\n        etc. Settings for these legends are set in the layout, under\n        `layout.legend`, `layout.legend2`, etc.\n\n        The \'legend\' property is an identifier of a particular\n        subplot, of type \'legend\', that may be specified as the string \'legend\'\n        optionally followed by an integer >= 1\n        (e.g. \'legend\', \'legend1\', \'legend2\', \'legend3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['legend']

    @legend.setter
    def legend(self, val):
        if False:
            while True:
                i = 10
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
            while True:
                i = 10
        self['legendgroup'] = val

    @property
    def legendgrouptitle(self):
        if False:
            while True:
                i = 10
        "\n        The 'legendgrouptitle' property is an instance of Legendgrouptitle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.choroplethmapbox.Legendgrouptitle`\n          - A dict of string/value properties that will be passed\n            to the Legendgrouptitle constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend group's title font.\n                text\n                    Sets the title of the legend group.\n\n        Returns\n        -------\n        plotly.graph_objs.choroplethmapbox.Legendgrouptitle\n        "
        return self['legendgrouptitle']

    @legendgrouptitle.setter
    def legendgrouptitle(self, val):
        if False:
            print('Hello World!')
        self['legendgrouptitle'] = val

    @property
    def legendrank(self):
        if False:
            while True:
                i = 10
        '\n        Sets the legend rank for this trace. Items and groups with\n        smaller ranks are presented on top/left side while with\n        "reversed" `legend.traceorder` they are on bottom/right side.\n        The default legendrank is 1000, so that you can use ranks less\n        than 1000 to place certain items before all unranked items, and\n        ranks greater than 1000 to go after all unranked items. When\n        having unranked or equal rank items shapes would be displayed\n        after traces i.e. according to their order in data and layout.\n\n        The \'legendrank\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['legendrank']

    @legendrank.setter
    def legendrank(self, val):
        if False:
            i = 10
            return i + 15
        self['legendrank'] = val

    @property
    def legendwidth(self):
        if False:
            return 10
        "\n        Sets the width (in px or fraction) of the legend for this\n        trace.\n\n        The 'legendwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['legendwidth']

    @legendwidth.setter
    def legendwidth(self, val):
        if False:
            while True:
                i = 10
        self['legendwidth'] = val

    @property
    def locations(self):
        if False:
            while True:
                i = 10
        '\n        Sets which features found in "geojson" to plot using their\n        feature `id` field.\n\n        The \'locations\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['locations']

    @locations.setter
    def locations(self, val):
        if False:
            return 10
        self['locations'] = val

    @property
    def locationssrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for\n        `locations`.\n\n        The 'locationssrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['locationssrc']

    @locationssrc.setter
    def locationssrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['locationssrc'] = val

    @property
    def marker(self):
        if False:
            print('Hello World!')
        "\n        The 'marker' property is an instance of Marker\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.choroplethmapbox.Marker`\n          - A dict of string/value properties that will be passed\n            to the Marker constructor\n\n            Supported dict properties:\n\n                line\n                    :class:`plotly.graph_objects.choroplethmapbox.m\n                    arker.Line` instance or dict with compatible\n                    properties\n                opacity\n                    Sets the opacity of the locations.\n                opacitysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `opacity`.\n\n        Returns\n        -------\n        plotly.graph_objs.choroplethmapbox.Marker\n        "
        return self['marker']

    @marker.setter
    def marker(self, val):
        if False:
            return 10
        self['marker'] = val

    @property
    def meta(self):
        if False:
            while True:
                i = 10
        "\n        Assigns extra meta information associated with this trace that\n        can be used in various text attributes. Attributes such as\n        trace `name`, graph, axis and colorbar `title.text`, annotation\n        `text` `rangeselector`, `updatemenues` and `sliders` `label`\n        text all support `meta`. To access the trace `meta` values in\n        an attribute in the same trace, simply use `%{meta[i]}` where\n        `i` is the index or key of the `meta` item in question. To\n        access trace `meta` in layout attributes, use\n        `%{data[n[.meta[i]}` where `i` is the index or key of the\n        `meta` and `n` is the trace index.\n\n        The 'meta' property accepts values of any type\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['meta']

    @meta.setter
    def meta(self, val):
        if False:
            print('Hello World!')
        self['meta'] = val

    @property
    def metasrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `meta`.\n\n        The 'metasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['metasrc']

    @metasrc.setter
    def metasrc(self, val):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
        self['name'] = val

    @property
    def reversescale(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Reverses the color mapping if true. If true, `zmin` will\n        correspond to the last color in the array and `zmax` will\n        correspond to the first color.\n\n        The 'reversescale' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['reversescale']

    @reversescale.setter
    def reversescale(self, val):
        if False:
            return 10
        self['reversescale'] = val

    @property
    def selected(self):
        if False:
            return 10
        "\n        The 'selected' property is an instance of Selected\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.choroplethmapbox.Selected`\n          - A dict of string/value properties that will be passed\n            to the Selected constructor\n\n            Supported dict properties:\n\n                marker\n                    :class:`plotly.graph_objects.choroplethmapbox.s\n                    elected.Marker` instance or dict with\n                    compatible properties\n\n        Returns\n        -------\n        plotly.graph_objs.choroplethmapbox.Selected\n        "
        return self['selected']

    @selected.setter
    def selected(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['selected'] = val

    @property
    def selectedpoints(self):
        if False:
            while True:
                i = 10
        "\n        Array containing integer indices of selected points. Has an\n        effect only for traces that support selections. Note that an\n        empty array means an empty selection where the `unselected` are\n        turned on for all points, whereas, any other non-array values\n        means no selection all where the `selected` and `unselected`\n        styles have no effect.\n\n        The 'selectedpoints' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['selectedpoints']

    @selectedpoints.setter
    def selectedpoints(self, val):
        if False:
            print('Hello World!')
        self['selectedpoints'] = val

    @property
    def showlegend(self):
        if False:
            return 10
        "\n        Determines whether or not an item corresponding to this trace\n        is shown in the legend.\n\n        The 'showlegend' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showlegend']

    @showlegend.setter
    def showlegend(self, val):
        if False:
            while True:
                i = 10
        self['showlegend'] = val

    @property
    def showscale(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether or not a colorbar is displayed for this\n        trace.\n\n        The 'showscale' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showscale']

    @showscale.setter
    def showscale(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['showscale'] = val

    @property
    def stream(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'stream' property is an instance of Stream\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.choroplethmapbox.Stream`\n          - A dict of string/value properties that will be passed\n            to the Stream constructor\n\n            Supported dict properties:\n\n                maxpoints\n                    Sets the maximum number of points to keep on\n                    the plots from an incoming stream. If\n                    `maxpoints` is set to 50, only the newest 50\n                    points will be displayed on the plot.\n                token\n                    The stream id number links a data trace on a\n                    plot with a stream. See https://chart-\n                    studio.plotly.com/settings for more details.\n\n        Returns\n        -------\n        plotly.graph_objs.choroplethmapbox.Stream\n        "
        return self['stream']

    @stream.setter
    def stream(self, val):
        if False:
            i = 10
            return i + 15
        self['stream'] = val

    @property
    def subplot(self):
        if False:
            print('Hello World!')
        '\n        Sets a reference between this trace\'s data coordinates and a\n        mapbox subplot. If "mapbox" (the default value), the data refer\n        to `layout.mapbox`. If "mapbox2", the data refer to\n        `layout.mapbox2`, and so on.\n\n        The \'subplot\' property is an identifier of a particular\n        subplot, of type \'mapbox\', that may be specified as the string \'mapbox\'\n        optionally followed by an integer >= 1\n        (e.g. \'mapbox\', \'mapbox1\', \'mapbox2\', \'mapbox3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['subplot']

    @subplot.setter
    def subplot(self, val):
        if False:
            i = 10
            return i + 15
        self['subplot'] = val

    @property
    def text(self):
        if False:
            while True:
                i = 10
        "\n        Sets the text elements associated with each location.\n\n        The 'text' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        "
        return self['text']

    @text.setter
    def text(self, val):
        if False:
            while True:
                i = 10
        self['text'] = val

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
            for i in range(10):
                print('nop')
        self['textsrc'] = val

    @property
    def uid(self):
        if False:
            return 10
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
            print('Hello World!')
        "\n        Controls persistence of some user-driven changes to the trace:\n        `constraintrange` in `parcoords` traces, as well as some\n        `editable: true` modifications such as `name` and\n        `colorbar.title`. Defaults to `layout.uirevision`. Note that\n        other user-driven trace attribute changes are controlled by\n        `layout` attributes: `trace.visible` is controlled by\n        `layout.legend.uirevision`, `selectedpoints` is controlled by\n        `layout.selectionrevision`, and `colorbar.(x|y)` (accessible\n        with `config: {editable: true}`) is controlled by\n        `layout.editrevision`. Trace changes are tracked by `uid`,\n        which only falls back on trace index if no `uid` is provided.\n        So if your app can add/remove traces before the end of the\n        `data` array, such that the same trace has a different index,\n        you can still preserve user-driven changes if you give each\n        trace a `uid` that stays with it as it moves.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            while True:
                i = 10
        self['uirevision'] = val

    @property
    def unselected(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'unselected' property is an instance of Unselected\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.choroplethmapbox.Unselected`\n          - A dict of string/value properties that will be passed\n            to the Unselected constructor\n\n            Supported dict properties:\n\n                marker\n                    :class:`plotly.graph_objects.choroplethmapbox.u\n                    nselected.Marker` instance or dict with\n                    compatible properties\n\n        Returns\n        -------\n        plotly.graph_objs.choroplethmapbox.Unselected\n        "
        return self['unselected']

    @unselected.setter
    def unselected(self, val):
        if False:
            while True:
                i = 10
        self['unselected'] = val

    @property
    def visible(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines whether or not this trace is visible. If\n        "legendonly", the trace is not drawn, but can appear as a\n        legend item (provided that the legend itself is visible).\n\n        The \'visible\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, \'legendonly\']\n\n        Returns\n        -------\n        Any\n        '
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            print('Hello World!')
        self['visible'] = val

    @property
    def z(self):
        if False:
            while True:
                i = 10
        "\n        Sets the color values.\n\n        The 'z' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['z']

    @z.setter
    def z(self, val):
        if False:
            return 10
        self['z'] = val

    @property
    def zauto(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether or not the color domain is computed with\n        respect to the input data (here in `z`) or the bounds set in\n        `zmin` and `zmax` Defaults to `false` when `zmin` and `zmax`\n        are set by the user.\n\n        The 'zauto' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['zauto']

    @zauto.setter
    def zauto(self, val):
        if False:
            return 10
        self['zauto'] = val

    @property
    def zmax(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the upper bound of the color domain. Value should have the\n        same units as in `z` and if set, `zmin` must be set as well.\n\n        The 'zmax' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['zmax']

    @zmax.setter
    def zmax(self, val):
        if False:
            while True:
                i = 10
        self['zmax'] = val

    @property
    def zmid(self):
        if False:
            return 10
        "\n        Sets the mid-point of the color domain by scaling `zmin` and/or\n        `zmax` to be equidistant to this point. Value should have the\n        same units as in `z`. Has no effect when `zauto` is `false`.\n\n        The 'zmid' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['zmid']

    @zmid.setter
    def zmid(self, val):
        if False:
            i = 10
            return i + 15
        self['zmid'] = val

    @property
    def zmin(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the lower bound of the color domain. Value should have the\n        same units as in `z` and if set, `zmax` must be set as well.\n\n        The 'zmin' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['zmin']

    @zmin.setter
    def zmin(self, val):
        if False:
            i = 10
            return i + 15
        self['zmin'] = val

    @property
    def zsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `z`.\n\n        The 'zsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['zsrc']

    @zsrc.setter
    def zsrc(self, val):
        if False:
            while True:
                i = 10
        self['zsrc'] = val

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        return self._props['type']

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        autocolorscale\n            Determines whether the colorscale is a default palette\n            (`autocolorscale: true`) or the palette determined by\n            `colorscale`. In case `colorscale` is unspecified or\n            `autocolorscale` is true, the default palette will be\n            chosen according to whether numbers in the `color`\n            array are all positive, all negative or mixed.\n        below\n            Determines if the choropleth polygons will be inserted\n            before the layer with the specified ID. By default,\n            choroplethmapbox traces are placed above the water\n            layers. If set to \'\', the layer will be inserted above\n            every existing layer.\n        coloraxis\n            Sets a reference to a shared color axis. References to\n            these shared color axes are "coloraxis", "coloraxis2",\n            "coloraxis3", etc. Settings for these shared color axes\n            are set in the layout, under `layout.coloraxis`,\n            `layout.coloraxis2`, etc. Note that multiple color\n            scales can be linked to the same color axis.\n        colorbar\n            :class:`plotly.graph_objects.choroplethmapbox.ColorBar`\n            instance or dict with compatible properties\n        colorscale\n            Sets the colorscale. The colorscale must be an array\n            containing arrays mapping a normalized value to an rgb,\n            rgba, hex, hsl, hsv, or named color string. At minimum,\n            a mapping for the lowest (0) and highest (1) values are\n            required. For example, `[[0, \'rgb(0,0,255)\'], [1,\n            \'rgb(255,0,0)\']]`. To control the bounds of the\n            colorscale in color space, use `zmin` and `zmax`.\n            Alternatively, `colorscale` may be a palette name\n            string of the following list: Blackbody,Bluered,Blues,C\n            ividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portl\n            and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        featureidkey\n            Sets the key in GeoJSON features which is used as id to\n            match the items included in the `locations` array.\n            Support nested property, for example "properties.name".\n        geojson\n            Sets the GeoJSON data associated with this trace. It\n            can be set as a valid GeoJSON object or as a URL\n            string. Note that we only accept GeoJSONs of type\n            "FeatureCollection" or "Feature" with geometries of\n            type "Polygon" or "MultiPolygon".\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.choroplethmapbox.Hoverlabe\n            l` instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variable `properties` Anything contained in tag\n            `<extra>` is displayed in the secondary box, for\n            example "<extra>{fullData.name}</extra>". To hide the\n            secondary box completely, use an empty tag\n            `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Same as `text`.\n        hovertextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertext`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.choroplethmapbox.Legendgro\n            uptitle` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        locations\n            Sets which features found in "geojson" to plot using\n            their feature `id` field.\n        locationssrc\n            Sets the source reference on Chart Studio Cloud for\n            `locations`.\n        marker\n            :class:`plotly.graph_objects.choroplethmapbox.Marker`\n            instance or dict with compatible properties\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        reversescale\n            Reverses the color mapping if true. If true, `zmin`\n            will correspond to the last color in the array and\n            `zmax` will correspond to the first color.\n        selected\n            :class:`plotly.graph_objects.choroplethmapbox.Selected`\n            instance or dict with compatible properties\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        showscale\n            Determines whether or not a colorbar is displayed for\n            this trace.\n        stream\n            :class:`plotly.graph_objects.choroplethmapbox.Stream`\n            instance or dict with compatible properties\n        subplot\n            Sets a reference between this trace\'s data coordinates\n            and a mapbox subplot. If "mapbox" (the default value),\n            the data refer to `layout.mapbox`. If "mapbox2", the\n            data refer to `layout.mapbox2`, and so on.\n        text\n            Sets the text elements associated with each location.\n        textsrc\n            Sets the source reference on Chart Studio Cloud for\n            `text`.\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        unselected\n            :class:`plotly.graph_objects.choroplethmapbox.Unselecte\n            d` instance or dict with compatible properties\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        z\n            Sets the color values.\n        zauto\n            Determines whether or not the color domain is computed\n            with respect to the input data (here in `z`) or the\n            bounds set in `zmin` and `zmax` Defaults to `false`\n            when `zmin` and `zmax` are set by the user.\n        zmax\n            Sets the upper bound of the color domain. Value should\n            have the same units as in `z` and if set, `zmin` must\n            be set as well.\n        zmid\n            Sets the mid-point of the color domain by scaling\n            `zmin` and/or `zmax` to be equidistant to this point.\n            Value should have the same units as in `z`. Has no\n            effect when `zauto` is `false`.\n        zmin\n            Sets the lower bound of the color domain. Value should\n            have the same units as in `z` and if set, `zmax` must\n            be set as well.\n        zsrc\n            Sets the source reference on Chart Studio Cloud for\n            `z`.\n        '

    def __init__(self, arg=None, autocolorscale=None, below=None, coloraxis=None, colorbar=None, colorscale=None, customdata=None, customdatasrc=None, featureidkey=None, geojson=None, hoverinfo=None, hoverinfosrc=None, hoverlabel=None, hovertemplate=None, hovertemplatesrc=None, hovertext=None, hovertextsrc=None, ids=None, idssrc=None, legend=None, legendgroup=None, legendgrouptitle=None, legendrank=None, legendwidth=None, locations=None, locationssrc=None, marker=None, meta=None, metasrc=None, name=None, reversescale=None, selected=None, selectedpoints=None, showlegend=None, showscale=None, stream=None, subplot=None, text=None, textsrc=None, uid=None, uirevision=None, unselected=None, visible=None, z=None, zauto=None, zmax=None, zmid=None, zmin=None, zsrc=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Choroplethmapbox object\n\n        GeoJSON features to be filled are set in `geojson` The data\n        that describes the choropleth value-to-color mapping is set in\n        `locations` and `z`.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.Choroplethmapbox`\n        autocolorscale\n            Determines whether the colorscale is a default palette\n            (`autocolorscale: true`) or the palette determined by\n            `colorscale`. In case `colorscale` is unspecified or\n            `autocolorscale` is true, the default palette will be\n            chosen according to whether numbers in the `color`\n            array are all positive, all negative or mixed.\n        below\n            Determines if the choropleth polygons will be inserted\n            before the layer with the specified ID. By default,\n            choroplethmapbox traces are placed above the water\n            layers. If set to \'\', the layer will be inserted above\n            every existing layer.\n        coloraxis\n            Sets a reference to a shared color axis. References to\n            these shared color axes are "coloraxis", "coloraxis2",\n            "coloraxis3", etc. Settings for these shared color axes\n            are set in the layout, under `layout.coloraxis`,\n            `layout.coloraxis2`, etc. Note that multiple color\n            scales can be linked to the same color axis.\n        colorbar\n            :class:`plotly.graph_objects.choroplethmapbox.ColorBar`\n            instance or dict with compatible properties\n        colorscale\n            Sets the colorscale. The colorscale must be an array\n            containing arrays mapping a normalized value to an rgb,\n            rgba, hex, hsl, hsv, or named color string. At minimum,\n            a mapping for the lowest (0) and highest (1) values are\n            required. For example, `[[0, \'rgb(0,0,255)\'], [1,\n            \'rgb(255,0,0)\']]`. To control the bounds of the\n            colorscale in color space, use `zmin` and `zmax`.\n            Alternatively, `colorscale` may be a palette name\n            string of the following list: Blackbody,Bluered,Blues,C\n            ividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portl\n            and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        featureidkey\n            Sets the key in GeoJSON features which is used as id to\n            match the items included in the `locations` array.\n            Support nested property, for example "properties.name".\n        geojson\n            Sets the GeoJSON data associated with this trace. It\n            can be set as a valid GeoJSON object or as a URL\n            string. Note that we only accept GeoJSONs of type\n            "FeatureCollection" or "Feature" with geometries of\n            type "Polygon" or "MultiPolygon".\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.choroplethmapbox.Hoverlabe\n            l` instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variable `properties` Anything contained in tag\n            `<extra>` is displayed in the secondary box, for\n            example "<extra>{fullData.name}</extra>". To hide the\n            secondary box completely, use an empty tag\n            `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Same as `text`.\n        hovertextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertext`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.choroplethmapbox.Legendgro\n            uptitle` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        locations\n            Sets which features found in "geojson" to plot using\n            their feature `id` field.\n        locationssrc\n            Sets the source reference on Chart Studio Cloud for\n            `locations`.\n        marker\n            :class:`plotly.graph_objects.choroplethmapbox.Marker`\n            instance or dict with compatible properties\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        reversescale\n            Reverses the color mapping if true. If true, `zmin`\n            will correspond to the last color in the array and\n            `zmax` will correspond to the first color.\n        selected\n            :class:`plotly.graph_objects.choroplethmapbox.Selected`\n            instance or dict with compatible properties\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        showscale\n            Determines whether or not a colorbar is displayed for\n            this trace.\n        stream\n            :class:`plotly.graph_objects.choroplethmapbox.Stream`\n            instance or dict with compatible properties\n        subplot\n            Sets a reference between this trace\'s data coordinates\n            and a mapbox subplot. If "mapbox" (the default value),\n            the data refer to `layout.mapbox`. If "mapbox2", the\n            data refer to `layout.mapbox2`, and so on.\n        text\n            Sets the text elements associated with each location.\n        textsrc\n            Sets the source reference on Chart Studio Cloud for\n            `text`.\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        unselected\n            :class:`plotly.graph_objects.choroplethmapbox.Unselecte\n            d` instance or dict with compatible properties\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        z\n            Sets the color values.\n        zauto\n            Determines whether or not the color domain is computed\n            with respect to the input data (here in `z`) or the\n            bounds set in `zmin` and `zmax` Defaults to `false`\n            when `zmin` and `zmax` are set by the user.\n        zmax\n            Sets the upper bound of the color domain. Value should\n            have the same units as in `z` and if set, `zmin` must\n            be set as well.\n        zmid\n            Sets the mid-point of the color domain by scaling\n            `zmin` and/or `zmax` to be equidistant to this point.\n            Value should have the same units as in `z`. Has no\n            effect when `zauto` is `false`.\n        zmin\n            Sets the lower bound of the color domain. Value should\n            have the same units as in `z` and if set, `zmax` must\n            be set as well.\n        zsrc\n            Sets the source reference on Chart Studio Cloud for\n            `z`.\n\n        Returns\n        -------\n        Choroplethmapbox\n        '
        super(Choroplethmapbox, self).__init__('choroplethmapbox')
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
            raise ValueError('The first argument to the plotly.graph_objs.Choroplethmapbox\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Choroplethmapbox`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('autocolorscale', None)
        _v = autocolorscale if autocolorscale is not None else _v
        if _v is not None:
            self['autocolorscale'] = _v
        _v = arg.pop('below', None)
        _v = below if below is not None else _v
        if _v is not None:
            self['below'] = _v
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
        _v = arg.pop('featureidkey', None)
        _v = featureidkey if featureidkey is not None else _v
        if _v is not None:
            self['featureidkey'] = _v
        _v = arg.pop('geojson', None)
        _v = geojson if geojson is not None else _v
        if _v is not None:
            self['geojson'] = _v
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
        _v = arg.pop('locations', None)
        _v = locations if locations is not None else _v
        if _v is not None:
            self['locations'] = _v
        _v = arg.pop('locationssrc', None)
        _v = locationssrc if locationssrc is not None else _v
        if _v is not None:
            self['locationssrc'] = _v
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
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('reversescale', None)
        _v = reversescale if reversescale is not None else _v
        if _v is not None:
            self['reversescale'] = _v
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
        _v = arg.pop('showscale', None)
        _v = showscale if showscale is not None else _v
        if _v is not None:
            self['showscale'] = _v
        _v = arg.pop('stream', None)
        _v = stream if stream is not None else _v
        if _v is not None:
            self['stream'] = _v
        _v = arg.pop('subplot', None)
        _v = subplot if subplot is not None else _v
        if _v is not None:
            self['subplot'] = _v
        _v = arg.pop('text', None)
        _v = text if text is not None else _v
        if _v is not None:
            self['text'] = _v
        _v = arg.pop('textsrc', None)
        _v = textsrc if textsrc is not None else _v
        if _v is not None:
            self['textsrc'] = _v
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
        _v = arg.pop('z', None)
        _v = z if z is not None else _v
        if _v is not None:
            self['z'] = _v
        _v = arg.pop('zauto', None)
        _v = zauto if zauto is not None else _v
        if _v is not None:
            self['zauto'] = _v
        _v = arg.pop('zmax', None)
        _v = zmax if zmax is not None else _v
        if _v is not None:
            self['zmax'] = _v
        _v = arg.pop('zmid', None)
        _v = zmid if zmid is not None else _v
        if _v is not None:
            self['zmid'] = _v
        _v = arg.pop('zmin', None)
        _v = zmin if zmin is not None else _v
        if _v is not None:
            self['zmin'] = _v
        _v = arg.pop('zsrc', None)
        _v = zsrc if zsrc is not None else _v
        if _v is not None:
            self['zsrc'] = _v
        self._props['type'] = 'choroplethmapbox'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
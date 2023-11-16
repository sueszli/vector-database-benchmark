from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy

class Scattergeo(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'scattergeo'
    _valid_props = {'connectgaps', 'customdata', 'customdatasrc', 'featureidkey', 'fill', 'fillcolor', 'geo', 'geojson', 'hoverinfo', 'hoverinfosrc', 'hoverlabel', 'hovertemplate', 'hovertemplatesrc', 'hovertext', 'hovertextsrc', 'ids', 'idssrc', 'lat', 'latsrc', 'legend', 'legendgroup', 'legendgrouptitle', 'legendrank', 'legendwidth', 'line', 'locationmode', 'locations', 'locationssrc', 'lon', 'lonsrc', 'marker', 'meta', 'metasrc', 'mode', 'name', 'opacity', 'selected', 'selectedpoints', 'showlegend', 'stream', 'text', 'textfont', 'textposition', 'textpositionsrc', 'textsrc', 'texttemplate', 'texttemplatesrc', 'type', 'uid', 'uirevision', 'unselected', 'visible'}

    @property
    def connectgaps(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether or not gaps (i.e. {nan} or missing values)\n        in the provided data arrays are connected.\n\n        The 'connectgaps' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['connectgaps']

    @connectgaps.setter
    def connectgaps(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['connectgaps'] = val

    @property
    def customdata(self):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        '\n        Sets the key in GeoJSON features which is used as id to match\n        the items included in the `locations` array. Only has an effect\n        when `geojson` is set. Support nested property, for example\n        "properties.name".\n\n        The \'featureidkey\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['featureidkey']

    @featureidkey.setter
    def featureidkey(self, val):
        if False:
            return 10
        self['featureidkey'] = val

    @property
    def fill(self):
        if False:
            while True:
                i = 10
        '\n        Sets the area to fill with a solid color. Use with `fillcolor`\n        if not "none". "toself" connects the endpoints of the trace (or\n        each segment of the trace if it has gaps) into a closed shape.\n\n        The \'fill\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'none\', \'toself\']\n\n        Returns\n        -------\n        Any\n        '
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
            while True:
                i = 10
        "\n        Sets the fill color. Defaults to a half-transparent variant of\n        the line color, marker color, or marker line color, whichever\n        is available.\n\n        The 'fillcolor' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['fillcolor']

    @fillcolor.setter
    def fillcolor(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['fillcolor'] = val

    @property
    def geo(self):
        if False:
            while True:
                i = 10
        '\n        Sets a reference between this trace\'s geospatial coordinates\n        and a geographic map. If "geo" (the default value), the\n        geospatial coordinates refer to `layout.geo`. If "geo2", the\n        geospatial coordinates refer to `layout.geo2`, and so on.\n\n        The \'geo\' property is an identifier of a particular\n        subplot, of type \'geo\', that may be specified as the string \'geo\'\n        optionally followed by an integer >= 1\n        (e.g. \'geo\', \'geo1\', \'geo2\', \'geo3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['geo']

    @geo.setter
    def geo(self, val):
        if False:
            print('Hello World!')
        self['geo'] = val

    @property
    def geojson(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets optional GeoJSON data associated with this trace. If not\n        given, the features on the base map are used when `locations`\n        is set. It can be set as a valid GeoJSON object or as a URL\n        string. Note that we only accept GeoJSONs of type\n        "FeatureCollection" or "Feature" with geometries of type\n        "Polygon" or "MultiPolygon".\n\n        The \'geojson\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['geojson']

    @geojson.setter
    def geojson(self, val):
        if False:
            i = 10
            return i + 15
        self['geojson'] = val

    @property
    def hoverinfo(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines which trace information appear on hover. If `none`\n        or `skip` are set, no information is displayed upon hovering.\n        But, if `none` is set, click and hover events are still fired.\n\n        The 'hoverinfo' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['lon', 'lat', 'location', 'text', 'name'] joined with '+' characters\n            (e.g. 'lon+lat')\n            OR exactly one of ['all', 'none', 'skip'] (e.g. 'skip')\n          - A list or array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
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
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `hoverinfo`.\n\n        The 'hoverinfosrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['hoverinfosrc']

    @hoverinfosrc.setter
    def hoverinfosrc(self, val):
        if False:
            while True:
                i = 10
        self['hoverinfosrc'] = val

    @property
    def hoverlabel(self):
        if False:
            return 10
        "\n        The 'hoverlabel' property is an instance of Hoverlabel\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scattergeo.Hoverlabel`\n          - A dict of string/value properties that will be passed\n            to the Hoverlabel constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the text\n                    content within hover label box. Has an effect\n                    only if the hover label text spans more two or\n                    more lines\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                bgcolor\n                    Sets the background color of the hover labels\n                    for this trace\n                bgcolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bgcolor`.\n                bordercolor\n                    Sets the border color of the hover labels for\n                    this trace.\n                bordercolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bordercolor`.\n                font\n                    Sets the font used in hover labels.\n                namelength\n                    Sets the default length (in number of\n                    characters) of the trace name in the hover\n                    labels for all traces. -1 shows the whole name\n                    regardless of length. 0-3 shows the first 0-3\n                    characters, and an integer >3 will show the\n                    whole name if it is less than that many\n                    characters, but if it is longer, will truncate\n                    to `namelength - 3` characters and add an\n                    ellipsis.\n                namelengthsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `namelength`.\n\n        Returns\n        -------\n        plotly.graph_objs.scattergeo.Hoverlabel\n        "
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        if False:
            print('Hello World!')
        self['hoverlabel'] = val

    @property
    def hovertemplate(self):
        if False:
            while True:
                i = 10
        '\n        Template string used for rendering the information that appear\n        on hover box. Note that this will override `hoverinfo`.\n        Variables are inserted using %{variable}, for example "y: %{y}"\n        as well as %{xother}, {%_xother}, {%_xother_}, {%xother_}. When\n        showing info for several points, "xother" will be added to\n        those with different x positions from the first point. An\n        underscore before or after "(x|y)other" will add a space on\n        that side, only when this field is shown. Numbers are formatted\n        using d3-format\'s syntax %{variable:d3-format}, for example\n        "Price: %{y:$.2f}".\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for\n        details on the formatting syntax. Dates are formatted using\n        d3-time-format\'s syntax %{variable|d3-time-format}, for example\n        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format for details on the date\n        formatting syntax. The variables available in `hovertemplate`\n        are the ones emitted as event data described at this link\n        https://plotly.com/javascript/plotlyjs-events/#event-data.\n        Additionally, every attributes that can be specified per-point\n        (the ones that are `arrayOk: true`) are available.  Anything\n        contained in tag `<extra>` is displayed in the secondary box,\n        for example "<extra>{fullData.name}</extra>". To hide the\n        secondary box completely, use an empty tag `<extra></extra>`.\n\n        The \'hovertemplate\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['hovertemplate']

    @hovertemplate.setter
    def hovertemplate(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['hovertemplate'] = val

    @property
    def hovertemplatesrc(self):
        if False:
            for i in range(10):
                print('nop')
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
        '\n        Sets hover text elements associated with each (lon,lat) pair or\n        item in `locations`. If a single string, the same string\n        appears over all the data points. If an array of string, the\n        items are mapped in order to the this trace\'s (lon,lat) or\n        `locations` coordinates. To be seen, trace `hoverinfo` must\n        contain a "text" flag.\n\n        The \'hovertext\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['hovertext']

    @hovertext.setter
    def hovertext(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['hovertext'] = val

    @property
    def hovertextsrc(self):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
        "\n        Assigns id labels to each datum. These ids for object constancy\n        of data points during animation. Should be an array of strings,\n        not numbers or any other type.\n\n        The 'ids' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['ids']

    @ids.setter
    def ids(self, val):
        if False:
            while True:
                i = 10
        self['ids'] = val

    @property
    def idssrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `ids`.\n\n        The 'idssrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['idssrc']

    @idssrc.setter
    def idssrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['idssrc'] = val

    @property
    def lat(self):
        if False:
            return 10
        "\n        Sets the latitude coordinates (in degrees North).\n\n        The 'lat' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['lat']

    @lat.setter
    def lat(self, val):
        if False:
            return 10
        self['lat'] = val

    @property
    def latsrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `lat`.\n\n        The 'latsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['latsrc']

    @latsrc.setter
    def latsrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['latsrc'] = val

    @property
    def legend(self):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
        "\n        Sets the legend group for this trace. Traces and shapes part of\n        the same legend group hide/show at the same time when toggling\n        legend items.\n\n        The 'legendgroup' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['legendgroup']

    @legendgroup.setter
    def legendgroup(self, val):
        if False:
            print('Hello World!')
        self['legendgroup'] = val

    @property
    def legendgrouptitle(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'legendgrouptitle' property is an instance of Legendgrouptitle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scattergeo.Legendgrouptitle`\n          - A dict of string/value properties that will be passed\n            to the Legendgrouptitle constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend group's title font.\n                text\n                    Sets the title of the legend group.\n\n        Returns\n        -------\n        plotly.graph_objs.scattergeo.Legendgrouptitle\n        "
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
            for i in range(10):
                print('nop')
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
            return 10
        self['legendwidth'] = val

    @property
    def line(self):
        if False:
            print('Hello World!')
        '\n        The \'line\' property is an instance of Line\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scattergeo.Line`\n          - A dict of string/value properties that will be passed\n            to the Line constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the line color.\n                dash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                width\n                    Sets the line width (in px).\n\n        Returns\n        -------\n        plotly.graph_objs.scattergeo.Line\n        '
        return self['line']

    @line.setter
    def line(self, val):
        if False:
            while True:
                i = 10
        self['line'] = val

    @property
    def locationmode(self):
        if False:
            print('Hello World!')
        '\n        Determines the set of locations used to match entries in\n        `locations` to regions on the map. Values "ISO-3", "USA-\n        states", *country names* correspond to features on the base map\n        and value "geojson-id" corresponds to features from a custom\n        GeoJSON linked to the `geojson` attribute.\n\n        The \'locationmode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'ISO-3\', \'USA-states\', \'country names\', \'geojson-id\']\n\n        Returns\n        -------\n        Any\n        '
        return self['locationmode']

    @locationmode.setter
    def locationmode(self, val):
        if False:
            i = 10
            return i + 15
        self['locationmode'] = val

    @property
    def locations(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the coordinates via location IDs or names. Coordinates\n        correspond to the centroid of each location given. See\n        `locationmode` for more info.\n\n        The 'locations' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['locations']

    @locations.setter
    def locations(self, val):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
        self['locationssrc'] = val

    @property
    def lon(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the longitude coordinates (in degrees East).\n\n        The 'lon' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['lon']

    @lon.setter
    def lon(self, val):
        if False:
            i = 10
            return i + 15
        self['lon'] = val

    @property
    def lonsrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `lon`.\n\n        The 'lonsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['lonsrc']

    @lonsrc.setter
    def lonsrc(self, val):
        if False:
            print('Hello World!')
        self['lonsrc'] = val

    @property
    def marker(self):
        if False:
            print('Hello World!')
        '\n        The \'marker\' property is an instance of Marker\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scattergeo.Marker`\n          - A dict of string/value properties that will be passed\n            to the Marker constructor\n\n            Supported dict properties:\n\n                angle\n                    Sets the marker angle in respect to `angleref`.\n                angleref\n                    Sets the reference for marker angle. With\n                    "previous", angle 0 points along the line from\n                    the previous point to this one. With "up",\n                    angle 0 points toward the top of the screen.\n                    With "north", angle 0 points north based on the\n                    current map projection.\n                anglesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `angle`.\n                autocolorscale\n                    Determines whether the colorscale is a default\n                    palette (`autocolorscale: true`) or the palette\n                    determined by `marker.colorscale`. Has an\n                    effect only if in `marker.color` is set to a\n                    numerical array. In case `colorscale` is\n                    unspecified or `autocolorscale` is true, the\n                    default palette will be chosen according to\n                    whether numbers in the `color` array are all\n                    positive, all negative or mixed.\n                cauto\n                    Determines whether or not the color domain is\n                    computed with respect to the input data (here\n                    in `marker.color`) or the bounds set in\n                    `marker.cmin` and `marker.cmax` Has an effect\n                    only if in `marker.color` is set to a numerical\n                    array. Defaults to `false` when `marker.cmin`\n                    and `marker.cmax` are set by the user.\n                cmax\n                    Sets the upper bound of the color domain. Has\n                    an effect only if in `marker.color` is set to a\n                    numerical array. Value should have the same\n                    units as in `marker.color` and if set,\n                    `marker.cmin` must be set as well.\n                cmid\n                    Sets the mid-point of the color domain by\n                    scaling `marker.cmin` and/or `marker.cmax` to\n                    be equidistant to this point. Has an effect\n                    only if in `marker.color` is set to a numerical\n                    array. Value should have the same units as in\n                    `marker.color`. Has no effect when\n                    `marker.cauto` is `false`.\n                cmin\n                    Sets the lower bound of the color domain. Has\n                    an effect only if in `marker.color` is set to a\n                    numerical array. Value should have the same\n                    units as in `marker.color` and if set,\n                    `marker.cmax` must be set as well.\n                color\n                    Sets the marker color. It accepts either a\n                    specific color or an array of numbers that are\n                    mapped to the colorscale relative to the max\n                    and min values of the array or relative to\n                    `marker.cmin` and `marker.cmax` if set.\n                coloraxis\n                    Sets a reference to a shared color axis.\n                    References to these shared color axes are\n                    "coloraxis", "coloraxis2", "coloraxis3", etc.\n                    Settings for these shared color axes are set in\n                    the layout, under `layout.coloraxis`,\n                    `layout.coloraxis2`, etc. Note that multiple\n                    color scales can be linked to the same color\n                    axis.\n                colorbar\n                    :class:`plotly.graph_objects.scattergeo.marker.\n                    ColorBar` instance or dict with compatible\n                    properties\n                colorscale\n                    Sets the colorscale. Has an effect only if in\n                    `marker.color` is set to a numerical array. The\n                    colorscale must be an array containing arrays\n                    mapping a normalized value to an rgb, rgba,\n                    hex, hsl, hsv, or named color string. At\n                    minimum, a mapping for the lowest (0) and\n                    highest (1) values are required. For example,\n                    `[[0, \'rgb(0,0,255)\'], [1, \'rgb(255,0,0)\']]`.\n                    To control the bounds of the colorscale in\n                    color space, use `marker.cmin` and\n                    `marker.cmax`. Alternatively, `colorscale` may\n                    be a palette name string of the following list:\n                    Blackbody,Bluered,Blues,Cividis,Earth,Electric,\n                    Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,Rd\n                    Bu,Reds,Viridis,YlGnBu,YlOrRd.\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                gradient\n                    :class:`plotly.graph_objects.scattergeo.marker.\n                    Gradient` instance or dict with compatible\n                    properties\n                line\n                    :class:`plotly.graph_objects.scattergeo.marker.\n                    Line` instance or dict with compatible\n                    properties\n                opacity\n                    Sets the marker opacity.\n                opacitysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `opacity`.\n                reversescale\n                    Reverses the color mapping if true. Has an\n                    effect only if in `marker.color` is set to a\n                    numerical array. If true, `marker.cmin` will\n                    correspond to the last color in the array and\n                    `marker.cmax` will correspond to the first\n                    color.\n                showscale\n                    Determines whether or not a colorbar is\n                    displayed for this trace. Has an effect only if\n                    in `marker.color` is set to a numerical array.\n                size\n                    Sets the marker size (in px).\n                sizemin\n                    Has an effect only if `marker.size` is set to a\n                    numerical array. Sets the minimum size (in px)\n                    of the rendered marker points.\n                sizemode\n                    Has an effect only if `marker.size` is set to a\n                    numerical array. Sets the rule for which the\n                    data in `size` is converted to pixels.\n                sizeref\n                    Has an effect only if `marker.size` is set to a\n                    numerical array. Sets the scale factor used to\n                    determine the rendered size of marker points.\n                    Use with `sizemin` and `sizemode`.\n                sizesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `size`.\n                standoff\n                    Moves the marker away from the data point in\n                    the direction of `angle` (in px). This can be\n                    useful for example if you have another marker\n                    at this location and you want to point an\n                    arrowhead marker at it.\n                standoffsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `standoff`.\n                symbol\n                    Sets the marker symbol type. Adding 100 is\n                    equivalent to appending "-open" to a symbol\n                    name. Adding 200 is equivalent to appending\n                    "-dot" to a symbol name. Adding 300 is\n                    equivalent to appending "-open-dot" or "dot-\n                    open" to a symbol name.\n                symbolsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `symbol`.\n\n        Returns\n        -------\n        plotly.graph_objs.scattergeo.Marker\n        '
        return self['marker']

    @marker.setter
    def marker(self, val):
        if False:
            return 10
        self['marker'] = val

    @property
    def meta(self):
        if False:
            i = 10
            return i + 15
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
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `meta`.\n\n        The 'metasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['metasrc']

    @metasrc.setter
    def metasrc(self, val):
        if False:
            i = 10
            return i + 15
        self['metasrc'] = val

    @property
    def mode(self):
        if False:
            while True:
                i = 10
        '\n        Determines the drawing mode for this scatter trace. If the\n        provided `mode` includes "text" then the `text` elements appear\n        at the coordinates. Otherwise, the `text` elements appear on\n        hover. If there are less than 20 points and the trace is not\n        stacked then the default is "lines+markers". Otherwise,\n        "lines".\n\n        The \'mode\' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of [\'lines\', \'markers\', \'text\'] joined with \'+\' characters\n            (e.g. \'lines+markers\')\n            OR exactly one of [\'none\'] (e.g. \'none\')\n\n        Returns\n        -------\n        Any\n        '
        return self['mode']

    @mode.setter
    def mode(self, val):
        if False:
            i = 10
            return i + 15
        self['mode'] = val

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
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
        "\n        The 'selected' property is an instance of Selected\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scattergeo.Selected`\n          - A dict of string/value properties that will be passed\n            to the Selected constructor\n\n            Supported dict properties:\n\n                marker\n                    :class:`plotly.graph_objects.scattergeo.selecte\n                    d.Marker` instance or dict with compatible\n                    properties\n                textfont\n                    :class:`plotly.graph_objects.scattergeo.selecte\n                    d.Textfont` instance or dict with compatible\n                    properties\n\n        Returns\n        -------\n        plotly.graph_objs.scattergeo.Selected\n        "
        return self['selected']

    @selected.setter
    def selected(self, val):
        if False:
            return 10
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
            print('Hello World!')
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
            print('Hello World!')
        self['showlegend'] = val

    @property
    def stream(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'stream' property is an instance of Stream\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scattergeo.Stream`\n          - A dict of string/value properties that will be passed\n            to the Stream constructor\n\n            Supported dict properties:\n\n                maxpoints\n                    Sets the maximum number of points to keep on\n                    the plots from an incoming stream. If\n                    `maxpoints` is set to 50, only the newest 50\n                    points will be displayed on the plot.\n                token\n                    The stream id number links a data trace on a\n                    plot with a stream. See https://chart-\n                    studio.plotly.com/settings for more details.\n\n        Returns\n        -------\n        plotly.graph_objs.scattergeo.Stream\n        "
        return self['stream']

    @stream.setter
    def stream(self, val):
        if False:
            i = 10
            return i + 15
        self['stream'] = val

    @property
    def text(self):
        if False:
            print('Hello World!')
        '\n        Sets text elements associated with each (lon,lat) pair or item\n        in `locations`. If a single string, the same string appears\n        over all the data points. If an array of string, the items are\n        mapped in order to the this trace\'s (lon,lat) or `locations`\n        coordinates. If trace `hoverinfo` contains a "text" flag and\n        "hovertext" is not set, these elements will be seen in the\n        hover labels.\n\n        The \'text\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
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
        '\n        Sets the text font.\n\n        The \'textfont\' property is an instance of Textfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scattergeo.Textfont`\n          - A dict of string/value properties that will be passed\n            to the Textfont constructor\n\n            Supported dict properties:\n\n                color\n\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                familysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `family`.\n                size\n\n                sizesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `size`.\n\n        Returns\n        -------\n        plotly.graph_objs.scattergeo.Textfont\n        '
        return self['textfont']

    @textfont.setter
    def textfont(self, val):
        if False:
            print('Hello World!')
        self['textfont'] = val

    @property
    def textposition(self):
        if False:
            while True:
                i = 10
        "\n        Sets the positions of the `text` elements with respects to the\n        (x,y) coordinates.\n\n        The 'textposition' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['top left', 'top center', 'top right', 'middle left',\n                'middle center', 'middle right', 'bottom left', 'bottom\n                center', 'bottom right']\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['textposition']

    @textposition.setter
    def textposition(self, val):
        if False:
            while True:
                i = 10
        self['textposition'] = val

    @property
    def textpositionsrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for\n        `textposition`.\n\n        The 'textpositionsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['textpositionsrc']

    @textpositionsrc.setter
    def textpositionsrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['textpositionsrc'] = val

    @property
    def textsrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `text`.\n\n        The 'textsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['textsrc']

    @textsrc.setter
    def textsrc(self, val):
        if False:
            return 10
        self['textsrc'] = val

    @property
    def texttemplate(self):
        if False:
            i = 10
            return i + 15
        '\n        Template string used for rendering the information text that\n        appear on points. Note that this will override `textinfo`.\n        Variables are inserted using %{variable}, for example "y:\n        %{y}". Numbers are formatted using d3-format\'s syntax\n        %{variable:d3-format}, for example "Price: %{y:$.2f}".\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for\n        details on the formatting syntax. Dates are formatted using\n        d3-time-format\'s syntax %{variable|d3-time-format}, for example\n        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format for details on the date\n        formatting syntax. Every attributes that can be specified per-\n        point (the ones that are `arrayOk: true`) are available.\n        Finally, the template string has access to variables `lat`,\n        `lon`, `location` and `text`.\n\n        The \'texttemplate\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['texttemplate']

    @texttemplate.setter
    def texttemplate(self, val):
        if False:
            return 10
        self['texttemplate'] = val

    @property
    def texttemplatesrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `texttemplate`.\n\n        The 'texttemplatesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['texttemplatesrc']

    @texttemplatesrc.setter
    def texttemplatesrc(self, val):
        if False:
            while True:
                i = 10
        self['texttemplatesrc'] = val

    @property
    def uid(self):
        if False:
            i = 10
            return i + 15
        "\n        Assign an id to this trace, Use this to provide object\n        constancy between traces during animations and transitions.\n\n        The 'uid' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['uid']

    @uid.setter
    def uid(self, val):
        if False:
            return 10
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
            while True:
                i = 10
        self['uirevision'] = val

    @property
    def unselected(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'unselected' property is an instance of Unselected\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.scattergeo.Unselected`\n          - A dict of string/value properties that will be passed\n            to the Unselected constructor\n\n            Supported dict properties:\n\n                marker\n                    :class:`plotly.graph_objects.scattergeo.unselec\n                    ted.Marker` instance or dict with compatible\n                    properties\n                textfont\n                    :class:`plotly.graph_objects.scattergeo.unselec\n                    ted.Textfont` instance or dict with compatible\n                    properties\n\n        Returns\n        -------\n        plotly.graph_objs.scattergeo.Unselected\n        "
        return self['unselected']

    @unselected.setter
    def unselected(self, val):
        if False:
            i = 10
            return i + 15
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
            return 10
        self['visible'] = val

    @property
    def type(self):
        if False:
            while True:
                i = 10
        return self._props['type']

    @property
    def _prop_descriptions(self):
        if False:
            i = 10
            return i + 15
        return '        connectgaps\n            Determines whether or not gaps (i.e. {nan} or missing\n            values) in the provided data arrays are connected.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        featureidkey\n            Sets the key in GeoJSON features which is used as id to\n            match the items included in the `locations` array. Only\n            has an effect when `geojson` is set. Support nested\n            property, for example "properties.name".\n        fill\n            Sets the area to fill with a solid color. Use with\n            `fillcolor` if not "none". "toself" connects the\n            endpoints of the trace (or each segment of the trace if\n            it has gaps) into a closed shape.\n        fillcolor\n            Sets the fill color. Defaults to a half-transparent\n            variant of the line color, marker color, or marker line\n            color, whichever is available.\n        geo\n            Sets a reference between this trace\'s geospatial\n            coordinates and a geographic map. If "geo" (the default\n            value), the geospatial coordinates refer to\n            `layout.geo`. If "geo2", the geospatial coordinates\n            refer to `layout.geo2`, and so on.\n        geojson\n            Sets optional GeoJSON data associated with this trace.\n            If not given, the features on the base map are used\n            when `locations` is set. It can be set as a valid\n            GeoJSON object or as a URL string. Note that we only\n            accept GeoJSONs of type "FeatureCollection" or\n            "Feature" with geometries of type "Polygon" or\n            "MultiPolygon".\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.scattergeo.Hoverlabel`\n            instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available.  Anything contained in tag `<extra>` is\n            displayed in the secondary box, for example\n            "<extra>{fullData.name}</extra>". To hide the secondary\n            box completely, use an empty tag `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Sets hover text elements associated with each (lon,lat)\n            pair or item in `locations`. If a single string, the\n            same string appears over all the data points. If an\n            array of string, the items are mapped in order to the\n            this trace\'s (lon,lat) or `locations` coordinates. To\n            be seen, trace `hoverinfo` must contain a "text" flag.\n        hovertextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertext`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        lat\n            Sets the latitude coordinates (in degrees North).\n        latsrc\n            Sets the source reference on Chart Studio Cloud for\n            `lat`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.scattergeo.Legendgrouptitl\n            e` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        line\n            :class:`plotly.graph_objects.scattergeo.Line` instance\n            or dict with compatible properties\n        locationmode\n            Determines the set of locations used to match entries\n            in `locations` to regions on the map. Values "ISO-3",\n            "USA-states", *country names* correspond to features on\n            the base map and value "geojson-id" corresponds to\n            features from a custom GeoJSON linked to the `geojson`\n            attribute.\n        locations\n            Sets the coordinates via location IDs or names.\n            Coordinates correspond to the centroid of each location\n            given. See `locationmode` for more info.\n        locationssrc\n            Sets the source reference on Chart Studio Cloud for\n            `locations`.\n        lon\n            Sets the longitude coordinates (in degrees East).\n        lonsrc\n            Sets the source reference on Chart Studio Cloud for\n            `lon`.\n        marker\n            :class:`plotly.graph_objects.scattergeo.Marker`\n            instance or dict with compatible properties\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        mode\n            Determines the drawing mode for this scatter trace. If\n            the provided `mode` includes "text" then the `text`\n            elements appear at the coordinates. Otherwise, the\n            `text` elements appear on hover. If there are less than\n            20 points and the trace is not stacked then the default\n            is "lines+markers". Otherwise, "lines".\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        opacity\n            Sets the opacity of the trace.\n        selected\n            :class:`plotly.graph_objects.scattergeo.Selected`\n            instance or dict with compatible properties\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        stream\n            :class:`plotly.graph_objects.scattergeo.Stream`\n            instance or dict with compatible properties\n        text\n            Sets text elements associated with each (lon,lat) pair\n            or item in `locations`. If a single string, the same\n            string appears over all the data points. If an array of\n            string, the items are mapped in order to the this\n            trace\'s (lon,lat) or `locations` coordinates. If trace\n            `hoverinfo` contains a "text" flag and "hovertext" is\n            not set, these elements will be seen in the hover\n            labels.\n        textfont\n            Sets the text font.\n        textposition\n            Sets the positions of the `text` elements with respects\n            to the (x,y) coordinates.\n        textpositionsrc\n            Sets the source reference on Chart Studio Cloud for\n            `textposition`.\n        textsrc\n            Sets the source reference on Chart Studio Cloud for\n            `text`.\n        texttemplate\n            Template string used for rendering the information text\n            that appear on points. Note that this will override\n            `textinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}". Numbers are formatted using\n            d3-format\'s syntax %{variable:d3-format}, for example\n            "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. Every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `lat`, `lon`, `location` and `text`.\n        texttemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `texttemplate`.\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        unselected\n            :class:`plotly.graph_objects.scattergeo.Unselected`\n            instance or dict with compatible properties\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        '

    def __init__(self, arg=None, connectgaps=None, customdata=None, customdatasrc=None, featureidkey=None, fill=None, fillcolor=None, geo=None, geojson=None, hoverinfo=None, hoverinfosrc=None, hoverlabel=None, hovertemplate=None, hovertemplatesrc=None, hovertext=None, hovertextsrc=None, ids=None, idssrc=None, lat=None, latsrc=None, legend=None, legendgroup=None, legendgrouptitle=None, legendrank=None, legendwidth=None, line=None, locationmode=None, locations=None, locationssrc=None, lon=None, lonsrc=None, marker=None, meta=None, metasrc=None, mode=None, name=None, opacity=None, selected=None, selectedpoints=None, showlegend=None, stream=None, text=None, textfont=None, textposition=None, textpositionsrc=None, textsrc=None, texttemplate=None, texttemplatesrc=None, uid=None, uirevision=None, unselected=None, visible=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Scattergeo object\n\n        The data visualized as scatter point or lines on a geographic\n        map is provided either by longitude/latitude pairs in `lon` and\n        `lat` respectively or by geographic location IDs or names in\n        `locations`.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.Scattergeo`\n        connectgaps\n            Determines whether or not gaps (i.e. {nan} or missing\n            values) in the provided data arrays are connected.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        featureidkey\n            Sets the key in GeoJSON features which is used as id to\n            match the items included in the `locations` array. Only\n            has an effect when `geojson` is set. Support nested\n            property, for example "properties.name".\n        fill\n            Sets the area to fill with a solid color. Use with\n            `fillcolor` if not "none". "toself" connects the\n            endpoints of the trace (or each segment of the trace if\n            it has gaps) into a closed shape.\n        fillcolor\n            Sets the fill color. Defaults to a half-transparent\n            variant of the line color, marker color, or marker line\n            color, whichever is available.\n        geo\n            Sets a reference between this trace\'s geospatial\n            coordinates and a geographic map. If "geo" (the default\n            value), the geospatial coordinates refer to\n            `layout.geo`. If "geo2", the geospatial coordinates\n            refer to `layout.geo2`, and so on.\n        geojson\n            Sets optional GeoJSON data associated with this trace.\n            If not given, the features on the base map are used\n            when `locations` is set. It can be set as a valid\n            GeoJSON object or as a URL string. Note that we only\n            accept GeoJSONs of type "FeatureCollection" or\n            "Feature" with geometries of type "Polygon" or\n            "MultiPolygon".\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.scattergeo.Hoverlabel`\n            instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available.  Anything contained in tag `<extra>` is\n            displayed in the secondary box, for example\n            "<extra>{fullData.name}</extra>". To hide the secondary\n            box completely, use an empty tag `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Sets hover text elements associated with each (lon,lat)\n            pair or item in `locations`. If a single string, the\n            same string appears over all the data points. If an\n            array of string, the items are mapped in order to the\n            this trace\'s (lon,lat) or `locations` coordinates. To\n            be seen, trace `hoverinfo` must contain a "text" flag.\n        hovertextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertext`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        lat\n            Sets the latitude coordinates (in degrees North).\n        latsrc\n            Sets the source reference on Chart Studio Cloud for\n            `lat`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.scattergeo.Legendgrouptitl\n            e` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        line\n            :class:`plotly.graph_objects.scattergeo.Line` instance\n            or dict with compatible properties\n        locationmode\n            Determines the set of locations used to match entries\n            in `locations` to regions on the map. Values "ISO-3",\n            "USA-states", *country names* correspond to features on\n            the base map and value "geojson-id" corresponds to\n            features from a custom GeoJSON linked to the `geojson`\n            attribute.\n        locations\n            Sets the coordinates via location IDs or names.\n            Coordinates correspond to the centroid of each location\n            given. See `locationmode` for more info.\n        locationssrc\n            Sets the source reference on Chart Studio Cloud for\n            `locations`.\n        lon\n            Sets the longitude coordinates (in degrees East).\n        lonsrc\n            Sets the source reference on Chart Studio Cloud for\n            `lon`.\n        marker\n            :class:`plotly.graph_objects.scattergeo.Marker`\n            instance or dict with compatible properties\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        mode\n            Determines the drawing mode for this scatter trace. If\n            the provided `mode` includes "text" then the `text`\n            elements appear at the coordinates. Otherwise, the\n            `text` elements appear on hover. If there are less than\n            20 points and the trace is not stacked then the default\n            is "lines+markers". Otherwise, "lines".\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        opacity\n            Sets the opacity of the trace.\n        selected\n            :class:`plotly.graph_objects.scattergeo.Selected`\n            instance or dict with compatible properties\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        stream\n            :class:`plotly.graph_objects.scattergeo.Stream`\n            instance or dict with compatible properties\n        text\n            Sets text elements associated with each (lon,lat) pair\n            or item in `locations`. If a single string, the same\n            string appears over all the data points. If an array of\n            string, the items are mapped in order to the this\n            trace\'s (lon,lat) or `locations` coordinates. If trace\n            `hoverinfo` contains a "text" flag and "hovertext" is\n            not set, these elements will be seen in the hover\n            labels.\n        textfont\n            Sets the text font.\n        textposition\n            Sets the positions of the `text` elements with respects\n            to the (x,y) coordinates.\n        textpositionsrc\n            Sets the source reference on Chart Studio Cloud for\n            `textposition`.\n        textsrc\n            Sets the source reference on Chart Studio Cloud for\n            `text`.\n        texttemplate\n            Template string used for rendering the information text\n            that appear on points. Note that this will override\n            `textinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}". Numbers are formatted using\n            d3-format\'s syntax %{variable:d3-format}, for example\n            "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. Every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `lat`, `lon`, `location` and `text`.\n        texttemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `texttemplate`.\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        unselected\n            :class:`plotly.graph_objects.scattergeo.Unselected`\n            instance or dict with compatible properties\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n\n        Returns\n        -------\n        Scattergeo\n        '
        super(Scattergeo, self).__init__('scattergeo')
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
            raise ValueError('The first argument to the plotly.graph_objs.Scattergeo\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Scattergeo`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('connectgaps', None)
        _v = connectgaps if connectgaps is not None else _v
        if _v is not None:
            self['connectgaps'] = _v
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
        _v = arg.pop('fill', None)
        _v = fill if fill is not None else _v
        if _v is not None:
            self['fill'] = _v
        _v = arg.pop('fillcolor', None)
        _v = fillcolor if fillcolor is not None else _v
        if _v is not None:
            self['fillcolor'] = _v
        _v = arg.pop('geo', None)
        _v = geo if geo is not None else _v
        if _v is not None:
            self['geo'] = _v
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
        _v = arg.pop('lat', None)
        _v = lat if lat is not None else _v
        if _v is not None:
            self['lat'] = _v
        _v = arg.pop('latsrc', None)
        _v = latsrc if latsrc is not None else _v
        if _v is not None:
            self['latsrc'] = _v
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
        _v = arg.pop('locationmode', None)
        _v = locationmode if locationmode is not None else _v
        if _v is not None:
            self['locationmode'] = _v
        _v = arg.pop('locations', None)
        _v = locations if locations is not None else _v
        if _v is not None:
            self['locations'] = _v
        _v = arg.pop('locationssrc', None)
        _v = locationssrc if locationssrc is not None else _v
        if _v is not None:
            self['locationssrc'] = _v
        _v = arg.pop('lon', None)
        _v = lon if lon is not None else _v
        if _v is not None:
            self['lon'] = _v
        _v = arg.pop('lonsrc', None)
        _v = lonsrc if lonsrc is not None else _v
        if _v is not None:
            self['lonsrc'] = _v
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
        self._props['type'] = 'scattergeo'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy

class Waterfall(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'waterfall'
    _valid_props = {'alignmentgroup', 'base', 'cliponaxis', 'connector', 'constraintext', 'customdata', 'customdatasrc', 'decreasing', 'dx', 'dy', 'hoverinfo', 'hoverinfosrc', 'hoverlabel', 'hovertemplate', 'hovertemplatesrc', 'hovertext', 'hovertextsrc', 'ids', 'idssrc', 'increasing', 'insidetextanchor', 'insidetextfont', 'legend', 'legendgroup', 'legendgrouptitle', 'legendrank', 'legendwidth', 'measure', 'measuresrc', 'meta', 'metasrc', 'name', 'offset', 'offsetgroup', 'offsetsrc', 'opacity', 'orientation', 'outsidetextfont', 'selectedpoints', 'showlegend', 'stream', 'text', 'textangle', 'textfont', 'textinfo', 'textposition', 'textpositionsrc', 'textsrc', 'texttemplate', 'texttemplatesrc', 'totals', 'type', 'uid', 'uirevision', 'visible', 'width', 'widthsrc', 'x', 'x0', 'xaxis', 'xhoverformat', 'xperiod', 'xperiod0', 'xperiodalignment', 'xsrc', 'y', 'y0', 'yaxis', 'yhoverformat', 'yperiod', 'yperiod0', 'yperiodalignment', 'ysrc'}

    @property
    def alignmentgroup(self):
        if False:
            print('Hello World!')
        "\n        Set several traces linked to the same position axis or matching\n        axes to the same alignmentgroup. This controls whether bars\n        compute their positional range dependently or independently.\n\n        The 'alignmentgroup' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['alignmentgroup']

    @alignmentgroup.setter
    def alignmentgroup(self, val):
        if False:
            while True:
                i = 10
        self['alignmentgroup'] = val

    @property
    def base(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets where the bar base is drawn (in position axis units).\n\n        The 'base' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['base']

    @base.setter
    def base(self, val):
        if False:
            i = 10
            return i + 15
        self['base'] = val

    @property
    def cliponaxis(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether the text nodes are clipped about the subplot\n        axes. To show the text nodes above axis lines and tick labels,\n        make sure to set `xaxis.layer` and `yaxis.layer` to *below\n        traces*.\n\n        The 'cliponaxis' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['cliponaxis']

    @cliponaxis.setter
    def cliponaxis(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['cliponaxis'] = val

    @property
    def connector(self):
        if False:
            return 10
        "\n        The 'connector' property is an instance of Connector\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Connector`\n          - A dict of string/value properties that will be passed\n            to the Connector constructor\n\n            Supported dict properties:\n\n                line\n                    :class:`plotly.graph_objects.waterfall.connecto\n                    r.Line` instance or dict with compatible\n                    properties\n                mode\n                    Sets the shape of connector lines.\n                visible\n                    Determines if connector lines are drawn.\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Connector\n        "
        return self['connector']

    @connector.setter
    def connector(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['connector'] = val

    @property
    def constraintext(self):
        if False:
            while True:
                i = 10
        "\n        Constrain the size of text inside or outside a bar to be no\n        larger than the bar itself.\n\n        The 'constraintext' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['inside', 'outside', 'both', 'none']\n\n        Returns\n        -------\n        Any\n        "
        return self['constraintext']

    @constraintext.setter
    def constraintext(self, val):
        if False:
            while True:
                i = 10
        self['constraintext'] = val

    @property
    def customdata(self):
        if False:
            print('Hello World!')
        '\n        Assigns extra data each datum. This may be useful when\n        listening to hover, click and selection events. Note that,\n        "scatter" traces also appends customdata items in the markers\n        DOM elements\n\n        The \'customdata\' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        '
        return self['customdata']

    @customdata.setter
    def customdata(self, val):
        if False:
            while True:
                i = 10
        self['customdata'] = val

    @property
    def customdatasrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for\n        `customdata`.\n\n        The 'customdatasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['customdatasrc']

    @customdatasrc.setter
    def customdatasrc(self, val):
        if False:
            while True:
                i = 10
        self['customdatasrc'] = val

    @property
    def decreasing(self):
        if False:
            while True:
                i = 10
        "\n        The 'decreasing' property is an instance of Decreasing\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Decreasing`\n          - A dict of string/value properties that will be passed\n            to the Decreasing constructor\n\n            Supported dict properties:\n\n                marker\n                    :class:`plotly.graph_objects.waterfall.decreasi\n                    ng.Marker` instance or dict with compatible\n                    properties\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Decreasing\n        "
        return self['decreasing']

    @decreasing.setter
    def decreasing(self, val):
        if False:
            return 10
        self['decreasing'] = val

    @property
    def dx(self):
        if False:
            while True:
                i = 10
        "\n        Sets the x coordinate step. See `x0` for more info.\n\n        The 'dx' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['dx']

    @dx.setter
    def dx(self, val):
        if False:
            return 10
        self['dx'] = val

    @property
    def dy(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the y coordinate step. See `y0` for more info.\n\n        The 'dy' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['dy']

    @dy.setter
    def dy(self, val):
        if False:
            print('Hello World!')
        self['dy'] = val

    @property
    def hoverinfo(self):
        if False:
            print('Hello World!')
        "\n        Determines which trace information appear on hover. If `none`\n        or `skip` are set, no information is displayed upon hovering.\n        But, if `none` is set, click and hover events are still fired.\n\n        The 'hoverinfo' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['name', 'x', 'y', 'text', 'initial', 'delta', 'final'] joined with '+' characters\n            (e.g. 'name+x')\n            OR exactly one of ['all', 'none', 'skip'] (e.g. 'skip')\n          - A list or array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['hoverinfo']

    @hoverinfo.setter
    def hoverinfo(self, val):
        if False:
            return 10
        self['hoverinfo'] = val

    @property
    def hoverinfosrc(self):
        if False:
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
        "\n        The 'hoverlabel' property is an instance of Hoverlabel\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Hoverlabel`\n          - A dict of string/value properties that will be passed\n            to the Hoverlabel constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the text\n                    content within hover label box. Has an effect\n                    only if the hover label text spans more two or\n                    more lines\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                bgcolor\n                    Sets the background color of the hover labels\n                    for this trace\n                bgcolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bgcolor`.\n                bordercolor\n                    Sets the border color of the hover labels for\n                    this trace.\n                bordercolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bordercolor`.\n                font\n                    Sets the font used in hover labels.\n                namelength\n                    Sets the default length (in number of\n                    characters) of the trace name in the hover\n                    labels for all traces. -1 shows the whole name\n                    regardless of length. 0-3 shows the first 0-3\n                    characters, and an integer >3 will show the\n                    whole name if it is less than that many\n                    characters, but if it is longer, will truncate\n                    to `namelength - 3` characters and add an\n                    ellipsis.\n                namelengthsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `namelength`.\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Hoverlabel\n        "
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        if False:
            print('Hello World!')
        self['hoverlabel'] = val

    @property
    def hovertemplate(self):
        if False:
            i = 10
            return i + 15
        '\n        Template string used for rendering the information that appear\n        on hover box. Note that this will override `hoverinfo`.\n        Variables are inserted using %{variable}, for example "y: %{y}"\n        as well as %{xother}, {%_xother}, {%_xother_}, {%xother_}. When\n        showing info for several points, "xother" will be added to\n        those with different x positions from the first point. An\n        underscore before or after "(x|y)other" will add a space on\n        that side, only when this field is shown. Numbers are formatted\n        using d3-format\'s syntax %{variable:d3-format}, for example\n        "Price: %{y:$.2f}".\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for\n        details on the formatting syntax. Dates are formatted using\n        d3-time-format\'s syntax %{variable|d3-time-format}, for example\n        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format for details on the date\n        formatting syntax. The variables available in `hovertemplate`\n        are the ones emitted as event data described at this link\n        https://plotly.com/javascript/plotlyjs-events/#event-data.\n        Additionally, every attributes that can be specified per-point\n        (the ones that are `arrayOk: true`) are available. Finally, the\n        template string has access to variables `initial`, `delta` and\n        `final`. Anything contained in tag `<extra>` is displayed in\n        the secondary box, for example\n        "<extra>{fullData.name}</extra>". To hide the secondary box\n        completely, use an empty tag `<extra></extra>`.\n\n        The \'hovertemplate\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['hovertemplate']

    @hovertemplate.setter
    def hovertemplate(self, val):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
        self['hovertemplatesrc'] = val

    @property
    def hovertext(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets hover text elements associated with each (x,y) pair. If a\n        single string, the same string appears over all the data\n        points. If an array of string, the items are mapped in order to\n        the this trace\'s (x,y) coordinates. To be seen, trace\n        `hoverinfo` must contain a "text" flag.\n\n        The \'hovertext\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
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
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `ids`.\n\n        The 'idssrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['idssrc']

    @idssrc.setter
    def idssrc(self, val):
        if False:
            return 10
        self['idssrc'] = val

    @property
    def increasing(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'increasing' property is an instance of Increasing\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Increasing`\n          - A dict of string/value properties that will be passed\n            to the Increasing constructor\n\n            Supported dict properties:\n\n                marker\n                    :class:`plotly.graph_objects.waterfall.increasi\n                    ng.Marker` instance or dict with compatible\n                    properties\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Increasing\n        "
        return self['increasing']

    @increasing.setter
    def increasing(self, val):
        if False:
            return 10
        self['increasing'] = val

    @property
    def insidetextanchor(self):
        if False:
            return 10
        '\n        Determines if texts are kept at center or start/end points in\n        `textposition` "inside" mode.\n\n        The \'insidetextanchor\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'end\', \'middle\', \'start\']\n\n        Returns\n        -------\n        Any\n        '
        return self['insidetextanchor']

    @insidetextanchor.setter
    def insidetextanchor(self, val):
        if False:
            while True:
                i = 10
        self['insidetextanchor'] = val

    @property
    def insidetextfont(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the font used for `text` lying inside the bar.\n\n        The \'insidetextfont\' property is an instance of Insidetextfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Insidetextfont`\n          - A dict of string/value properties that will be passed\n            to the Insidetextfont constructor\n\n            Supported dict properties:\n\n                color\n\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                familysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `family`.\n                size\n\n                sizesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `size`.\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Insidetextfont\n        '
        return self['insidetextfont']

    @insidetextfont.setter
    def insidetextfont(self, val):
        if False:
            while True:
                i = 10
        self['insidetextfont'] = val

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
            while True:
                i = 10
        self['legend'] = val

    @property
    def legendgroup(self):
        if False:
            return 10
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
            print('Hello World!')
        "\n        The 'legendgrouptitle' property is an instance of Legendgrouptitle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Legendgrouptitle`\n          - A dict of string/value properties that will be passed\n            to the Legendgrouptitle constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend group's title font.\n                text\n                    Sets the title of the legend group.\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Legendgrouptitle\n        "
        return self['legendgrouptitle']

    @legendgrouptitle.setter
    def legendgrouptitle(self, val):
        if False:
            return 10
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
            for i in range(10):
                print('nop')
        self['legendrank'] = val

    @property
    def legendwidth(self):
        if False:
            print('Hello World!')
        "\n        Sets the width (in px or fraction) of the legend for this\n        trace.\n\n        The 'legendwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['legendwidth']

    @legendwidth.setter
    def legendwidth(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['legendwidth'] = val

    @property
    def measure(self):
        if False:
            print('Hello World!')
        "\n        An array containing types of values. By default the values are\n        considered as 'relative'. However; it is possible to use\n        'total' to compute the sums. Also 'absolute' could be applied\n        to reset the computed total or to declare an initial value\n        where needed.\n\n        The 'measure' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['measure']

    @measure.setter
    def measure(self, val):
        if False:
            return 10
        self['measure'] = val

    @property
    def measuresrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `measure`.\n\n        The 'measuresrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['measuresrc']

    @measuresrc.setter
    def measuresrc(self, val):
        if False:
            return 10
        self['measuresrc'] = val

    @property
    def meta(self):
        if False:
            return 10
        "\n        Assigns extra meta information associated with this trace that\n        can be used in various text attributes. Attributes such as\n        trace `name`, graph, axis and colorbar `title.text`, annotation\n        `text` `rangeselector`, `updatemenues` and `sliders` `label`\n        text all support `meta`. To access the trace `meta` values in\n        an attribute in the same trace, simply use `%{meta[i]}` where\n        `i` is the index or key of the `meta` item in question. To\n        access trace `meta` in layout attributes, use\n        `%{data[n[.meta[i]}` where `i` is the index or key of the\n        `meta` and `n` is the trace index.\n\n        The 'meta' property accepts values of any type\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['meta']

    @meta.setter
    def meta(self, val):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
        "\n        Sets the trace name. The trace name appears as the legend item\n        and on hover.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['name'] = val

    @property
    def offset(self):
        if False:
            return 10
        '\n        Shifts the position where the bar is drawn (in position axis\n        units). In "group" barmode, traces that set "offset" will be\n        excluded and drawn in "overlay" mode instead.\n\n        The \'offset\' property is a number and may be specified as:\n          - An int or float\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        int|float|numpy.ndarray\n        '
        return self['offset']

    @offset.setter
    def offset(self, val):
        if False:
            i = 10
            return i + 15
        self['offset'] = val

    @property
    def offsetgroup(self):
        if False:
            i = 10
            return i + 15
        "\n        Set several traces linked to the same position axis or matching\n        axes to the same offsetgroup where bars of the same position\n        coordinate will line up.\n\n        The 'offsetgroup' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['offsetgroup']

    @offsetgroup.setter
    def offsetgroup(self, val):
        if False:
            return 10
        self['offsetgroup'] = val

    @property
    def offsetsrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `offset`.\n\n        The 'offsetsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['offsetsrc']

    @offsetsrc.setter
    def offsetsrc(self, val):
        if False:
            print('Hello World!')
        self['offsetsrc'] = val

    @property
    def opacity(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the opacity of the trace.\n\n        The 'opacity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            while True:
                i = 10
        self['opacity'] = val

    @property
    def orientation(self):
        if False:
            print('Hello World!')
        '\n        Sets the orientation of the bars. With "v" ("h"), the value of\n        the each bar spans along the vertical (horizontal).\n\n        The \'orientation\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'v\', \'h\']\n\n        Returns\n        -------\n        Any\n        '
        return self['orientation']

    @orientation.setter
    def orientation(self, val):
        if False:
            while True:
                i = 10
        self['orientation'] = val

    @property
    def outsidetextfont(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the font used for `text` lying outside the bar.\n\n        The \'outsidetextfont\' property is an instance of Outsidetextfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Outsidetextfont`\n          - A dict of string/value properties that will be passed\n            to the Outsidetextfont constructor\n\n            Supported dict properties:\n\n                color\n\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                familysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `family`.\n                size\n\n                sizesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `size`.\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Outsidetextfont\n        '
        return self['outsidetextfont']

    @outsidetextfont.setter
    def outsidetextfont(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['outsidetextfont'] = val

    @property
    def selectedpoints(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Array containing integer indices of selected points. Has an\n        effect only for traces that support selections. Note that an\n        empty array means an empty selection where the `unselected` are\n        turned on for all points, whereas, any other non-array values\n        means no selection all where the `selected` and `unselected`\n        styles have no effect.\n\n        The 'selectedpoints' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['selectedpoints']

    @selectedpoints.setter
    def selectedpoints(self, val):
        if False:
            i = 10
            return i + 15
        self['selectedpoints'] = val

    @property
    def showlegend(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not an item corresponding to this trace\n        is shown in the legend.\n\n        The 'showlegend' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['showlegend']

    @showlegend.setter
    def showlegend(self, val):
        if False:
            while True:
                i = 10
        self['showlegend'] = val

    @property
    def stream(self):
        if False:
            while True:
                i = 10
        "\n        The 'stream' property is an instance of Stream\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Stream`\n          - A dict of string/value properties that will be passed\n            to the Stream constructor\n\n            Supported dict properties:\n\n                maxpoints\n                    Sets the maximum number of points to keep on\n                    the plots from an incoming stream. If\n                    `maxpoints` is set to 50, only the newest 50\n                    points will be displayed on the plot.\n                token\n                    The stream id number links a data trace on a\n                    plot with a stream. See https://chart-\n                    studio.plotly.com/settings for more details.\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Stream\n        "
        return self['stream']

    @stream.setter
    def stream(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['stream'] = val

    @property
    def text(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets text elements associated with each (x,y) pair. If a single\n        string, the same string appears over all the data points. If an\n        array of string, the items are mapped in order to the this\n        trace\'s (x,y) coordinates. If trace `hoverinfo` contains a\n        "text" flag and "hovertext" is not set, these elements will be\n        seen in the hover labels.\n\n        The \'text\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['text']

    @text.setter
    def text(self, val):
        if False:
            print('Hello World!')
        self['text'] = val

    @property
    def textangle(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the angle of the tick labels with respect to the bar. For\n        example, a `tickangle` of -90 draws the tick labels vertically.\n        With "auto" the texts may automatically be rotated to fit with\n        the maximum size in bars.\n\n        The \'textangle\' property is a angle (in degrees) that may be\n        specified as a number between -180 and 180.\n        Numeric values outside this range are converted to the equivalent value\n        (e.g. 270 is converted to -90).\n\n        Returns\n        -------\n        int|float\n        '
        return self['textangle']

    @textangle.setter
    def textangle(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['textangle'] = val

    @property
    def textfont(self):
        if False:
            print('Hello World!')
        '\n        Sets the font used for `text`.\n\n        The \'textfont\' property is an instance of Textfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Textfont`\n          - A dict of string/value properties that will be passed\n            to the Textfont constructor\n\n            Supported dict properties:\n\n                color\n\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                familysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `family`.\n                size\n\n                sizesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `size`.\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Textfont\n        '
        return self['textfont']

    @textfont.setter
    def textfont(self, val):
        if False:
            return 10
        self['textfont'] = val

    @property
    def textinfo(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines which trace information appear on the graph. In the\n        case of having multiple waterfalls, totals are computed\n        separately (per trace).\n\n        The 'textinfo' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['label', 'text', 'initial', 'delta', 'final'] joined with '+' characters\n            (e.g. 'label+text')\n            OR exactly one of ['none'] (e.g. 'none')\n\n        Returns\n        -------\n        Any\n        "
        return self['textinfo']

    @textinfo.setter
    def textinfo(self, val):
        if False:
            i = 10
            return i + 15
        self['textinfo'] = val

    @property
    def textposition(self):
        if False:
            return 10
        '\n        Specifies the location of the `text`. "inside" positions `text`\n        inside, next to the bar end (rotated and scaled if needed).\n        "outside" positions `text` outside, next to the bar end (scaled\n        if needed), unless there is another bar stacked on this one,\n        then the text gets pushed inside. "auto" tries to position\n        `text` inside the bar, but if the bar is too small and no bar\n        is stacked on this one the text is moved outside. If "none", no\n        text appears.\n\n        The \'textposition\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'inside\', \'outside\', \'auto\', \'none\']\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        '
        return self['textposition']

    @textposition.setter
    def textposition(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['textposition'] = val

    @property
    def textpositionsrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `textposition`.\n\n        The 'textpositionsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['textpositionsrc']

    @textpositionsrc.setter
    def textpositionsrc(self, val):
        if False:
            i = 10
            return i + 15
        self['textpositionsrc'] = val

    @property
    def textsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `text`.\n\n        The 'textsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['textsrc']

    @textsrc.setter
    def textsrc(self, val):
        if False:
            while True:
                i = 10
        self['textsrc'] = val

    @property
    def texttemplate(self):
        if False:
            return 10
        '\n        Template string used for rendering the information text that\n        appear on points. Note that this will override `textinfo`.\n        Variables are inserted using %{variable}, for example "y:\n        %{y}". Numbers are formatted using d3-format\'s syntax\n        %{variable:d3-format}, for example "Price: %{y:$.2f}".\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for\n        details on the formatting syntax. Dates are formatted using\n        d3-time-format\'s syntax %{variable|d3-time-format}, for example\n        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format for details on the date\n        formatting syntax. Every attributes that can be specified per-\n        point (the ones that are `arrayOk: true`) are available.\n        Finally, the template string has access to variables `initial`,\n        `delta`, `final` and `label`.\n\n        The \'texttemplate\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['texttemplate']

    @texttemplate.setter
    def texttemplate(self, val):
        if False:
            print('Hello World!')
        self['texttemplate'] = val

    @property
    def texttemplatesrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for\n        `texttemplate`.\n\n        The 'texttemplatesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['texttemplatesrc']

    @texttemplatesrc.setter
    def texttemplatesrc(self, val):
        if False:
            return 10
        self['texttemplatesrc'] = val

    @property
    def totals(self):
        if False:
            while True:
                i = 10
        "\n        The 'totals' property is an instance of Totals\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.Totals`\n          - A dict of string/value properties that will be passed\n            to the Totals constructor\n\n            Supported dict properties:\n\n                marker\n                    :class:`plotly.graph_objects.waterfall.totals.M\n                    arker` instance or dict with compatible\n                    properties\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.Totals\n        "
        return self['totals']

    @totals.setter
    def totals(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['totals'] = val

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
            i = 10
            return i + 15
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
    def visible(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines whether or not this trace is visible. If\n        "legendonly", the trace is not drawn, but can appear as a\n        legend item (provided that the legend itself is visible).\n\n        The \'visible\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, \'legendonly\']\n\n        Returns\n        -------\n        Any\n        '
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            while True:
                i = 10
        self['visible'] = val

    @property
    def width(self):
        if False:
            while True:
                i = 10
        "\n        Sets the bar width (in position axis units).\n\n        The 'width' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        int|float|numpy.ndarray\n        "
        return self['width']

    @width.setter
    def width(self, val):
        if False:
            i = 10
            return i + 15
        self['width'] = val

    @property
    def widthsrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `width`.\n\n        The 'widthsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['widthsrc']

    @widthsrc.setter
    def widthsrc(self, val):
        if False:
            print('Hello World!')
        self['widthsrc'] = val

    @property
    def x(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the x coordinates.\n\n        The 'x' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['x'] = val

    @property
    def x0(self):
        if False:
            print('Hello World!')
        "\n        Alternate to `x`. Builds a linear space of x coordinates. Use\n        with `dx` where `x0` is the starting coordinate and `dx` the\n        step.\n\n        The 'x0' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['x0']

    @x0.setter
    def x0(self, val):
        if False:
            print('Hello World!')
        self['x0'] = val

    @property
    def xaxis(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets a reference between this trace\'s x coordinates and a 2D\n        cartesian x axis. If "x" (the default value), the x coordinates\n        refer to `layout.xaxis`. If "x2", the x coordinates refer to\n        `layout.xaxis2`, and so on.\n\n        The \'xaxis\' property is an identifier of a particular\n        subplot, of type \'x\', that may be specified as the string \'x\'\n        optionally followed by an integer >= 1\n        (e.g. \'x\', \'x1\', \'x2\', \'x3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['xaxis']

    @xaxis.setter
    def xaxis(self, val):
        if False:
            print('Hello World!')
        self['xaxis'] = val

    @property
    def xhoverformat(self):
        if False:
            print('Hello World!')
        '\n        Sets the hover text formatting rulefor `x`  using d3 formatting\n        mini-languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display *09~15~23.46*By default the values\n        are formatted using `xaxis.hoverformat`.\n\n        The \'xhoverformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['xhoverformat']

    @xhoverformat.setter
    def xhoverformat(self, val):
        if False:
            i = 10
            return i + 15
        self['xhoverformat'] = val

    @property
    def xperiod(self):
        if False:
            return 10
        '\n        Only relevant when the axis `type` is "date". Sets the period\n        positioning in milliseconds or "M<n>" on the x axis. Special\n        values in the form of "M<n>" could be used to declare the\n        number of months. In this case `n` must be a positive integer.\n\n        The \'xperiod\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['xperiod']

    @xperiod.setter
    def xperiod(self, val):
        if False:
            i = 10
            return i + 15
        self['xperiod'] = val

    @property
    def xperiod0(self):
        if False:
            while True:
                i = 10
        '\n        Only relevant when the axis `type` is "date". Sets the base for\n        period positioning in milliseconds or date string on the x0\n        axis. When `x0period` is round number of weeks, the `x0period0`\n        by default would be on a Sunday i.e. 2000-01-02, otherwise it\n        would be at 2000-01-01.\n\n        The \'xperiod0\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['xperiod0']

    @xperiod0.setter
    def xperiod0(self, val):
        if False:
            return 10
        self['xperiod0'] = val

    @property
    def xperiodalignment(self):
        if False:
            print('Hello World!')
        '\n        Only relevant when the axis `type` is "date". Sets the\n        alignment of data points on the x axis.\n\n        The \'xperiodalignment\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'start\', \'middle\', \'end\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xperiodalignment']

    @xperiodalignment.setter
    def xperiodalignment(self, val):
        if False:
            return 10
        self['xperiodalignment'] = val

    @property
    def xsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `x`.\n\n        The 'xsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['xsrc']

    @xsrc.setter
    def xsrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['xsrc'] = val

    @property
    def y(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the y coordinates.\n\n        The 'y' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            print('Hello World!')
        self['y'] = val

    @property
    def y0(self):
        if False:
            print('Hello World!')
        "\n        Alternate to `y`. Builds a linear space of y coordinates. Use\n        with `dy` where `y0` is the starting coordinate and `dy` the\n        step.\n\n        The 'y0' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['y0']

    @y0.setter
    def y0(self, val):
        if False:
            while True:
                i = 10
        self['y0'] = val

    @property
    def yaxis(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets a reference between this trace\'s y coordinates and a 2D\n        cartesian y axis. If "y" (the default value), the y coordinates\n        refer to `layout.yaxis`. If "y2", the y coordinates refer to\n        `layout.yaxis2`, and so on.\n\n        The \'yaxis\' property is an identifier of a particular\n        subplot, of type \'y\', that may be specified as the string \'y\'\n        optionally followed by an integer >= 1\n        (e.g. \'y\', \'y1\', \'y2\', \'y3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['yaxis']

    @yaxis.setter
    def yaxis(self, val):
        if False:
            i = 10
            return i + 15
        self['yaxis'] = val

    @property
    def yhoverformat(self):
        if False:
            while True:
                i = 10
        '\n        Sets the hover text formatting rulefor `y`  using d3 formatting\n        mini-languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for\n        dates see: https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format. We add two items to d3\'s date\n        formatter: "%h" for half of the year as a decimal number as\n        well as "%{n}f" for fractional seconds with n digits. For\n        example, *2016-10-13 09:15:23.456* with tickformat\n        "%H~%M~%S.%2f" would display *09~15~23.46*By default the values\n        are formatted using `yaxis.hoverformat`.\n\n        The \'yhoverformat\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['yhoverformat']

    @yhoverformat.setter
    def yhoverformat(self, val):
        if False:
            return 10
        self['yhoverformat'] = val

    @property
    def yperiod(self):
        if False:
            while True:
                i = 10
        '\n        Only relevant when the axis `type` is "date". Sets the period\n        positioning in milliseconds or "M<n>" on the y axis. Special\n        values in the form of "M<n>" could be used to declare the\n        number of months. In this case `n` must be a positive integer.\n\n        The \'yperiod\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['yperiod']

    @yperiod.setter
    def yperiod(self, val):
        if False:
            print('Hello World!')
        self['yperiod'] = val

    @property
    def yperiod0(self):
        if False:
            return 10
        '\n        Only relevant when the axis `type` is "date". Sets the base for\n        period positioning in milliseconds or date string on the y0\n        axis. When `y0period` is round number of weeks, the `y0period0`\n        by default would be on a Sunday i.e. 2000-01-02, otherwise it\n        would be at 2000-01-01.\n\n        The \'yperiod0\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['yperiod0']

    @yperiod0.setter
    def yperiod0(self, val):
        if False:
            print('Hello World!')
        self['yperiod0'] = val

    @property
    def yperiodalignment(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Only relevant when the axis `type` is "date". Sets the\n        alignment of data points on the y axis.\n\n        The \'yperiodalignment\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'start\', \'middle\', \'end\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yperiodalignment']

    @yperiodalignment.setter
    def yperiodalignment(self, val):
        if False:
            return 10
        self['yperiodalignment'] = val

    @property
    def ysrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `y`.\n\n        The 'ysrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['ysrc']

    @ysrc.setter
    def ysrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['ysrc'] = val

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        return self._props['type']

    @property
    def _prop_descriptions(self):
        if False:
            i = 10
            return i + 15
        return '        alignmentgroup\n            Set several traces linked to the same position axis or\n            matching axes to the same alignmentgroup. This controls\n            whether bars compute their positional range dependently\n            or independently.\n        base\n            Sets where the bar base is drawn (in position axis\n            units).\n        cliponaxis\n            Determines whether the text nodes are clipped about the\n            subplot axes. To show the text nodes above axis lines\n            and tick labels, make sure to set `xaxis.layer` and\n            `yaxis.layer` to *below traces*.\n        connector\n            :class:`plotly.graph_objects.waterfall.Connector`\n            instance or dict with compatible properties\n        constraintext\n            Constrain the size of text inside or outside a bar to\n            be no larger than the bar itself.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        decreasing\n            :class:`plotly.graph_objects.waterfall.Decreasing`\n            instance or dict with compatible properties\n        dx\n            Sets the x coordinate step. See `x0` for more info.\n        dy\n            Sets the y coordinate step. See `y0` for more info.\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.waterfall.Hoverlabel`\n            instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `initial`, `delta` and `final`. Anything\n            contained in tag `<extra>` is displayed in the\n            secondary box, for example\n            "<extra>{fullData.name}</extra>". To hide the secondary\n            box completely, use an empty tag `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Sets hover text elements associated with each (x,y)\n            pair. If a single string, the same string appears over\n            all the data points. If an array of string, the items\n            are mapped in order to the this trace\'s (x,y)\n            coordinates. To be seen, trace `hoverinfo` must contain\n            a "text" flag.\n        hovertextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertext`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        increasing\n            :class:`plotly.graph_objects.waterfall.Increasing`\n            instance or dict with compatible properties\n        insidetextanchor\n            Determines if texts are kept at center or start/end\n            points in `textposition` "inside" mode.\n        insidetextfont\n            Sets the font used for `text` lying inside the bar.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.waterfall.Legendgrouptitle\n            ` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        measure\n            An array containing types of values. By default the\n            values are considered as \'relative\'. However; it is\n            possible to use \'total\' to compute the sums. Also\n            \'absolute\' could be applied to reset the computed total\n            or to declare an initial value where needed.\n        measuresrc\n            Sets the source reference on Chart Studio Cloud for\n            `measure`.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        offset\n            Shifts the position where the bar is drawn (in position\n            axis units). In "group" barmode, traces that set\n            "offset" will be excluded and drawn in "overlay" mode\n            instead.\n        offsetgroup\n            Set several traces linked to the same position axis or\n            matching axes to the same offsetgroup where bars of the\n            same position coordinate will line up.\n        offsetsrc\n            Sets the source reference on Chart Studio Cloud for\n            `offset`.\n        opacity\n            Sets the opacity of the trace.\n        orientation\n            Sets the orientation of the bars. With "v" ("h"), the\n            value of the each bar spans along the vertical\n            (horizontal).\n        outsidetextfont\n            Sets the font used for `text` lying outside the bar.\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        stream\n            :class:`plotly.graph_objects.waterfall.Stream` instance\n            or dict with compatible properties\n        text\n            Sets text elements associated with each (x,y) pair. If\n            a single string, the same string appears over all the\n            data points. If an array of string, the items are\n            mapped in order to the this trace\'s (x,y) coordinates.\n            If trace `hoverinfo` contains a "text" flag and\n            "hovertext" is not set, these elements will be seen in\n            the hover labels.\n        textangle\n            Sets the angle of the tick labels with respect to the\n            bar. For example, a `tickangle` of -90 draws the tick\n            labels vertically. With "auto" the texts may\n            automatically be rotated to fit with the maximum size\n            in bars.\n        textfont\n            Sets the font used for `text`.\n        textinfo\n            Determines which trace information appear on the graph.\n            In the case of having multiple waterfalls, totals are\n            computed separately (per trace).\n        textposition\n            Specifies the location of the `text`. "inside"\n            positions `text` inside, next to the bar end (rotated\n            and scaled if needed). "outside" positions `text`\n            outside, next to the bar end (scaled if needed), unless\n            there is another bar stacked on this one, then the text\n            gets pushed inside. "auto" tries to position `text`\n            inside the bar, but if the bar is too small and no bar\n            is stacked on this one the text is moved outside. If\n            "none", no text appears.\n        textpositionsrc\n            Sets the source reference on Chart Studio Cloud for\n            `textposition`.\n        textsrc\n            Sets the source reference on Chart Studio Cloud for\n            `text`.\n        texttemplate\n            Template string used for rendering the information text\n            that appear on points. Note that this will override\n            `textinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}". Numbers are formatted using\n            d3-format\'s syntax %{variable:d3-format}, for example\n            "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. Every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `initial`, `delta`, `final` and `label`.\n        texttemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `texttemplate`.\n        totals\n            :class:`plotly.graph_objects.waterfall.Totals` instance\n            or dict with compatible properties\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        width\n            Sets the bar width (in position axis units).\n        widthsrc\n            Sets the source reference on Chart Studio Cloud for\n            `width`.\n        x\n            Sets the x coordinates.\n        x0\n            Alternate to `x`. Builds a linear space of x\n            coordinates. Use with `dx` where `x0` is the starting\n            coordinate and `dx` the step.\n        xaxis\n            Sets a reference between this trace\'s x coordinates and\n            a 2D cartesian x axis. If "x" (the default value), the\n            x coordinates refer to `layout.xaxis`. If "x2", the x\n            coordinates refer to `layout.xaxis2`, and so on.\n        xhoverformat\n            Sets the hover text formatting rulefor `x`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `xaxis.hoverformat`.\n        xperiod\n            Only relevant when the axis `type` is "date". Sets the\n            period positioning in milliseconds or "M<n>" on the x\n            axis. Special values in the form of "M<n>" could be\n            used to declare the number of months. In this case `n`\n            must be a positive integer.\n        xperiod0\n            Only relevant when the axis `type` is "date". Sets the\n            base for period positioning in milliseconds or date\n            string on the x0 axis. When `x0period` is round number\n            of weeks, the `x0period0` by default would be on a\n            Sunday i.e. 2000-01-02, otherwise it would be at\n            2000-01-01.\n        xperiodalignment\n            Only relevant when the axis `type` is "date". Sets the\n            alignment of data points on the x axis.\n        xsrc\n            Sets the source reference on Chart Studio Cloud for\n            `x`.\n        y\n            Sets the y coordinates.\n        y0\n            Alternate to `y`. Builds a linear space of y\n            coordinates. Use with `dy` where `y0` is the starting\n            coordinate and `dy` the step.\n        yaxis\n            Sets a reference between this trace\'s y coordinates and\n            a 2D cartesian y axis. If "y" (the default value), the\n            y coordinates refer to `layout.yaxis`. If "y2", the y\n            coordinates refer to `layout.yaxis2`, and so on.\n        yhoverformat\n            Sets the hover text formatting rulefor `y`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `yaxis.hoverformat`.\n        yperiod\n            Only relevant when the axis `type` is "date". Sets the\n            period positioning in milliseconds or "M<n>" on the y\n            axis. Special values in the form of "M<n>" could be\n            used to declare the number of months. In this case `n`\n            must be a positive integer.\n        yperiod0\n            Only relevant when the axis `type` is "date". Sets the\n            base for period positioning in milliseconds or date\n            string on the y0 axis. When `y0period` is round number\n            of weeks, the `y0period0` by default would be on a\n            Sunday i.e. 2000-01-02, otherwise it would be at\n            2000-01-01.\n        yperiodalignment\n            Only relevant when the axis `type` is "date". Sets the\n            alignment of data points on the y axis.\n        ysrc\n            Sets the source reference on Chart Studio Cloud for\n            `y`.\n        '

    def __init__(self, arg=None, alignmentgroup=None, base=None, cliponaxis=None, connector=None, constraintext=None, customdata=None, customdatasrc=None, decreasing=None, dx=None, dy=None, hoverinfo=None, hoverinfosrc=None, hoverlabel=None, hovertemplate=None, hovertemplatesrc=None, hovertext=None, hovertextsrc=None, ids=None, idssrc=None, increasing=None, insidetextanchor=None, insidetextfont=None, legend=None, legendgroup=None, legendgrouptitle=None, legendrank=None, legendwidth=None, measure=None, measuresrc=None, meta=None, metasrc=None, name=None, offset=None, offsetgroup=None, offsetsrc=None, opacity=None, orientation=None, outsidetextfont=None, selectedpoints=None, showlegend=None, stream=None, text=None, textangle=None, textfont=None, textinfo=None, textposition=None, textpositionsrc=None, textsrc=None, texttemplate=None, texttemplatesrc=None, totals=None, uid=None, uirevision=None, visible=None, width=None, widthsrc=None, x=None, x0=None, xaxis=None, xhoverformat=None, xperiod=None, xperiod0=None, xperiodalignment=None, xsrc=None, y=None, y0=None, yaxis=None, yhoverformat=None, yperiod=None, yperiod0=None, yperiodalignment=None, ysrc=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Waterfall object\n\n        Draws waterfall trace which is useful graph to displays the\n        contribution of various elements (either positive or negative)\n        in a bar chart. The data visualized by the span of the bars is\n        set in `y` if `orientation` is set to "v" (the default) and the\n        labels are set in `x`. By setting `orientation` to "h", the\n        roles are interchanged.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.Waterfall`\n        alignmentgroup\n            Set several traces linked to the same position axis or\n            matching axes to the same alignmentgroup. This controls\n            whether bars compute their positional range dependently\n            or independently.\n        base\n            Sets where the bar base is drawn (in position axis\n            units).\n        cliponaxis\n            Determines whether the text nodes are clipped about the\n            subplot axes. To show the text nodes above axis lines\n            and tick labels, make sure to set `xaxis.layer` and\n            `yaxis.layer` to *below traces*.\n        connector\n            :class:`plotly.graph_objects.waterfall.Connector`\n            instance or dict with compatible properties\n        constraintext\n            Constrain the size of text inside or outside a bar to\n            be no larger than the bar itself.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        decreasing\n            :class:`plotly.graph_objects.waterfall.Decreasing`\n            instance or dict with compatible properties\n        dx\n            Sets the x coordinate step. See `x0` for more info.\n        dy\n            Sets the y coordinate step. See `y0` for more info.\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.waterfall.Hoverlabel`\n            instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `initial`, `delta` and `final`. Anything\n            contained in tag `<extra>` is displayed in the\n            secondary box, for example\n            "<extra>{fullData.name}</extra>". To hide the secondary\n            box completely, use an empty tag `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        hovertext\n            Sets hover text elements associated with each (x,y)\n            pair. If a single string, the same string appears over\n            all the data points. If an array of string, the items\n            are mapped in order to the this trace\'s (x,y)\n            coordinates. To be seen, trace `hoverinfo` must contain\n            a "text" flag.\n        hovertextsrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertext`.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        increasing\n            :class:`plotly.graph_objects.waterfall.Increasing`\n            instance or dict with compatible properties\n        insidetextanchor\n            Determines if texts are kept at center or start/end\n            points in `textposition` "inside" mode.\n        insidetextfont\n            Sets the font used for `text` lying inside the bar.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgroup\n            Sets the legend group for this trace. Traces and shapes\n            part of the same legend group hide/show at the same\n            time when toggling legend items.\n        legendgrouptitle\n            :class:`plotly.graph_objects.waterfall.Legendgrouptitle\n            ` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        measure\n            An array containing types of values. By default the\n            values are considered as \'relative\'. However; it is\n            possible to use \'total\' to compute the sums. Also\n            \'absolute\' could be applied to reset the computed total\n            or to declare an initial value where needed.\n        measuresrc\n            Sets the source reference on Chart Studio Cloud for\n            `measure`.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        offset\n            Shifts the position where the bar is drawn (in position\n            axis units). In "group" barmode, traces that set\n            "offset" will be excluded and drawn in "overlay" mode\n            instead.\n        offsetgroup\n            Set several traces linked to the same position axis or\n            matching axes to the same offsetgroup where bars of the\n            same position coordinate will line up.\n        offsetsrc\n            Sets the source reference on Chart Studio Cloud for\n            `offset`.\n        opacity\n            Sets the opacity of the trace.\n        orientation\n            Sets the orientation of the bars. With "v" ("h"), the\n            value of the each bar spans along the vertical\n            (horizontal).\n        outsidetextfont\n            Sets the font used for `text` lying outside the bar.\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        showlegend\n            Determines whether or not an item corresponding to this\n            trace is shown in the legend.\n        stream\n            :class:`plotly.graph_objects.waterfall.Stream` instance\n            or dict with compatible properties\n        text\n            Sets text elements associated with each (x,y) pair. If\n            a single string, the same string appears over all the\n            data points. If an array of string, the items are\n            mapped in order to the this trace\'s (x,y) coordinates.\n            If trace `hoverinfo` contains a "text" flag and\n            "hovertext" is not set, these elements will be seen in\n            the hover labels.\n        textangle\n            Sets the angle of the tick labels with respect to the\n            bar. For example, a `tickangle` of -90 draws the tick\n            labels vertically. With "auto" the texts may\n            automatically be rotated to fit with the maximum size\n            in bars.\n        textfont\n            Sets the font used for `text`.\n        textinfo\n            Determines which trace information appear on the graph.\n            In the case of having multiple waterfalls, totals are\n            computed separately (per trace).\n        textposition\n            Specifies the location of the `text`. "inside"\n            positions `text` inside, next to the bar end (rotated\n            and scaled if needed). "outside" positions `text`\n            outside, next to the bar end (scaled if needed), unless\n            there is another bar stacked on this one, then the text\n            gets pushed inside. "auto" tries to position `text`\n            inside the bar, but if the bar is too small and no bar\n            is stacked on this one the text is moved outside. If\n            "none", no text appears.\n        textpositionsrc\n            Sets the source reference on Chart Studio Cloud for\n            `textposition`.\n        textsrc\n            Sets the source reference on Chart Studio Cloud for\n            `text`.\n        texttemplate\n            Template string used for rendering the information text\n            that appear on points. Note that this will override\n            `textinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}". Numbers are formatted using\n            d3-format\'s syntax %{variable:d3-format}, for example\n            "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. Every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available. Finally, the template string has access\n            to variables `initial`, `delta`, `final` and `label`.\n        texttemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `texttemplate`.\n        totals\n            :class:`plotly.graph_objects.waterfall.Totals` instance\n            or dict with compatible properties\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        width\n            Sets the bar width (in position axis units).\n        widthsrc\n            Sets the source reference on Chart Studio Cloud for\n            `width`.\n        x\n            Sets the x coordinates.\n        x0\n            Alternate to `x`. Builds a linear space of x\n            coordinates. Use with `dx` where `x0` is the starting\n            coordinate and `dx` the step.\n        xaxis\n            Sets a reference between this trace\'s x coordinates and\n            a 2D cartesian x axis. If "x" (the default value), the\n            x coordinates refer to `layout.xaxis`. If "x2", the x\n            coordinates refer to `layout.xaxis2`, and so on.\n        xhoverformat\n            Sets the hover text formatting rulefor `x`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `xaxis.hoverformat`.\n        xperiod\n            Only relevant when the axis `type` is "date". Sets the\n            period positioning in milliseconds or "M<n>" on the x\n            axis. Special values in the form of "M<n>" could be\n            used to declare the number of months. In this case `n`\n            must be a positive integer.\n        xperiod0\n            Only relevant when the axis `type` is "date". Sets the\n            base for period positioning in milliseconds or date\n            string on the x0 axis. When `x0period` is round number\n            of weeks, the `x0period0` by default would be on a\n            Sunday i.e. 2000-01-02, otherwise it would be at\n            2000-01-01.\n        xperiodalignment\n            Only relevant when the axis `type` is "date". Sets the\n            alignment of data points on the x axis.\n        xsrc\n            Sets the source reference on Chart Studio Cloud for\n            `x`.\n        y\n            Sets the y coordinates.\n        y0\n            Alternate to `y`. Builds a linear space of y\n            coordinates. Use with `dy` where `y0` is the starting\n            coordinate and `dy` the step.\n        yaxis\n            Sets a reference between this trace\'s y coordinates and\n            a 2D cartesian y axis. If "y" (the default value), the\n            y coordinates refer to `layout.yaxis`. If "y2", the y\n            coordinates refer to `layout.yaxis2`, and so on.\n        yhoverformat\n            Sets the hover text formatting rulefor `y`  using d3\n            formatting mini-languages which are very similar to\n            those in Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n            And for dates see: https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format. We add two items to\n            d3\'s date formatter: "%h" for half of the year as a\n            decimal number as well as "%{n}f" for fractional\n            seconds with n digits. For example, *2016-10-13\n            09:15:23.456* with tickformat "%H~%M~%S.%2f" would\n            display *09~15~23.46*By default the values are\n            formatted using `yaxis.hoverformat`.\n        yperiod\n            Only relevant when the axis `type` is "date". Sets the\n            period positioning in milliseconds or "M<n>" on the y\n            axis. Special values in the form of "M<n>" could be\n            used to declare the number of months. In this case `n`\n            must be a positive integer.\n        yperiod0\n            Only relevant when the axis `type` is "date". Sets the\n            base for period positioning in milliseconds or date\n            string on the y0 axis. When `y0period` is round number\n            of weeks, the `y0period0` by default would be on a\n            Sunday i.e. 2000-01-02, otherwise it would be at\n            2000-01-01.\n        yperiodalignment\n            Only relevant when the axis `type` is "date". Sets the\n            alignment of data points on the y axis.\n        ysrc\n            Sets the source reference on Chart Studio Cloud for\n            `y`.\n\n        Returns\n        -------\n        Waterfall\n        '
        super(Waterfall, self).__init__('waterfall')
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
            raise ValueError('The first argument to the plotly.graph_objs.Waterfall\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Waterfall`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('alignmentgroup', None)
        _v = alignmentgroup if alignmentgroup is not None else _v
        if _v is not None:
            self['alignmentgroup'] = _v
        _v = arg.pop('base', None)
        _v = base if base is not None else _v
        if _v is not None:
            self['base'] = _v
        _v = arg.pop('cliponaxis', None)
        _v = cliponaxis if cliponaxis is not None else _v
        if _v is not None:
            self['cliponaxis'] = _v
        _v = arg.pop('connector', None)
        _v = connector if connector is not None else _v
        if _v is not None:
            self['connector'] = _v
        _v = arg.pop('constraintext', None)
        _v = constraintext if constraintext is not None else _v
        if _v is not None:
            self['constraintext'] = _v
        _v = arg.pop('customdata', None)
        _v = customdata if customdata is not None else _v
        if _v is not None:
            self['customdata'] = _v
        _v = arg.pop('customdatasrc', None)
        _v = customdatasrc if customdatasrc is not None else _v
        if _v is not None:
            self['customdatasrc'] = _v
        _v = arg.pop('decreasing', None)
        _v = decreasing if decreasing is not None else _v
        if _v is not None:
            self['decreasing'] = _v
        _v = arg.pop('dx', None)
        _v = dx if dx is not None else _v
        if _v is not None:
            self['dx'] = _v
        _v = arg.pop('dy', None)
        _v = dy if dy is not None else _v
        if _v is not None:
            self['dy'] = _v
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
        _v = arg.pop('increasing', None)
        _v = increasing if increasing is not None else _v
        if _v is not None:
            self['increasing'] = _v
        _v = arg.pop('insidetextanchor', None)
        _v = insidetextanchor if insidetextanchor is not None else _v
        if _v is not None:
            self['insidetextanchor'] = _v
        _v = arg.pop('insidetextfont', None)
        _v = insidetextfont if insidetextfont is not None else _v
        if _v is not None:
            self['insidetextfont'] = _v
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
        _v = arg.pop('measure', None)
        _v = measure if measure is not None else _v
        if _v is not None:
            self['measure'] = _v
        _v = arg.pop('measuresrc', None)
        _v = measuresrc if measuresrc is not None else _v
        if _v is not None:
            self['measuresrc'] = _v
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
        _v = arg.pop('offset', None)
        _v = offset if offset is not None else _v
        if _v is not None:
            self['offset'] = _v
        _v = arg.pop('offsetgroup', None)
        _v = offsetgroup if offsetgroup is not None else _v
        if _v is not None:
            self['offsetgroup'] = _v
        _v = arg.pop('offsetsrc', None)
        _v = offsetsrc if offsetsrc is not None else _v
        if _v is not None:
            self['offsetsrc'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        _v = arg.pop('orientation', None)
        _v = orientation if orientation is not None else _v
        if _v is not None:
            self['orientation'] = _v
        _v = arg.pop('outsidetextfont', None)
        _v = outsidetextfont if outsidetextfont is not None else _v
        if _v is not None:
            self['outsidetextfont'] = _v
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
        _v = arg.pop('textangle', None)
        _v = textangle if textangle is not None else _v
        if _v is not None:
            self['textangle'] = _v
        _v = arg.pop('textfont', None)
        _v = textfont if textfont is not None else _v
        if _v is not None:
            self['textfont'] = _v
        _v = arg.pop('textinfo', None)
        _v = textinfo if textinfo is not None else _v
        if _v is not None:
            self['textinfo'] = _v
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
        _v = arg.pop('totals', None)
        _v = totals if totals is not None else _v
        if _v is not None:
            self['totals'] = _v
        _v = arg.pop('uid', None)
        _v = uid if uid is not None else _v
        if _v is not None:
            self['uid'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        _v = arg.pop('width', None)
        _v = width if width is not None else _v
        if _v is not None:
            self['width'] = _v
        _v = arg.pop('widthsrc', None)
        _v = widthsrc if widthsrc is not None else _v
        if _v is not None:
            self['widthsrc'] = _v
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('x0', None)
        _v = x0 if x0 is not None else _v
        if _v is not None:
            self['x0'] = _v
        _v = arg.pop('xaxis', None)
        _v = xaxis if xaxis is not None else _v
        if _v is not None:
            self['xaxis'] = _v
        _v = arg.pop('xhoverformat', None)
        _v = xhoverformat if xhoverformat is not None else _v
        if _v is not None:
            self['xhoverformat'] = _v
        _v = arg.pop('xperiod', None)
        _v = xperiod if xperiod is not None else _v
        if _v is not None:
            self['xperiod'] = _v
        _v = arg.pop('xperiod0', None)
        _v = xperiod0 if xperiod0 is not None else _v
        if _v is not None:
            self['xperiod0'] = _v
        _v = arg.pop('xperiodalignment', None)
        _v = xperiodalignment if xperiodalignment is not None else _v
        if _v is not None:
            self['xperiodalignment'] = _v
        _v = arg.pop('xsrc', None)
        _v = xsrc if xsrc is not None else _v
        if _v is not None:
            self['xsrc'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('y0', None)
        _v = y0 if y0 is not None else _v
        if _v is not None:
            self['y0'] = _v
        _v = arg.pop('yaxis', None)
        _v = yaxis if yaxis is not None else _v
        if _v is not None:
            self['yaxis'] = _v
        _v = arg.pop('yhoverformat', None)
        _v = yhoverformat if yhoverformat is not None else _v
        if _v is not None:
            self['yhoverformat'] = _v
        _v = arg.pop('yperiod', None)
        _v = yperiod if yperiod is not None else _v
        if _v is not None:
            self['yperiod'] = _v
        _v = arg.pop('yperiod0', None)
        _v = yperiod0 if yperiod0 is not None else _v
        if _v is not None:
            self['yperiod0'] = _v
        _v = arg.pop('yperiodalignment', None)
        _v = yperiodalignment if yperiodalignment is not None else _v
        if _v is not None:
            self['yperiodalignment'] = _v
        _v = arg.pop('ysrc', None)
        _v = ysrc if ysrc is not None else _v
        if _v is not None:
            self['ysrc'] = _v
        self._props['type'] = 'waterfall'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
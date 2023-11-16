from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy

class Sankey(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'sankey'
    _valid_props = {'arrangement', 'customdata', 'customdatasrc', 'domain', 'hoverinfo', 'hoverlabel', 'ids', 'idssrc', 'legend', 'legendgrouptitle', 'legendrank', 'legendwidth', 'link', 'meta', 'metasrc', 'name', 'node', 'orientation', 'selectedpoints', 'stream', 'textfont', 'type', 'uid', 'uirevision', 'valueformat', 'valuesuffix', 'visible'}

    @property
    def arrangement(self):
        if False:
            return 10
        "\n        If value is `snap` (the default), the node arrangement is\n        assisted by automatic snapping of elements to preserve space\n        between nodes specified via `nodepad`. If value is\n        `perpendicular`, the nodes can only move along a line\n        perpendicular to the flow. If value is `freeform`, the nodes\n        can freely move on the plane. If value is `fixed`, the nodes\n        are stationary.\n\n        The 'arrangement' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['snap', 'perpendicular', 'freeform', 'fixed']\n\n        Returns\n        -------\n        Any\n        "
        return self['arrangement']

    @arrangement.setter
    def arrangement(self, val):
        if False:
            return 10
        self['arrangement'] = val

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
            i = 10
            return i + 15
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
            while True:
                i = 10
        self['customdatasrc'] = val

    @property
    def domain(self):
        if False:
            return 10
        "\n        The 'domain' property is an instance of Domain\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.Domain`\n          - A dict of string/value properties that will be passed\n            to the Domain constructor\n\n            Supported dict properties:\n\n                column\n                    If there is a layout grid, use the domain for\n                    this column in the grid for this sankey trace .\n                row\n                    If there is a layout grid, use the domain for\n                    this row in the grid for this sankey trace .\n                x\n                    Sets the horizontal domain of this sankey trace\n                    (in plot fraction).\n                y\n                    Sets the vertical domain of this sankey trace\n                    (in plot fraction).\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.Domain\n        "
        return self['domain']

    @domain.setter
    def domain(self, val):
        if False:
            i = 10
            return i + 15
        self['domain'] = val

    @property
    def hoverinfo(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines which trace information appear on hover. If `none`\n        or `skip` are set, no information is displayed upon hovering.\n        But, if `none` is set, click and hover events are still fired.\n        Note that this attribute is superseded by `node.hoverinfo` and\n        `node.hoverinfo` for nodes and links respectively.\n\n        The 'hoverinfo' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of [] joined with '+' characters\n            (e.g. '')\n            OR exactly one of ['all', 'none', 'skip'] (e.g. 'skip')\n\n        Returns\n        -------\n        Any\n        "
        return self['hoverinfo']

    @hoverinfo.setter
    def hoverinfo(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['hoverinfo'] = val

    @property
    def hoverlabel(self):
        if False:
            print('Hello World!')
        "\n        The 'hoverlabel' property is an instance of Hoverlabel\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.Hoverlabel`\n          - A dict of string/value properties that will be passed\n            to the Hoverlabel constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the text\n                    content within hover label box. Has an effect\n                    only if the hover label text spans more two or\n                    more lines\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                bgcolor\n                    Sets the background color of the hover labels\n                    for this trace\n                bgcolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bgcolor`.\n                bordercolor\n                    Sets the border color of the hover labels for\n                    this trace.\n                bordercolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bordercolor`.\n                font\n                    Sets the font used in hover labels.\n                namelength\n                    Sets the default length (in number of\n                    characters) of the trace name in the hover\n                    labels for all traces. -1 shows the whole name\n                    regardless of length. 0-3 shows the first 0-3\n                    characters, and an integer >3 will show the\n                    whole name if it is less than that many\n                    characters, but if it is longer, will truncate\n                    to `namelength - 3` characters and add an\n                    ellipsis.\n                namelengthsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `namelength`.\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.Hoverlabel\n        "
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        if False:
            return 10
        self['hoverlabel'] = val

    @property
    def ids(self):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
        '\n        Sets the reference to a legend to show this trace in.\n        References to these legends are "legend", "legend2", "legend3",\n        etc. Settings for these legends are set in the layout, under\n        `layout.legend`, `layout.legend2`, etc.\n\n        The \'legend\' property is an identifier of a particular\n        subplot, of type \'legend\', that may be specified as the string \'legend\'\n        optionally followed by an integer >= 1\n        (e.g. \'legend\', \'legend1\', \'legend2\', \'legend3\', etc.)\n\n        Returns\n        -------\n        str\n        '
        return self['legend']

    @legend.setter
    def legend(self, val):
        if False:
            print('Hello World!')
        self['legend'] = val

    @property
    def legendgrouptitle(self):
        if False:
            while True:
                i = 10
        "\n        The 'legendgrouptitle' property is an instance of Legendgrouptitle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.Legendgrouptitle`\n          - A dict of string/value properties that will be passed\n            to the Legendgrouptitle constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend group's title font.\n                text\n                    Sets the title of the legend group.\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.Legendgrouptitle\n        "
        return self['legendgrouptitle']

    @legendgrouptitle.setter
    def legendgrouptitle(self, val):
        if False:
            print('Hello World!')
        self['legendgrouptitle'] = val

    @property
    def legendrank(self):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
        "\n        Sets the width (in px or fraction) of the legend for this\n        trace.\n\n        The 'legendwidth' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['legendwidth']

    @legendwidth.setter
    def legendwidth(self, val):
        if False:
            while True:
                i = 10
        self['legendwidth'] = val

    @property
    def link(self):
        if False:
            while True:
                i = 10
        '\n        The links of the Sankey plot.\n\n        The \'link\' property is an instance of Link\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.Link`\n          - A dict of string/value properties that will be passed\n            to the Link constructor\n\n            Supported dict properties:\n\n                arrowlen\n                    Sets the length (in px) of the links arrow, if\n                    0 no arrow will be drawn.\n                color\n                    Sets the `link` color. It can be a single\n                    value, or an array for specifying color for\n                    each `link`. If `link.color` is omitted, then\n                    by default, a translucent grey link will be\n                    used.\n                colorscales\n                    A tuple of :class:`plotly.graph_objects.sankey.\n                    link.Colorscale` instances or dicts with\n                    compatible properties\n                colorscaledefaults\n                    When used in a template (as layout.template.dat\n                    a.sankey.link.colorscaledefaults), sets the\n                    default property values to use for elements of\n                    sankey.link.colorscales\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                customdata\n                    Assigns extra data to each link.\n                customdatasrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `customdata`.\n                hoverinfo\n                    Determines which trace information appear when\n                    hovering links. If `none` or `skip` are set, no\n                    information is displayed upon hovering. But, if\n                    `none` is set, click and hover events are still\n                    fired.\n                hoverlabel\n                    :class:`plotly.graph_objects.sankey.link.Hoverl\n                    abel` instance or dict with compatible\n                    properties\n                hovertemplate\n                    Template string used for rendering the\n                    information that appear on hover box. Note that\n                    this will override `hoverinfo`. Variables are\n                    inserted using %{variable}, for example "y:\n                    %{y}" as well as %{xother}, {%_xother},\n                    {%_xother_}, {%xother_}. When showing info for\n                    several points, "xother" will be added to those\n                    with different x positions from the first\n                    point. An underscore before or after\n                    "(x|y)other" will add a space on that side,\n                    only when this field is shown. Numbers are\n                    formatted using d3-format\'s syntax\n                    %{variable:d3-format}, for example "Price:\n                    %{y:$.2f}". https://github.com/d3/d3-\n                    format/tree/v1.4.5#d3-format for details on the\n                    formatting syntax. Dates are formatted using\n                    d3-time-format\'s syntax %{variable|d3-time-\n                    format}, for example "Day: %{2019-01-01|%A}".\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format for details on\n                    the date formatting syntax. The variables\n                    available in `hovertemplate` are the ones\n                    emitted as event data described at this link\n                    https://plotly.com/javascript/plotlyjs-\n                    events/#event-data. Additionally, every\n                    attributes that can be specified per-point (the\n                    ones that are `arrayOk: true`) are available.\n                    Variables `source` and `target` are node\n                    objects.Finally, the template string has access\n                    to variables `value` and `label`. Anything\n                    contained in tag `<extra>` is displayed in the\n                    secondary box, for example\n                    "<extra>{fullData.name}</extra>". To hide the\n                    secondary box completely, use an empty tag\n                    `<extra></extra>`.\n                hovertemplatesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `hovertemplate`.\n                label\n                    The shown name of the link.\n                labelsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `label`.\n                line\n                    :class:`plotly.graph_objects.sankey.link.Line`\n                    instance or dict with compatible properties\n                source\n                    An integer number `[0..nodes.length - 1]` that\n                    represents the source node.\n                sourcesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `source`.\n                target\n                    An integer number `[0..nodes.length - 1]` that\n                    represents the target node.\n                targetsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `target`.\n                value\n                    A numeric value representing the flow volume\n                    value.\n                valuesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `value`.\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.Link\n        '
        return self['link']

    @link.setter
    def link(self, val):
        if False:
            return 10
        self['link'] = val

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
            return 10
        self['meta'] = val

    @property
    def metasrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for `meta`.\n\n        The 'metasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['metasrc']

    @metasrc.setter
    def metasrc(self, val):
        if False:
            print('Hello World!')
        self['metasrc'] = val

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
            i = 10
            return i + 15
        self['name'] = val

    @property
    def node(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The nodes of the Sankey plot.\n\n        The \'node\' property is an instance of Node\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.Node`\n          - A dict of string/value properties that will be passed\n            to the Node constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the `node` color. It can be a single\n                    value, or an array for specifying color for\n                    each `node`. If `node.color` is omitted, then\n                    the default `Plotly` color palette will be\n                    cycled through to have a variety of colors.\n                    These defaults are not fully opaque, to allow\n                    some visibility of what is beneath the node.\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                customdata\n                    Assigns extra data to each node.\n                customdatasrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `customdata`.\n                groups\n                    Groups of nodes. Each group is defined by an\n                    array with the indices of the nodes it\n                    contains. Multiple groups can be specified.\n                hoverinfo\n                    Determines which trace information appear when\n                    hovering nodes. If `none` or `skip` are set, no\n                    information is displayed upon hovering. But, if\n                    `none` is set, click and hover events are still\n                    fired.\n                hoverlabel\n                    :class:`plotly.graph_objects.sankey.node.Hoverl\n                    abel` instance or dict with compatible\n                    properties\n                hovertemplate\n                    Template string used for rendering the\n                    information that appear on hover box. Note that\n                    this will override `hoverinfo`. Variables are\n                    inserted using %{variable}, for example "y:\n                    %{y}" as well as %{xother}, {%_xother},\n                    {%_xother_}, {%xother_}. When showing info for\n                    several points, "xother" will be added to those\n                    with different x positions from the first\n                    point. An underscore before or after\n                    "(x|y)other" will add a space on that side,\n                    only when this field is shown. Numbers are\n                    formatted using d3-format\'s syntax\n                    %{variable:d3-format}, for example "Price:\n                    %{y:$.2f}". https://github.com/d3/d3-\n                    format/tree/v1.4.5#d3-format for details on the\n                    formatting syntax. Dates are formatted using\n                    d3-time-format\'s syntax %{variable|d3-time-\n                    format}, for example "Day: %{2019-01-01|%A}".\n                    https://github.com/d3/d3-time-\n                    format/tree/v2.2.3#locale_format for details on\n                    the date formatting syntax. The variables\n                    available in `hovertemplate` are the ones\n                    emitted as event data described at this link\n                    https://plotly.com/javascript/plotlyjs-\n                    events/#event-data. Additionally, every\n                    attributes that can be specified per-point (the\n                    ones that are `arrayOk: true`) are available.\n                    Variables `sourceLinks` and `targetLinks` are\n                    arrays of link objects.Finally, the template\n                    string has access to variables `value` and\n                    `label`. Anything contained in tag `<extra>` is\n                    displayed in the secondary box, for example\n                    "<extra>{fullData.name}</extra>". To hide the\n                    secondary box completely, use an empty tag\n                    `<extra></extra>`.\n                hovertemplatesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `hovertemplate`.\n                label\n                    The shown name of the node.\n                labelsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `label`.\n                line\n                    :class:`plotly.graph_objects.sankey.node.Line`\n                    instance or dict with compatible properties\n                pad\n                    Sets the padding (in px) between the `nodes`.\n                thickness\n                    Sets the thickness (in px) of the `nodes`.\n                x\n                    The normalized horizontal position of the node.\n                xsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `x`.\n                y\n                    The normalized vertical position of the node.\n                ysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `y`.\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.Node\n        '
        return self['node']

    @node.setter
    def node(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['node'] = val

    @property
    def orientation(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the orientation of the Sankey diagram.\n\n        The 'orientation' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['v', 'h']\n\n        Returns\n        -------\n        Any\n        "
        return self['orientation']

    @orientation.setter
    def orientation(self, val):
        if False:
            print('Hello World!')
        self['orientation'] = val

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
            return 10
        self['selectedpoints'] = val

    @property
    def stream(self):
        if False:
            return 10
        "\n        The 'stream' property is an instance of Stream\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.Stream`\n          - A dict of string/value properties that will be passed\n            to the Stream constructor\n\n            Supported dict properties:\n\n                maxpoints\n                    Sets the maximum number of points to keep on\n                    the plots from an incoming stream. If\n                    `maxpoints` is set to 50, only the newest 50\n                    points will be displayed on the plot.\n                token\n                    The stream id number links a data trace on a\n                    plot with a stream. See https://chart-\n                    studio.plotly.com/settings for more details.\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.Stream\n        "
        return self['stream']

    @stream.setter
    def stream(self, val):
        if False:
            return 10
        self['stream'] = val

    @property
    def textfont(self):
        if False:
            return 10
        '\n        Sets the font for node labels\n\n        The \'textfont\' property is an instance of Textfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.Textfont`\n          - A dict of string/value properties that will be passed\n            to the Textfont constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.Textfont\n        '
        return self['textfont']

    @textfont.setter
    def textfont(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['textfont'] = val

    @property
    def uid(self):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
        "\n        Controls persistence of some user-driven changes to the trace:\n        `constraintrange` in `parcoords` traces, as well as some\n        `editable: true` modifications such as `name` and\n        `colorbar.title`. Defaults to `layout.uirevision`. Note that\n        other user-driven trace attribute changes are controlled by\n        `layout` attributes: `trace.visible` is controlled by\n        `layout.legend.uirevision`, `selectedpoints` is controlled by\n        `layout.selectionrevision`, and `colorbar.(x|y)` (accessible\n        with `config: {editable: true}`) is controlled by\n        `layout.editrevision`. Trace changes are tracked by `uid`,\n        which only falls back on trace index if no `uid` is provided.\n        So if your app can add/remove traces before the end of the\n        `data` array, such that the same trace has a different index,\n        you can still preserve user-driven changes if you give each\n        trace a `uid` that stays with it as it moves.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            i = 10
            return i + 15
        self['uirevision'] = val

    @property
    def valueformat(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the value formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n\n        The 'valueformat' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['valueformat']

    @valueformat.setter
    def valueformat(self, val):
        if False:
            i = 10
            return i + 15
        self['valueformat'] = val

    @property
    def valuesuffix(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Adds a unit to follow the value in the hover tooltip. Add a\n        space if a separation is necessary from the value.\n\n        The 'valuesuffix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['valuesuffix']

    @valuesuffix.setter
    def valuesuffix(self, val):
        if False:
            while True:
                i = 10
        self['valuesuffix'] = val

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
            return 10
        self['visible'] = val

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        return self._props['type']

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        arrangement\n            If value is `snap` (the default), the node arrangement\n            is assisted by automatic snapping of elements to\n            preserve space between nodes specified via `nodepad`.\n            If value is `perpendicular`, the nodes can only move\n            along a line perpendicular to the flow. If value is\n            `freeform`, the nodes can freely move on the plane. If\n            value is `fixed`, the nodes are stationary.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        domain\n            :class:`plotly.graph_objects.sankey.Domain` instance or\n            dict with compatible properties\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired. Note that this attribute is\n            superseded by `node.hoverinfo` and `node.hoverinfo` for\n            nodes and links respectively.\n        hoverlabel\n            :class:`plotly.graph_objects.sankey.Hoverlabel`\n            instance or dict with compatible properties\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgrouptitle\n            :class:`plotly.graph_objects.sankey.Legendgrouptitle`\n            instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        link\n            The links of the Sankey plot.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        node\n            The nodes of the Sankey plot.\n        orientation\n            Sets the orientation of the Sankey diagram.\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        stream\n            :class:`plotly.graph_objects.sankey.Stream` instance or\n            dict with compatible properties\n        textfont\n            Sets the font for node labels\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        valueformat\n            Sets the value formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n        valuesuffix\n            Adds a unit to follow the value in the hover tooltip.\n            Add a space if a separation is necessary from the\n            value.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        '

    def __init__(self, arg=None, arrangement=None, customdata=None, customdatasrc=None, domain=None, hoverinfo=None, hoverlabel=None, ids=None, idssrc=None, legend=None, legendgrouptitle=None, legendrank=None, legendwidth=None, link=None, meta=None, metasrc=None, name=None, node=None, orientation=None, selectedpoints=None, stream=None, textfont=None, uid=None, uirevision=None, valueformat=None, valuesuffix=None, visible=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Sankey object\n\n        Sankey plots for network flow data analysis. The nodes are\n        specified in `nodes` and the links between sources and targets\n        in `links`. The colors are set in `nodes[i].color` and\n        `links[i].color`, otherwise defaults are used.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.Sankey`\n        arrangement\n            If value is `snap` (the default), the node arrangement\n            is assisted by automatic snapping of elements to\n            preserve space between nodes specified via `nodepad`.\n            If value is `perpendicular`, the nodes can only move\n            along a line perpendicular to the flow. If value is\n            `freeform`, the nodes can freely move on the plane. If\n            value is `fixed`, the nodes are stationary.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        domain\n            :class:`plotly.graph_objects.sankey.Domain` instance or\n            dict with compatible properties\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired. Note that this attribute is\n            superseded by `node.hoverinfo` and `node.hoverinfo` for\n            nodes and links respectively.\n        hoverlabel\n            :class:`plotly.graph_objects.sankey.Hoverlabel`\n            instance or dict with compatible properties\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgrouptitle\n            :class:`plotly.graph_objects.sankey.Legendgrouptitle`\n            instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        link\n            The links of the Sankey plot.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        node\n            The nodes of the Sankey plot.\n        orientation\n            Sets the orientation of the Sankey diagram.\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        stream\n            :class:`plotly.graph_objects.sankey.Stream` instance or\n            dict with compatible properties\n        textfont\n            Sets the font for node labels\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        valueformat\n            Sets the value formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n        valuesuffix\n            Adds a unit to follow the value in the hover tooltip.\n            Add a space if a separation is necessary from the\n            value.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n\n        Returns\n        -------\n        Sankey\n        '
        super(Sankey, self).__init__('sankey')
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
            raise ValueError('The first argument to the plotly.graph_objs.Sankey\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Sankey`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('arrangement', None)
        _v = arrangement if arrangement is not None else _v
        if _v is not None:
            self['arrangement'] = _v
        _v = arg.pop('customdata', None)
        _v = customdata if customdata is not None else _v
        if _v is not None:
            self['customdata'] = _v
        _v = arg.pop('customdatasrc', None)
        _v = customdatasrc if customdatasrc is not None else _v
        if _v is not None:
            self['customdatasrc'] = _v
        _v = arg.pop('domain', None)
        _v = domain if domain is not None else _v
        if _v is not None:
            self['domain'] = _v
        _v = arg.pop('hoverinfo', None)
        _v = hoverinfo if hoverinfo is not None else _v
        if _v is not None:
            self['hoverinfo'] = _v
        _v = arg.pop('hoverlabel', None)
        _v = hoverlabel if hoverlabel is not None else _v
        if _v is not None:
            self['hoverlabel'] = _v
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
        _v = arg.pop('link', None)
        _v = link if link is not None else _v
        if _v is not None:
            self['link'] = _v
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
        _v = arg.pop('node', None)
        _v = node if node is not None else _v
        if _v is not None:
            self['node'] = _v
        _v = arg.pop('orientation', None)
        _v = orientation if orientation is not None else _v
        if _v is not None:
            self['orientation'] = _v
        _v = arg.pop('selectedpoints', None)
        _v = selectedpoints if selectedpoints is not None else _v
        if _v is not None:
            self['selectedpoints'] = _v
        _v = arg.pop('stream', None)
        _v = stream if stream is not None else _v
        if _v is not None:
            self['stream'] = _v
        _v = arg.pop('textfont', None)
        _v = textfont if textfont is not None else _v
        if _v is not None:
            self['textfont'] = _v
        _v = arg.pop('uid', None)
        _v = uid if uid is not None else _v
        if _v is not None:
            self['uid'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('valueformat', None)
        _v = valueformat if valueformat is not None else _v
        if _v is not None:
            self['valueformat'] = _v
        _v = arg.pop('valuesuffix', None)
        _v = valuesuffix if valuesuffix is not None else _v
        if _v is not None:
            self['valuesuffix'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._props['type'] = 'sankey'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
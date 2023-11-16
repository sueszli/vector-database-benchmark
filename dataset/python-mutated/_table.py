from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy

class Table(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'table'
    _valid_props = {'cells', 'columnorder', 'columnordersrc', 'columnwidth', 'columnwidthsrc', 'customdata', 'customdatasrc', 'domain', 'header', 'hoverinfo', 'hoverinfosrc', 'hoverlabel', 'ids', 'idssrc', 'legend', 'legendgrouptitle', 'legendrank', 'legendwidth', 'meta', 'metasrc', 'name', 'stream', 'type', 'uid', 'uirevision', 'visible'}

    @property
    def cells(self):
        if False:
            print('Hello World!')
        "\n        The 'cells' property is an instance of Cells\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.table.Cells`\n          - A dict of string/value properties that will be passed\n            to the Cells constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the `text`\n                    within the box. Has an effect only if `text`\n                    spans two or more lines (i.e. `text` contains\n                    one or more <br> HTML tags) or if an explicit\n                    width is set to override the text width.\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                fill\n                    :class:`plotly.graph_objects.table.cells.Fill`\n                    instance or dict with compatible properties\n                font\n                    :class:`plotly.graph_objects.table.cells.Font`\n                    instance or dict with compatible properties\n                format\n                    Sets the cell value formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format.\n                formatsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `format`.\n                height\n                    The height of cells.\n                line\n                    :class:`plotly.graph_objects.table.cells.Line`\n                    instance or dict with compatible properties\n                prefix\n                    Prefix for cell values.\n                prefixsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `prefix`.\n                suffix\n                    Suffix for cell values.\n                suffixsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `suffix`.\n                values\n                    Cell values. `values[m][n]` represents the\n                    value of the `n`th point in column `m`,\n                    therefore the `values[m]` vector length for all\n                    columns must be the same (longer vectors will\n                    be truncated). Each value must be a finite\n                    number or a string.\n                valuessrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `values`.\n\n        Returns\n        -------\n        plotly.graph_objs.table.Cells\n        "
        return self['cells']

    @cells.setter
    def cells(self, val):
        if False:
            return 10
        self['cells'] = val

    @property
    def columnorder(self):
        if False:
            print('Hello World!')
        "\n        Specifies the rendered order of the data columns; for example,\n        a value `2` at position `0` means that column index `0` in the\n        data will be rendered as the third column, as columns have an\n        index base of zero.\n\n        The 'columnorder' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['columnorder']

    @columnorder.setter
    def columnorder(self, val):
        if False:
            return 10
        self['columnorder'] = val

    @property
    def columnordersrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `columnorder`.\n\n        The 'columnordersrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['columnordersrc']

    @columnordersrc.setter
    def columnordersrc(self, val):
        if False:
            return 10
        self['columnordersrc'] = val

    @property
    def columnwidth(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The width of columns expressed as a ratio. Columns fill the\n        available width in proportion of their specified column widths.\n\n        The 'columnwidth' property is a number and may be specified as:\n          - An int or float\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        int|float|numpy.ndarray\n        "
        return self['columnwidth']

    @columnwidth.setter
    def columnwidth(self, val):
        if False:
            print('Hello World!')
        self['columnwidth'] = val

    @property
    def columnwidthsrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `columnwidth`.\n\n        The 'columnwidthsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['columnwidthsrc']

    @columnwidthsrc.setter
    def columnwidthsrc(self, val):
        if False:
            print('Hello World!')
        self['columnwidthsrc'] = val

    @property
    def customdata(self):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for\n        `customdata`.\n\n        The 'customdatasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['customdatasrc']

    @customdatasrc.setter
    def customdatasrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['customdatasrc'] = val

    @property
    def domain(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'domain' property is an instance of Domain\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.table.Domain`\n          - A dict of string/value properties that will be passed\n            to the Domain constructor\n\n            Supported dict properties:\n\n                column\n                    If there is a layout grid, use the domain for\n                    this column in the grid for this table trace .\n                row\n                    If there is a layout grid, use the domain for\n                    this row in the grid for this table trace .\n                x\n                    Sets the horizontal domain of this table trace\n                    (in plot fraction).\n                y\n                    Sets the vertical domain of this table trace\n                    (in plot fraction).\n\n        Returns\n        -------\n        plotly.graph_objs.table.Domain\n        "
        return self['domain']

    @domain.setter
    def domain(self, val):
        if False:
            return 10
        self['domain'] = val

    @property
    def header(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'header' property is an instance of Header\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.table.Header`\n          - A dict of string/value properties that will be passed\n            to the Header constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the `text`\n                    within the box. Has an effect only if `text`\n                    spans two or more lines (i.e. `text` contains\n                    one or more <br> HTML tags) or if an explicit\n                    width is set to override the text width.\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                fill\n                    :class:`plotly.graph_objects.table.header.Fill`\n                    instance or dict with compatible properties\n                font\n                    :class:`plotly.graph_objects.table.header.Font`\n                    instance or dict with compatible properties\n                format\n                    Sets the cell value formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format.\n                formatsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `format`.\n                height\n                    The height of cells.\n                line\n                    :class:`plotly.graph_objects.table.header.Line`\n                    instance or dict with compatible properties\n                prefix\n                    Prefix for cell values.\n                prefixsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `prefix`.\n                suffix\n                    Suffix for cell values.\n                suffixsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `suffix`.\n                values\n                    Header cell values. `values[m][n]` represents\n                    the value of the `n`th point in column `m`,\n                    therefore the `values[m]` vector length for all\n                    columns must be the same (longer vectors will\n                    be truncated). Each value must be a finite\n                    number or a string.\n                valuessrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `values`.\n\n        Returns\n        -------\n        plotly.graph_objs.table.Header\n        "
        return self['header']

    @header.setter
    def header(self, val):
        if False:
            return 10
        self['header'] = val

    @property
    def hoverinfo(self):
        if False:
            print('Hello World!')
        "\n        Determines which trace information appear on hover. If `none`\n        or `skip` are set, no information is displayed upon hovering.\n        But, if `none` is set, click and hover events are still fired.\n\n        The 'hoverinfo' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['x', 'y', 'z', 'text', 'name'] joined with '+' characters\n            (e.g. 'x+y')\n            OR exactly one of ['all', 'none', 'skip'] (e.g. 'skip')\n          - A list or array of the above\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['hoverinfo']

    @hoverinfo.setter
    def hoverinfo(self, val):
        if False:
            return 10
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
            i = 10
            return i + 15
        self['hoverinfosrc'] = val

    @property
    def hoverlabel(self):
        if False:
            print('Hello World!')
        "\n        The 'hoverlabel' property is an instance of Hoverlabel\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.table.Hoverlabel`\n          - A dict of string/value properties that will be passed\n            to the Hoverlabel constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the text\n                    content within hover label box. Has an effect\n                    only if the hover label text spans more two or\n                    more lines\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                bgcolor\n                    Sets the background color of the hover labels\n                    for this trace\n                bgcolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bgcolor`.\n                bordercolor\n                    Sets the border color of the hover labels for\n                    this trace.\n                bordercolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bordercolor`.\n                font\n                    Sets the font used in hover labels.\n                namelength\n                    Sets the default length (in number of\n                    characters) of the trace name in the hover\n                    labels for all traces. -1 shows the whole name\n                    regardless of length. 0-3 shows the first 0-3\n                    characters, and an integer >3 will show the\n                    whole name if it is less than that many\n                    characters, but if it is longer, will truncate\n                    to `namelength - 3` characters and add an\n                    ellipsis.\n                namelengthsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `namelength`.\n\n        Returns\n        -------\n        plotly.graph_objs.table.Hoverlabel\n        "
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        if False:
            return 10
        self['hoverlabel'] = val

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
            i = 10
            return i + 15
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
            i = 10
            return i + 15
        self['idssrc'] = val

    @property
    def legend(self):
        if False:
            i = 10
            return i + 15
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
        "\n        The 'legendgrouptitle' property is an instance of Legendgrouptitle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.table.Legendgrouptitle`\n          - A dict of string/value properties that will be passed\n            to the Legendgrouptitle constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend group's title font.\n                text\n                    Sets the title of the legend group.\n\n        Returns\n        -------\n        plotly.graph_objs.table.Legendgrouptitle\n        "
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
            while True:
                i = 10
        '\n        Sets the legend rank for this trace. Items and groups with\n        smaller ranks are presented on top/left side while with\n        "reversed" `legend.traceorder` they are on bottom/right side.\n        The default legendrank is 1000, so that you can use ranks less\n        than 1000 to place certain items before all unranked items, and\n        ranks greater than 1000 to go after all unranked items. When\n        having unranked or equal rank items shapes would be displayed\n        after traces i.e. according to their order in data and layout.\n\n        The \'legendrank\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['legendrank']

    @legendrank.setter
    def legendrank(self, val):
        if False:
            return 10
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
            i = 10
            return i + 15
        self['legendwidth'] = val

    @property
    def meta(self):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `meta`.\n\n        The 'metasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['metasrc']

    @metasrc.setter
    def metasrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['metasrc'] = val

    @property
    def name(self):
        if False:
            while True:
                i = 10
        "\n        Sets the trace name. The trace name appears as the legend item\n        and on hover.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            while True:
                i = 10
        self['name'] = val

    @property
    def stream(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'stream' property is an instance of Stream\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.table.Stream`\n          - A dict of string/value properties that will be passed\n            to the Stream constructor\n\n            Supported dict properties:\n\n                maxpoints\n                    Sets the maximum number of points to keep on\n                    the plots from an incoming stream. If\n                    `maxpoints` is set to 50, only the newest 50\n                    points will be displayed on the plot.\n                token\n                    The stream id number links a data trace on a\n                    plot with a stream. See https://chart-\n                    studio.plotly.com/settings for more details.\n\n        Returns\n        -------\n        plotly.graph_objs.table.Stream\n        "
        return self['stream']

    @stream.setter
    def stream(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['stream'] = val

    @property
    def uid(self):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
        "\n        Controls persistence of some user-driven changes to the trace:\n        `constraintrange` in `parcoords` traces, as well as some\n        `editable: true` modifications such as `name` and\n        `colorbar.title`. Defaults to `layout.uirevision`. Note that\n        other user-driven trace attribute changes are controlled by\n        `layout` attributes: `trace.visible` is controlled by\n        `layout.legend.uirevision`, `selectedpoints` is controlled by\n        `layout.selectionrevision`, and `colorbar.(x|y)` (accessible\n        with `config: {editable: true}`) is controlled by\n        `layout.editrevision`. Trace changes are tracked by `uid`,\n        which only falls back on trace index if no `uid` is provided.\n        So if your app can add/remove traces before the end of the\n        `data` array, such that the same trace has a different index,\n        you can still preserve user-driven changes if you give each\n        trace a `uid` that stays with it as it moves.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            i = 10
            return i + 15
        self['uirevision'] = val

    @property
    def visible(self):
        if False:
            print('Hello World!')
        '\n        Determines whether or not this trace is visible. If\n        "legendonly", the trace is not drawn, but can appear as a\n        legend item (provided that the legend itself is visible).\n\n        The \'visible\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, \'legendonly\']\n\n        Returns\n        -------\n        Any\n        '
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            while True:
                i = 10
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
            print('Hello World!')
        return '        cells\n            :class:`plotly.graph_objects.table.Cells` instance or\n            dict with compatible properties\n        columnorder\n            Specifies the rendered order of the data columns; for\n            example, a value `2` at position `0` means that column\n            index `0` in the data will be rendered as the third\n            column, as columns have an index base of zero.\n        columnordersrc\n            Sets the source reference on Chart Studio Cloud for\n            `columnorder`.\n        columnwidth\n            The width of columns expressed as a ratio. Columns fill\n            the available width in proportion of their specified\n            column widths.\n        columnwidthsrc\n            Sets the source reference on Chart Studio Cloud for\n            `columnwidth`.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        domain\n            :class:`plotly.graph_objects.table.Domain` instance or\n            dict with compatible properties\n        header\n            :class:`plotly.graph_objects.table.Header` instance or\n            dict with compatible properties\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.table.Hoverlabel` instance\n            or dict with compatible properties\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgrouptitle\n            :class:`plotly.graph_objects.table.Legendgrouptitle`\n            instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        stream\n            :class:`plotly.graph_objects.table.Stream` instance or\n            dict with compatible properties\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        '

    def __init__(self, arg=None, cells=None, columnorder=None, columnordersrc=None, columnwidth=None, columnwidthsrc=None, customdata=None, customdatasrc=None, domain=None, header=None, hoverinfo=None, hoverinfosrc=None, hoverlabel=None, ids=None, idssrc=None, legend=None, legendgrouptitle=None, legendrank=None, legendwidth=None, meta=None, metasrc=None, name=None, stream=None, uid=None, uirevision=None, visible=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Table object\n\n        Table view for detailed data viewing. The data are arranged in\n        a grid of rows and columns. Most styling can be specified for\n        columns, rows or individual cells. Table is using a column-\n        major order, ie. the grid is represented as a vector of column\n        vectors.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.Table`\n        cells\n            :class:`plotly.graph_objects.table.Cells` instance or\n            dict with compatible properties\n        columnorder\n            Specifies the rendered order of the data columns; for\n            example, a value `2` at position `0` means that column\n            index `0` in the data will be rendered as the third\n            column, as columns have an index base of zero.\n        columnordersrc\n            Sets the source reference on Chart Studio Cloud for\n            `columnorder`.\n        columnwidth\n            The width of columns expressed as a ratio. Columns fill\n            the available width in proportion of their specified\n            column widths.\n        columnwidthsrc\n            Sets the source reference on Chart Studio Cloud for\n            `columnwidth`.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        domain\n            :class:`plotly.graph_objects.table.Domain` instance or\n            dict with compatible properties\n        header\n            :class:`plotly.graph_objects.table.Header` instance or\n            dict with compatible properties\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired.\n        hoverinfosrc\n            Sets the source reference on Chart Studio Cloud for\n            `hoverinfo`.\n        hoverlabel\n            :class:`plotly.graph_objects.table.Hoverlabel` instance\n            or dict with compatible properties\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgrouptitle\n            :class:`plotly.graph_objects.table.Legendgrouptitle`\n            instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        stream\n            :class:`plotly.graph_objects.table.Stream` instance or\n            dict with compatible properties\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n\n        Returns\n        -------\n        Table\n        '
        super(Table, self).__init__('table')
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
            raise ValueError('The first argument to the plotly.graph_objs.Table\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Table`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('cells', None)
        _v = cells if cells is not None else _v
        if _v is not None:
            self['cells'] = _v
        _v = arg.pop('columnorder', None)
        _v = columnorder if columnorder is not None else _v
        if _v is not None:
            self['columnorder'] = _v
        _v = arg.pop('columnordersrc', None)
        _v = columnordersrc if columnordersrc is not None else _v
        if _v is not None:
            self['columnordersrc'] = _v
        _v = arg.pop('columnwidth', None)
        _v = columnwidth if columnwidth is not None else _v
        if _v is not None:
            self['columnwidth'] = _v
        _v = arg.pop('columnwidthsrc', None)
        _v = columnwidthsrc if columnwidthsrc is not None else _v
        if _v is not None:
            self['columnwidthsrc'] = _v
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
        _v = arg.pop('header', None)
        _v = header if header is not None else _v
        if _v is not None:
            self['header'] = _v
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
        _v = arg.pop('stream', None)
        _v = stream if stream is not None else _v
        if _v is not None:
            self['stream'] = _v
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
        self._props['type'] = 'table'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy

class Indicator(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'indicator'
    _valid_props = {'align', 'customdata', 'customdatasrc', 'delta', 'domain', 'gauge', 'ids', 'idssrc', 'legend', 'legendgrouptitle', 'legendrank', 'legendwidth', 'meta', 'metasrc', 'mode', 'name', 'number', 'stream', 'title', 'type', 'uid', 'uirevision', 'value', 'visible'}

    @property
    def align(self):
        if False:
            return 10
        "\n        Sets the horizontal alignment of the `text` within the box.\n        Note that this attribute has no effect if an angular gauge is\n        displayed: in this case, it is always centered\n\n        The 'align' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['left', 'center', 'right']\n\n        Returns\n        -------\n        Any\n        "
        return self['align']

    @align.setter
    def align(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['align'] = val

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
            while True:
                i = 10
        self['customdata'] = val

    @property
    def customdatasrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `customdata`.\n\n        The 'customdatasrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['customdatasrc']

    @customdatasrc.setter
    def customdatasrc(self, val):
        if False:
            while True:
                i = 10
        self['customdatasrc'] = val

    @property
    def delta(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'delta' property is an instance of Delta\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.Delta`\n          - A dict of string/value properties that will be passed\n            to the Delta constructor\n\n            Supported dict properties:\n\n                decreasing\n                    :class:`plotly.graph_objects.indicator.delta.De\n                    creasing` instance or dict with compatible\n                    properties\n                font\n                    Set the font used to display the delta\n                increasing\n                    :class:`plotly.graph_objects.indicator.delta.In\n                    creasing` instance or dict with compatible\n                    properties\n                position\n                    Sets the position of delta with respect to the\n                    number.\n                prefix\n                    Sets a prefix appearing before the delta.\n                reference\n                    Sets the reference value to compute the delta.\n                    By default, it is set to the current value.\n                relative\n                    Show relative change\n                suffix\n                    Sets a suffix appearing next to the delta.\n                valueformat\n                    Sets the value formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format.\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.Delta\n        "
        return self['delta']

    @delta.setter
    def delta(self, val):
        if False:
            print('Hello World!')
        self['delta'] = val

    @property
    def domain(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'domain' property is an instance of Domain\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.Domain`\n          - A dict of string/value properties that will be passed\n            to the Domain constructor\n\n            Supported dict properties:\n\n                column\n                    If there is a layout grid, use the domain for\n                    this column in the grid for this indicator\n                    trace .\n                row\n                    If there is a layout grid, use the domain for\n                    this row in the grid for this indicator trace .\n                x\n                    Sets the horizontal domain of this indicator\n                    trace (in plot fraction).\n                y\n                    Sets the vertical domain of this indicator\n                    trace (in plot fraction).\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.Domain\n        "
        return self['domain']

    @domain.setter
    def domain(self, val):
        if False:
            while True:
                i = 10
        self['domain'] = val

    @property
    def gauge(self):
        if False:
            print('Hello World!')
        "\n        The gauge of the Indicator plot.\n\n        The 'gauge' property is an instance of Gauge\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.Gauge`\n          - A dict of string/value properties that will be passed\n            to the Gauge constructor\n\n            Supported dict properties:\n\n                axis\n                    :class:`plotly.graph_objects.indicator.gauge.Ax\n                    is` instance or dict with compatible properties\n                bar\n                    Set the appearance of the gauge's value\n                bgcolor\n                    Sets the gauge background color.\n                bordercolor\n                    Sets the color of the border enclosing the\n                    gauge.\n                borderwidth\n                    Sets the width (in px) of the border enclosing\n                    the gauge.\n                shape\n                    Set the shape of the gauge\n                steps\n                    A tuple of :class:`plotly.graph_objects.indicat\n                    or.gauge.Step` instances or dicts with\n                    compatible properties\n                stepdefaults\n                    When used in a template (as layout.template.dat\n                    a.indicator.gauge.stepdefaults), sets the\n                    default property values to use for elements of\n                    indicator.gauge.steps\n                threshold\n                    :class:`plotly.graph_objects.indicator.gauge.Th\n                    reshold` instance or dict with compatible\n                    properties\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.Gauge\n        "
        return self['gauge']

    @gauge.setter
    def gauge(self, val):
        if False:
            i = 10
            return i + 15
        self['gauge'] = val

    @property
    def ids(self):
        if False:
            return 10
        "\n        Assigns id labels to each datum. These ids for object constancy\n        of data points during animation. Should be an array of strings,\n        not numbers or any other type.\n\n        The 'ids' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['ids']

    @ids.setter
    def ids(self, val):
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
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
            return 10
        self['legend'] = val

    @property
    def legendgrouptitle(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'legendgrouptitle' property is an instance of Legendgrouptitle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.Legendgrouptitle`\n          - A dict of string/value properties that will be passed\n            to the Legendgrouptitle constructor\n\n            Supported dict properties:\n\n                font\n                    Sets this legend group's title font.\n                text\n                    Sets the title of the legend group.\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.Legendgrouptitle\n        "
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
            return 10
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
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
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
            return 10
        self['metasrc'] = val

    @property
    def mode(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines how the value is displayed on the graph. `number`\n        displays the value numerically in text. `delta` displays the\n        difference to a reference value in text. Finally, `gauge`\n        displays the value graphically on an axis.\n\n        The 'mode' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['number', 'delta', 'gauge'] joined with '+' characters\n            (e.g. 'number+delta')\n\n        Returns\n        -------\n        Any\n        "
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
    def number(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'number' property is an instance of Number\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.Number`\n          - A dict of string/value properties that will be passed\n            to the Number constructor\n\n            Supported dict properties:\n\n                font\n                    Set the font used to display main number\n                prefix\n                    Sets a prefix appearing before the number.\n                suffix\n                    Sets a suffix appearing next to the number.\n                valueformat\n                    Sets the value formatting rule using d3\n                    formatting mini-languages which are very\n                    similar to those in Python. For numbers, see: h\n                    ttps://github.com/d3/d3-format/tree/v1.4.5#d3-\n                    format.\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.Number\n        "
        return self['number']

    @number.setter
    def number(self, val):
        if False:
            print('Hello World!')
        self['number'] = val

    @property
    def stream(self):
        if False:
            while True:
                i = 10
        "\n        The 'stream' property is an instance of Stream\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.Stream`\n          - A dict of string/value properties that will be passed\n            to the Stream constructor\n\n            Supported dict properties:\n\n                maxpoints\n                    Sets the maximum number of points to keep on\n                    the plots from an incoming stream. If\n                    `maxpoints` is set to 50, only the newest 50\n                    points will be displayed on the plot.\n                token\n                    The stream id number links a data trace on a\n                    plot with a stream. See https://chart-\n                    studio.plotly.com/settings for more details.\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.Stream\n        "
        return self['stream']

    @stream.setter
    def stream(self, val):
        if False:
            i = 10
            return i + 15
        self['stream'] = val

    @property
    def title(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'title' property is an instance of Title\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.Title`\n          - A dict of string/value properties that will be passed\n            to the Title constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the title. It\n                    defaults to `center` except for bullet charts\n                    for which it defaults to right.\n                font\n                    Set the font used to display the title\n                text\n                    Sets the title of this indicator.\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.Title\n        "
        return self['title']

    @title.setter
    def title(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['title'] = val

    @property
    def uid(self):
        if False:
            print('Hello World!')
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
            for i in range(10):
                print('nop')
        self['uirevision'] = val

    @property
    def value(self):
        if False:
            return 10
        "\n        Sets the number to be displayed.\n\n        The 'value' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['value']

    @value.setter
    def value(self, val):
        if False:
            while True:
                i = 10
        self['value'] = val

    @property
    def visible(self):
        if False:
            return 10
        '\n        Determines whether or not this trace is visible. If\n        "legendonly", the trace is not drawn, but can appear as a\n        legend item (provided that the legend itself is visible).\n\n        The \'visible\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [True, False, \'legendonly\']\n\n        Returns\n        -------\n        Any\n        '
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            i = 10
            return i + 15
        self['visible'] = val

    @property
    def type(self):
        if False:
            return 10
        return self._props['type']

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        align\n            Sets the horizontal alignment of the `text` within the\n            box. Note that this attribute has no effect if an\n            angular gauge is displayed: in this case, it is always\n            centered\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        delta\n            :class:`plotly.graph_objects.indicator.Delta` instance\n            or dict with compatible properties\n        domain\n            :class:`plotly.graph_objects.indicator.Domain` instance\n            or dict with compatible properties\n        gauge\n            The gauge of the Indicator plot.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgrouptitle\n            :class:`plotly.graph_objects.indicator.Legendgrouptitle\n            ` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        mode\n            Determines how the value is displayed on the graph.\n            `number` displays the value numerically in text.\n            `delta` displays the difference to a reference value in\n            text. Finally, `gauge` displays the value graphically\n            on an axis.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        number\n            :class:`plotly.graph_objects.indicator.Number` instance\n            or dict with compatible properties\n        stream\n            :class:`plotly.graph_objects.indicator.Stream` instance\n            or dict with compatible properties\n        title\n            :class:`plotly.graph_objects.indicator.Title` instance\n            or dict with compatible properties\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        value\n            Sets the number to be displayed.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        '

    def __init__(self, arg=None, align=None, customdata=None, customdatasrc=None, delta=None, domain=None, gauge=None, ids=None, idssrc=None, legend=None, legendgrouptitle=None, legendrank=None, legendwidth=None, meta=None, metasrc=None, mode=None, name=None, number=None, stream=None, title=None, uid=None, uirevision=None, value=None, visible=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Indicator object\n\n        An indicator is used to visualize a single `value` along with\n        some contextual information such as `steps` or a `threshold`,\n        using a combination of three visual elements: a number, a\n        delta, and/or a gauge. Deltas are taken with respect to a\n        `reference`. Gauges can be either angular or bullet (aka\n        linear) gauges.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.Indicator`\n        align\n            Sets the horizontal alignment of the `text` within the\n            box. Note that this attribute has no effect if an\n            angular gauge is displayed: in this case, it is always\n            centered\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        delta\n            :class:`plotly.graph_objects.indicator.Delta` instance\n            or dict with compatible properties\n        domain\n            :class:`plotly.graph_objects.indicator.Domain` instance\n            or dict with compatible properties\n        gauge\n            The gauge of the Indicator plot.\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgrouptitle\n            :class:`plotly.graph_objects.indicator.Legendgrouptitle\n            ` instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        mode\n            Determines how the value is displayed on the graph.\n            `number` displays the value numerically in text.\n            `delta` displays the difference to a reference value in\n            text. Finally, `gauge` displays the value graphically\n            on an axis.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        number\n            :class:`plotly.graph_objects.indicator.Number` instance\n            or dict with compatible properties\n        stream\n            :class:`plotly.graph_objects.indicator.Stream` instance\n            or dict with compatible properties\n        title\n            :class:`plotly.graph_objects.indicator.Title` instance\n            or dict with compatible properties\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        value\n            Sets the number to be displayed.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n\n        Returns\n        -------\n        Indicator\n        '
        super(Indicator, self).__init__('indicator')
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
            raise ValueError('The first argument to the plotly.graph_objs.Indicator\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Indicator`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('align', None)
        _v = align if align is not None else _v
        if _v is not None:
            self['align'] = _v
        _v = arg.pop('customdata', None)
        _v = customdata if customdata is not None else _v
        if _v is not None:
            self['customdata'] = _v
        _v = arg.pop('customdatasrc', None)
        _v = customdatasrc if customdatasrc is not None else _v
        if _v is not None:
            self['customdatasrc'] = _v
        _v = arg.pop('delta', None)
        _v = delta if delta is not None else _v
        if _v is not None:
            self['delta'] = _v
        _v = arg.pop('domain', None)
        _v = domain if domain is not None else _v
        if _v is not None:
            self['domain'] = _v
        _v = arg.pop('gauge', None)
        _v = gauge if gauge is not None else _v
        if _v is not None:
            self['gauge'] = _v
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
        _v = arg.pop('mode', None)
        _v = mode if mode is not None else _v
        if _v is not None:
            self['mode'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('number', None)
        _v = number if number is not None else _v
        if _v is not None:
            self['number'] = _v
        _v = arg.pop('stream', None)
        _v = stream if stream is not None else _v
        if _v is not None:
            self['stream'] = _v
        _v = arg.pop('title', None)
        _v = title if title is not None else _v
        if _v is not None:
            self['title'] = _v
        _v = arg.pop('uid', None)
        _v = uid if uid is not None else _v
        if _v is not None:
            self['uid'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('value', None)
        _v = value if value is not None else _v
        if _v is not None:
            self['value'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._props['type'] = 'indicator'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
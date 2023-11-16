from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Link(_BaseTraceHierarchyType):
    _parent_path_str = 'sankey'
    _path_str = 'sankey.link'
    _valid_props = {'arrowlen', 'color', 'colorscaledefaults', 'colorscales', 'colorsrc', 'customdata', 'customdatasrc', 'hoverinfo', 'hoverlabel', 'hovertemplate', 'hovertemplatesrc', 'label', 'labelsrc', 'line', 'source', 'sourcesrc', 'target', 'targetsrc', 'value', 'valuesrc'}

    @property
    def arrowlen(self):
        if False:
            print('Hello World!')
        "\n        Sets the length (in px) of the links arrow, if 0 no arrow will\n        be drawn.\n\n        The 'arrowlen' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['arrowlen']

    @arrowlen.setter
    def arrowlen(self, val):
        if False:
            i = 10
            return i + 15
        self['arrowlen'] = val

    @property
    def color(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the `link` color. It can be a single value, or an array\n        for specifying color for each `link`. If `link.color` is\n        omitted, then by default, a translucent grey link will be used.\n\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n          - A list or array of any of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['color'] = val

    @property
    def colorscales(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'colorscales' property is a tuple of instances of\n        Colorscale that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.sankey.link.Colorscale\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Colorscale constructor\n\n            Supported dict properties:\n\n                cmax\n                    Sets the upper bound of the color domain.\n                cmin\n                    Sets the lower bound of the color domain.\n                colorscale\n                    Sets the colorscale. The colorscale must be an\n                    array containing arrays mapping a normalized\n                    value to an rgb, rgba, hex, hsl, hsv, or named\n                    color string. At minimum, a mapping for the\n                    lowest (0) and highest (1) values are required.\n                    For example, `[[0, 'rgb(0,0,255)'], [1,\n                    'rgb(255,0,0)']]`. To control the bounds of the\n                    colorscale in color space, use `cmin` and\n                    `cmax`. Alternatively, `colorscale` may be a\n                    palette name string of the following list: Blac\n                    kbody,Bluered,Blues,Cividis,Earth,Electric,Gree\n                    ns,Greys,Hot,Jet,Picnic,Portland,Rainbow,RdBu,R\n                    eds,Viridis,YlGnBu,YlOrRd.\n                label\n                    The label of the links to color based on their\n                    concentration within a flow.\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.sankey.link.Colorscale]\n        "
        return self['colorscales']

    @colorscales.setter
    def colorscales(self, val):
        if False:
            return 10
        self['colorscales'] = val

    @property
    def colorscaledefaults(self):
        if False:
            print('Hello World!')
        "\n        When used in a template (as\n        layout.template.data.sankey.link.colorscaledefaults), sets the\n        default property values to use for elements of\n        sankey.link.colorscales\n\n        The 'colorscaledefaults' property is an instance of Colorscale\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.link.Colorscale`\n          - A dict of string/value properties that will be passed\n            to the Colorscale constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.link.Colorscale\n        "
        return self['colorscaledefaults']

    @colorscaledefaults.setter
    def colorscaledefaults(self, val):
        if False:
            return 10
        self['colorscaledefaults'] = val

    @property
    def colorsrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `color`.\n\n        The 'colorsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['colorsrc']

    @colorsrc.setter
    def colorsrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['colorsrc'] = val

    @property
    def customdata(self):
        if False:
            return 10
        "\n        Assigns extra data to each link.\n\n        The 'customdata' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['customdata']

    @customdata.setter
    def customdata(self, val):
        if False:
            print('Hello World!')
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
    def hoverinfo(self):
        if False:
            while True:
                i = 10
        "\n        Determines which trace information appear when hovering links.\n        If `none` or `skip` are set, no information is displayed upon\n        hovering. But, if `none` is set, click and hover events are\n        still fired.\n\n        The 'hoverinfo' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['all', 'none', 'skip']\n\n        Returns\n        -------\n        Any\n        "
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
            for i in range(10):
                print('nop')
        "\n        The 'hoverlabel' property is an instance of Hoverlabel\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.link.Hoverlabel`\n          - A dict of string/value properties that will be passed\n            to the Hoverlabel constructor\n\n            Supported dict properties:\n\n                align\n                    Sets the horizontal alignment of the text\n                    content within hover label box. Has an effect\n                    only if the hover label text spans more two or\n                    more lines\n                alignsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `align`.\n                bgcolor\n                    Sets the background color of the hover labels\n                    for this trace\n                bgcolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bgcolor`.\n                bordercolor\n                    Sets the border color of the hover labels for\n                    this trace.\n                bordercolorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `bordercolor`.\n                font\n                    Sets the font used in hover labels.\n                namelength\n                    Sets the default length (in number of\n                    characters) of the trace name in the hover\n                    labels for all traces. -1 shows the whole name\n                    regardless of length. 0-3 shows the first 0-3\n                    characters, and an integer >3 will show the\n                    whole name if it is less than that many\n                    characters, but if it is longer, will truncate\n                    to `namelength - 3` characters and add an\n                    ellipsis.\n                namelengthsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `namelength`.\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.link.Hoverlabel\n        "
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
            for i in range(10):
                print('nop')
        '\n        Template string used for rendering the information that appear\n        on hover box. Note that this will override `hoverinfo`.\n        Variables are inserted using %{variable}, for example "y: %{y}"\n        as well as %{xother}, {%_xother}, {%_xother_}, {%xother_}. When\n        showing info for several points, "xother" will be added to\n        those with different x positions from the first point. An\n        underscore before or after "(x|y)other" will add a space on\n        that side, only when this field is shown. Numbers are formatted\n        using d3-format\'s syntax %{variable:d3-format}, for example\n        "Price: %{y:$.2f}".\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format for\n        details on the formatting syntax. Dates are formatted using\n        d3-time-format\'s syntax %{variable|d3-time-format}, for example\n        "Day: %{2019-01-01|%A}". https://github.com/d3/d3-time-\n        format/tree/v2.2.3#locale_format for details on the date\n        formatting syntax. The variables available in `hovertemplate`\n        are the ones emitted as event data described at this link\n        https://plotly.com/javascript/plotlyjs-events/#event-data.\n        Additionally, every attributes that can be specified per-point\n        (the ones that are `arrayOk: true`) are available.  Variables\n        `source` and `target` are node objects.Finally, the template\n        string has access to variables `value` and `label`. Anything\n        contained in tag `<extra>` is displayed in the secondary box,\n        for example "<extra>{fullData.name}</extra>". To hide the\n        secondary box completely, use an empty tag `<extra></extra>`.\n\n        The \'hovertemplate\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n          - A tuple, list, or one-dimensional numpy array of the above\n\n        Returns\n        -------\n        str|numpy.ndarray\n        '
        return self['hovertemplate']

    @hovertemplate.setter
    def hovertemplate(self, val):
        if False:
            print('Hello World!')
        self['hovertemplate'] = val

    @property
    def hovertemplatesrc(self):
        if False:
            print('Hello World!')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `hovertemplate`.\n\n        The 'hovertemplatesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['hovertemplatesrc']

    @hovertemplatesrc.setter
    def hovertemplatesrc(self, val):
        if False:
            i = 10
            return i + 15
        self['hovertemplatesrc'] = val

    @property
    def label(self):
        if False:
            print('Hello World!')
        "\n        The shown name of the link.\n\n        The 'label' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['label']

    @label.setter
    def label(self, val):
        if False:
            print('Hello World!')
        self['label'] = val

    @property
    def labelsrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `label`.\n\n        The 'labelsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['labelsrc']

    @labelsrc.setter
    def labelsrc(self, val):
        if False:
            i = 10
            return i + 15
        self['labelsrc'] = val

    @property
    def line(self):
        if False:
            return 10
        "\n        The 'line' property is an instance of Line\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.sankey.link.Line`\n          - A dict of string/value properties that will be passed\n            to the Line constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the color of the `line` around each\n                    `link`.\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                width\n                    Sets the width (in px) of the `line` around\n                    each `link`.\n                widthsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `width`.\n\n        Returns\n        -------\n        plotly.graph_objs.sankey.link.Line\n        "
        return self['line']

    @line.setter
    def line(self, val):
        if False:
            while True:
                i = 10
        self['line'] = val

    @property
    def source(self):
        if False:
            return 10
        "\n        An integer number `[0..nodes.length - 1]` that represents the\n        source node.\n\n        The 'source' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['source']

    @source.setter
    def source(self, val):
        if False:
            i = 10
            return i + 15
        self['source'] = val

    @property
    def sourcesrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `source`.\n\n        The 'sourcesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['sourcesrc']

    @sourcesrc.setter
    def sourcesrc(self, val):
        if False:
            return 10
        self['sourcesrc'] = val

    @property
    def target(self):
        if False:
            i = 10
            return i + 15
        "\n        An integer number `[0..nodes.length - 1]` that represents the\n        target node.\n\n        The 'target' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['target']

    @target.setter
    def target(self, val):
        if False:
            print('Hello World!')
        self['target'] = val

    @property
    def targetsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `target`.\n\n        The 'targetsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['targetsrc']

    @targetsrc.setter
    def targetsrc(self, val):
        if False:
            return 10
        self['targetsrc'] = val

    @property
    def value(self):
        if False:
            while True:
                i = 10
        "\n        A numeric value representing the flow volume value.\n\n        The 'value' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['value']

    @value.setter
    def value(self, val):
        if False:
            return 10
        self['value'] = val

    @property
    def valuesrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `value`.\n\n        The 'valuesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['valuesrc']

    @valuesrc.setter
    def valuesrc(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['valuesrc'] = val

    @property
    def _prop_descriptions(self):
        if False:
            i = 10
            return i + 15
        return '        arrowlen\n            Sets the length (in px) of the links arrow, if 0 no\n            arrow will be drawn.\n        color\n            Sets the `link` color. It can be a single value, or an\n            array for specifying color for each `link`. If\n            `link.color` is omitted, then by default, a translucent\n            grey link will be used.\n        colorscales\n            A tuple of\n            :class:`plotly.graph_objects.sankey.link.Colorscale`\n            instances or dicts with compatible properties\n        colorscaledefaults\n            When used in a template (as\n            layout.template.data.sankey.link.colorscaledefaults),\n            sets the default property values to use for elements of\n            sankey.link.colorscales\n        colorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `color`.\n        customdata\n            Assigns extra data to each link.\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        hoverinfo\n            Determines which trace information appear when hovering\n            links. If `none` or `skip` are set, no information is\n            displayed upon hovering. But, if `none` is set, click\n            and hover events are still fired.\n        hoverlabel\n            :class:`plotly.graph_objects.sankey.link.Hoverlabel`\n            instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available.  Variables `source` and `target` are\n            node objects.Finally, the template string has access to\n            variables `value` and `label`. Anything contained in\n            tag `<extra>` is displayed in the secondary box, for\n            example "<extra>{fullData.name}</extra>". To hide the\n            secondary box completely, use an empty tag\n            `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        label\n            The shown name of the link.\n        labelsrc\n            Sets the source reference on Chart Studio Cloud for\n            `label`.\n        line\n            :class:`plotly.graph_objects.sankey.link.Line` instance\n            or dict with compatible properties\n        source\n            An integer number `[0..nodes.length - 1]` that\n            represents the source node.\n        sourcesrc\n            Sets the source reference on Chart Studio Cloud for\n            `source`.\n        target\n            An integer number `[0..nodes.length - 1]` that\n            represents the target node.\n        targetsrc\n            Sets the source reference on Chart Studio Cloud for\n            `target`.\n        value\n            A numeric value representing the flow volume value.\n        valuesrc\n            Sets the source reference on Chart Studio Cloud for\n            `value`.\n        '

    def __init__(self, arg=None, arrowlen=None, color=None, colorscales=None, colorscaledefaults=None, colorsrc=None, customdata=None, customdatasrc=None, hoverinfo=None, hoverlabel=None, hovertemplate=None, hovertemplatesrc=None, label=None, labelsrc=None, line=None, source=None, sourcesrc=None, target=None, targetsrc=None, value=None, valuesrc=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Link object\n\n        The links of the Sankey plot.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.sankey.Link`\n        arrowlen\n            Sets the length (in px) of the links arrow, if 0 no\n            arrow will be drawn.\n        color\n            Sets the `link` color. It can be a single value, or an\n            array for specifying color for each `link`. If\n            `link.color` is omitted, then by default, a translucent\n            grey link will be used.\n        colorscales\n            A tuple of\n            :class:`plotly.graph_objects.sankey.link.Colorscale`\n            instances or dicts with compatible properties\n        colorscaledefaults\n            When used in a template (as\n            layout.template.data.sankey.link.colorscaledefaults),\n            sets the default property values to use for elements of\n            sankey.link.colorscales\n        colorsrc\n            Sets the source reference on Chart Studio Cloud for\n            `color`.\n        customdata\n            Assigns extra data to each link.\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        hoverinfo\n            Determines which trace information appear when hovering\n            links. If `none` or `skip` are set, no information is\n            displayed upon hovering. But, if `none` is set, click\n            and hover events are still fired.\n        hoverlabel\n            :class:`plotly.graph_objects.sankey.link.Hoverlabel`\n            instance or dict with compatible properties\n        hovertemplate\n            Template string used for rendering the information that\n            appear on hover box. Note that this will override\n            `hoverinfo`. Variables are inserted using %{variable},\n            for example "y: %{y}" as well as %{xother}, {%_xother},\n            {%_xother_}, {%xother_}. When showing info for several\n            points, "xother" will be added to those with different\n            x positions from the first point. An underscore before\n            or after "(x|y)other" will add a space on that side,\n            only when this field is shown. Numbers are formatted\n            using d3-format\'s syntax %{variable:d3-format}, for\n            example "Price: %{y:$.2f}".\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format\n            for details on the formatting syntax. Dates are\n            formatted using d3-time-format\'s syntax\n            %{variable|d3-time-format}, for example "Day:\n            %{2019-01-01|%A}". https://github.com/d3/d3-time-\n            format/tree/v2.2.3#locale_format for details on the\n            date formatting syntax. The variables available in\n            `hovertemplate` are the ones emitted as event data\n            described at this link\n            https://plotly.com/javascript/plotlyjs-events/#event-\n            data. Additionally, every attributes that can be\n            specified per-point (the ones that are `arrayOk: true`)\n            are available.  Variables `source` and `target` are\n            node objects.Finally, the template string has access to\n            variables `value` and `label`. Anything contained in\n            tag `<extra>` is displayed in the secondary box, for\n            example "<extra>{fullData.name}</extra>". To hide the\n            secondary box completely, use an empty tag\n            `<extra></extra>`.\n        hovertemplatesrc\n            Sets the source reference on Chart Studio Cloud for\n            `hovertemplate`.\n        label\n            The shown name of the link.\n        labelsrc\n            Sets the source reference on Chart Studio Cloud for\n            `label`.\n        line\n            :class:`plotly.graph_objects.sankey.link.Line` instance\n            or dict with compatible properties\n        source\n            An integer number `[0..nodes.length - 1]` that\n            represents the source node.\n        sourcesrc\n            Sets the source reference on Chart Studio Cloud for\n            `source`.\n        target\n            An integer number `[0..nodes.length - 1]` that\n            represents the target node.\n        targetsrc\n            Sets the source reference on Chart Studio Cloud for\n            `target`.\n        value\n            A numeric value representing the flow volume value.\n        valuesrc\n            Sets the source reference on Chart Studio Cloud for\n            `value`.\n\n        Returns\n        -------\n        Link\n        '
        super(Link, self).__init__('link')
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
            raise ValueError('The first argument to the plotly.graph_objs.sankey.Link\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.sankey.Link`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('arrowlen', None)
        _v = arrowlen if arrowlen is not None else _v
        if _v is not None:
            self['arrowlen'] = _v
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('colorscales', None)
        _v = colorscales if colorscales is not None else _v
        if _v is not None:
            self['colorscales'] = _v
        _v = arg.pop('colorscaledefaults', None)
        _v = colorscaledefaults if colorscaledefaults is not None else _v
        if _v is not None:
            self['colorscaledefaults'] = _v
        _v = arg.pop('colorsrc', None)
        _v = colorsrc if colorsrc is not None else _v
        if _v is not None:
            self['colorsrc'] = _v
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
        _v = arg.pop('label', None)
        _v = label if label is not None else _v
        if _v is not None:
            self['label'] = _v
        _v = arg.pop('labelsrc', None)
        _v = labelsrc if labelsrc is not None else _v
        if _v is not None:
            self['labelsrc'] = _v
        _v = arg.pop('line', None)
        _v = line if line is not None else _v
        if _v is not None:
            self['line'] = _v
        _v = arg.pop('source', None)
        _v = source if source is not None else _v
        if _v is not None:
            self['source'] = _v
        _v = arg.pop('sourcesrc', None)
        _v = sourcesrc if sourcesrc is not None else _v
        if _v is not None:
            self['sourcesrc'] = _v
        _v = arg.pop('target', None)
        _v = target if target is not None else _v
        if _v is not None:
            self['target'] = _v
        _v = arg.pop('targetsrc', None)
        _v = targetsrc if targetsrc is not None else _v
        if _v is not None:
            self['targetsrc'] = _v
        _v = arg.pop('value', None)
        _v = value if value is not None else _v
        if _v is not None:
            self['value'] = _v
        _v = arg.pop('valuesrc', None)
        _v = valuesrc if valuesrc is not None else _v
        if _v is not None:
            self['valuesrc'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
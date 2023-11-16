from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Layer(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.mapbox'
    _path_str = 'layout.mapbox.layer'
    _valid_props = {'below', 'circle', 'color', 'coordinates', 'fill', 'line', 'maxzoom', 'minzoom', 'name', 'opacity', 'source', 'sourceattribution', 'sourcelayer', 'sourcetype', 'symbol', 'templateitemname', 'type', 'visible'}

    @property
    def below(self):
        if False:
            return 10
        "\n        Determines if the layer will be inserted before the layer with\n        the specified ID. If omitted or set to '', the layer will be\n        inserted above every existing layer.\n\n        The 'below' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['below']

    @below.setter
    def below(self, val):
        if False:
            i = 10
            return i + 15
        self['below'] = val

    @property
    def circle(self):
        if False:
            i = 10
            return i + 15
        '\n        The \'circle\' property is an instance of Circle\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.mapbox.layer.Circle`\n          - A dict of string/value properties that will be passed\n            to the Circle constructor\n\n            Supported dict properties:\n\n                radius\n                    Sets the circle radius\n                    (mapbox.layer.paint.circle-radius). Has an\n                    effect only when `type` is set to "circle".\n\n        Returns\n        -------\n        plotly.graph_objs.layout.mapbox.layer.Circle\n        '
        return self['circle']

    @circle.setter
    def circle(self, val):
        if False:
            i = 10
            return i + 15
        self['circle'] = val

    @property
    def color(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the primary layer color. If `type` is "circle", color\n        corresponds to the circle color (mapbox.layer.paint.circle-\n        color) If `type` is "line", color corresponds to the line color\n        (mapbox.layer.paint.line-color) If `type` is "fill", color\n        corresponds to the fill color (mapbox.layer.paint.fill-color)\n        If `type` is "symbol", color corresponds to the icon color\n        (mapbox.layer.paint.icon-color)\n\n        The \'color\' property is a color and may be specified as:\n          - A hex string (e.g. \'#ff0000\')\n          - An rgb/rgba string (e.g. \'rgb(255,0,0)\')\n          - An hsl/hsla string (e.g. \'hsl(0,100%,50%)\')\n          - An hsv/hsva string (e.g. \'hsv(0,100%,100%)\')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        '
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            print('Hello World!')
        self['color'] = val

    @property
    def coordinates(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the coordinates array contains [longitude, latitude] pairs\n        for the image corners listed in clockwise order: top left, top\n        right, bottom right, bottom left. Only has an effect for\n        "image" `sourcetype`.\n\n        The \'coordinates\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['coordinates']

    @coordinates.setter
    def coordinates(self, val):
        if False:
            while True:
                i = 10
        self['coordinates'] = val

    @property
    def fill(self):
        if False:
            return 10
        '\n        The \'fill\' property is an instance of Fill\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.mapbox.layer.Fill`\n          - A dict of string/value properties that will be passed\n            to the Fill constructor\n\n            Supported dict properties:\n\n                outlinecolor\n                    Sets the fill outline color\n                    (mapbox.layer.paint.fill-outline-color). Has an\n                    effect only when `type` is set to "fill".\n\n        Returns\n        -------\n        plotly.graph_objs.layout.mapbox.layer.Fill\n        '
        return self['fill']

    @fill.setter
    def fill(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['fill'] = val

    @property
    def line(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The \'line\' property is an instance of Line\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.mapbox.layer.Line`\n          - A dict of string/value properties that will be passed\n            to the Line constructor\n\n            Supported dict properties:\n\n                dash\n                    Sets the length of dashes and gaps\n                    (mapbox.layer.paint.line-dasharray). Has an\n                    effect only when `type` is set to "line".\n                dashsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `dash`.\n                width\n                    Sets the line width (mapbox.layer.paint.line-\n                    width). Has an effect only when `type` is set\n                    to "line".\n\n        Returns\n        -------\n        plotly.graph_objs.layout.mapbox.layer.Line\n        '
        return self['line']

    @line.setter
    def line(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['line'] = val

    @property
    def maxzoom(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the maximum zoom level (mapbox.layer.maxzoom). At zoom\n        levels equal to or greater than the maxzoom, the layer will be\n        hidden.\n\n        The 'maxzoom' property is a number and may be specified as:\n          - An int or float in the interval [0, 24]\n\n        Returns\n        -------\n        int|float\n        "
        return self['maxzoom']

    @maxzoom.setter
    def maxzoom(self, val):
        if False:
            i = 10
            return i + 15
        self['maxzoom'] = val

    @property
    def minzoom(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the minimum zoom level (mapbox.layer.minzoom). At zoom\n        levels less than the minzoom, the layer will be hidden.\n\n        The 'minzoom' property is a number and may be specified as:\n          - An int or float in the interval [0, 24]\n\n        Returns\n        -------\n        int|float\n        "
        return self['minzoom']

    @minzoom.setter
    def minzoom(self, val):
        if False:
            while True:
                i = 10
        self['minzoom'] = val

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        "\n        When used in a template, named items are created in the output\n        figure in addition to any items the figure already has in this\n        array. You can modify these items in the output figure by\n        making your own item with `templateitemname` matching this\n        `name` alongside your modifications (including `visible: false`\n        or `enabled: false` to hide it). Has no effect outside of a\n        template.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            i = 10
            return i + 15
        self['name'] = val

    @property
    def opacity(self):
        if False:
            return 10
        '\n        Sets the opacity of the layer. If `type` is "circle", opacity\n        corresponds to the circle opacity (mapbox.layer.paint.circle-\n        opacity) If `type` is "line", opacity corresponds to the line\n        opacity (mapbox.layer.paint.line-opacity) If `type` is "fill",\n        opacity corresponds to the fill opacity\n        (mapbox.layer.paint.fill-opacity) If `type` is "symbol",\n        opacity corresponds to the icon/text opacity\n        (mapbox.layer.paint.text-opacity)\n\n        The \'opacity\' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        '
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            return 10
        self['opacity'] = val

    @property
    def source(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the source data for this layer (mapbox.layer.source). When\n        `sourcetype` is set to "geojson", `source` can be a URL to a\n        GeoJSON or a GeoJSON object. When `sourcetype` is set to\n        "vector" or "raster", `source` can be a URL or an array of tile\n        URLs. When `sourcetype` is set to "image", `source` can be a\n        URL to an image.\n\n        The \'source\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['source']

    @source.setter
    def source(self, val):
        if False:
            print('Hello World!')
        self['source'] = val

    @property
    def sourceattribution(self):
        if False:
            while True:
                i = 10
        "\n        Sets the attribution for this source.\n\n        The 'sourceattribution' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['sourceattribution']

    @sourceattribution.setter
    def sourceattribution(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['sourceattribution'] = val

    @property
    def sourcelayer(self):
        if False:
            return 10
        '\n        Specifies the layer to use from a vector tile source\n        (mapbox.layer.source-layer). Required for "vector" source type\n        that supports multiple layers.\n\n        The \'sourcelayer\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['sourcelayer']

    @sourcelayer.setter
    def sourcelayer(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['sourcelayer'] = val

    @property
    def sourcetype(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source type for this layer, that is the type of the\n        layer data.\n\n        The 'sourcetype' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['geojson', 'vector', 'raster', 'image']\n\n        Returns\n        -------\n        Any\n        "
        return self['sourcetype']

    @sourcetype.setter
    def sourcetype(self, val):
        if False:
            while True:
                i = 10
        self['sourcetype'] = val

    @property
    def symbol(self):
        if False:
            while True:
                i = 10
        '\n        The \'symbol\' property is an instance of Symbol\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.mapbox.layer.Symbol`\n          - A dict of string/value properties that will be passed\n            to the Symbol constructor\n\n            Supported dict properties:\n\n                icon\n                    Sets the symbol icon image\n                    (mapbox.layer.layout.icon-image). Full list:\n                    https://www.mapbox.com/maki-icons/\n                iconsize\n                    Sets the symbol icon size\n                    (mapbox.layer.layout.icon-size). Has an effect\n                    only when `type` is set to "symbol".\n                placement\n                    Sets the symbol and/or text placement\n                    (mapbox.layer.layout.symbol-placement). If\n                    `placement` is "point", the label is placed\n                    where the geometry is located If `placement` is\n                    "line", the label is placed along the line of\n                    the geometry If `placement` is "line-center",\n                    the label is placed on the center of the\n                    geometry\n                text\n                    Sets the symbol text (mapbox.layer.layout.text-\n                    field).\n                textfont\n                    Sets the icon text font\n                    (color=mapbox.layer.paint.text-color,\n                    size=mapbox.layer.layout.text-size). Has an\n                    effect only when `type` is set to "symbol".\n                textposition\n                    Sets the positions of the `text` elements with\n                    respects to the (x,y) coordinates.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.mapbox.layer.Symbol\n        '
        return self['symbol']

    @symbol.setter
    def symbol(self, val):
        if False:
            return 10
        self['symbol'] = val

    @property
    def templateitemname(self):
        if False:
            while True:
                i = 10
        "\n        Used to refer to a named item in this array in the template.\n        Named items from the template will be created even without a\n        matching item in the input figure, but you can modify one by\n        making an item with `templateitemname` matching its `name`,\n        alongside your modifications (including `visible: false` or\n        `enabled: false` to hide it). If there is no template or no\n        matching item, this item will be hidden unless you explicitly\n        show it with `visible: true`.\n\n        The 'templateitemname' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['templateitemname']

    @templateitemname.setter
    def templateitemname(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['templateitemname'] = val

    @property
    def type(self):
        if False:
            print('Hello World!')
        '\n        Sets the layer type, that is the how the layer data set in\n        `source` will be rendered With `sourcetype` set to "geojson",\n        the following values are allowed: "circle", "line", "fill" and\n        "symbol". but note that "line" and "fill" are not compatible\n        with Point GeoJSON geometries. With `sourcetype` set to\n        "vector", the following values are allowed:  "circle", "line",\n        "fill" and "symbol". With `sourcetype` set to "raster" or\n        `*image*`, only the "raster" value is allowed.\n\n        The \'type\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'circle\', \'line\', \'fill\', \'symbol\', \'raster\']\n\n        Returns\n        -------\n        Any\n        '
        return self['type']

    @type.setter
    def type(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['type'] = val

    @property
    def visible(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines whether this layer is displayed\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            return 10
        self['visible'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        below\n            Determines if the layer will be inserted before the\n            layer with the specified ID. If omitted or set to \'\',\n            the layer will be inserted above every existing layer.\n        circle\n            :class:`plotly.graph_objects.layout.mapbox.layer.Circle\n            ` instance or dict with compatible properties\n        color\n            Sets the primary layer color. If `type` is "circle",\n            color corresponds to the circle color\n            (mapbox.layer.paint.circle-color) If `type` is "line",\n            color corresponds to the line color\n            (mapbox.layer.paint.line-color) If `type` is "fill",\n            color corresponds to the fill color\n            (mapbox.layer.paint.fill-color) If `type` is "symbol",\n            color corresponds to the icon color\n            (mapbox.layer.paint.icon-color)\n        coordinates\n            Sets the coordinates array contains [longitude,\n            latitude] pairs for the image corners listed in\n            clockwise order: top left, top right, bottom right,\n            bottom left. Only has an effect for "image"\n            `sourcetype`.\n        fill\n            :class:`plotly.graph_objects.layout.mapbox.layer.Fill`\n            instance or dict with compatible properties\n        line\n            :class:`plotly.graph_objects.layout.mapbox.layer.Line`\n            instance or dict with compatible properties\n        maxzoom\n            Sets the maximum zoom level (mapbox.layer.maxzoom). At\n            zoom levels equal to or greater than the maxzoom, the\n            layer will be hidden.\n        minzoom\n            Sets the minimum zoom level (mapbox.layer.minzoom). At\n            zoom levels less than the minzoom, the layer will be\n            hidden.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        opacity\n            Sets the opacity of the layer. If `type` is "circle",\n            opacity corresponds to the circle opacity\n            (mapbox.layer.paint.circle-opacity) If `type` is\n            "line", opacity corresponds to the line opacity\n            (mapbox.layer.paint.line-opacity) If `type` is "fill",\n            opacity corresponds to the fill opacity\n            (mapbox.layer.paint.fill-opacity) If `type` is\n            "symbol", opacity corresponds to the icon/text opacity\n            (mapbox.layer.paint.text-opacity)\n        source\n            Sets the source data for this layer\n            (mapbox.layer.source). When `sourcetype` is set to\n            "geojson", `source` can be a URL to a GeoJSON or a\n            GeoJSON object. When `sourcetype` is set to "vector" or\n            "raster", `source` can be a URL or an array of tile\n            URLs. When `sourcetype` is set to "image", `source` can\n            be a URL to an image.\n        sourceattribution\n            Sets the attribution for this source.\n        sourcelayer\n            Specifies the layer to use from a vector tile source\n            (mapbox.layer.source-layer). Required for "vector"\n            source type that supports multiple layers.\n        sourcetype\n            Sets the source type for this layer, that is the type\n            of the layer data.\n        symbol\n            :class:`plotly.graph_objects.layout.mapbox.layer.Symbol\n            ` instance or dict with compatible properties\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        type\n            Sets the layer type, that is the how the layer data set\n            in `source` will be rendered With `sourcetype` set to\n            "geojson", the following values are allowed: "circle",\n            "line", "fill" and "symbol". but note that "line" and\n            "fill" are not compatible with Point GeoJSON\n            geometries. With `sourcetype` set to "vector", the\n            following values are allowed:  "circle", "line", "fill"\n            and "symbol". With `sourcetype` set to "raster" or\n            `*image*`, only the "raster" value is allowed.\n        visible\n            Determines whether this layer is displayed\n        '

    def __init__(self, arg=None, below=None, circle=None, color=None, coordinates=None, fill=None, line=None, maxzoom=None, minzoom=None, name=None, opacity=None, source=None, sourceattribution=None, sourcelayer=None, sourcetype=None, symbol=None, templateitemname=None, type=None, visible=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Layer object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.mapbox.Layer`\n        below\n            Determines if the layer will be inserted before the\n            layer with the specified ID. If omitted or set to \'\',\n            the layer will be inserted above every existing layer.\n        circle\n            :class:`plotly.graph_objects.layout.mapbox.layer.Circle\n            ` instance or dict with compatible properties\n        color\n            Sets the primary layer color. If `type` is "circle",\n            color corresponds to the circle color\n            (mapbox.layer.paint.circle-color) If `type` is "line",\n            color corresponds to the line color\n            (mapbox.layer.paint.line-color) If `type` is "fill",\n            color corresponds to the fill color\n            (mapbox.layer.paint.fill-color) If `type` is "symbol",\n            color corresponds to the icon color\n            (mapbox.layer.paint.icon-color)\n        coordinates\n            Sets the coordinates array contains [longitude,\n            latitude] pairs for the image corners listed in\n            clockwise order: top left, top right, bottom right,\n            bottom left. Only has an effect for "image"\n            `sourcetype`.\n        fill\n            :class:`plotly.graph_objects.layout.mapbox.layer.Fill`\n            instance or dict with compatible properties\n        line\n            :class:`plotly.graph_objects.layout.mapbox.layer.Line`\n            instance or dict with compatible properties\n        maxzoom\n            Sets the maximum zoom level (mapbox.layer.maxzoom). At\n            zoom levels equal to or greater than the maxzoom, the\n            layer will be hidden.\n        minzoom\n            Sets the minimum zoom level (mapbox.layer.minzoom). At\n            zoom levels less than the minzoom, the layer will be\n            hidden.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        opacity\n            Sets the opacity of the layer. If `type` is "circle",\n            opacity corresponds to the circle opacity\n            (mapbox.layer.paint.circle-opacity) If `type` is\n            "line", opacity corresponds to the line opacity\n            (mapbox.layer.paint.line-opacity) If `type` is "fill",\n            opacity corresponds to the fill opacity\n            (mapbox.layer.paint.fill-opacity) If `type` is\n            "symbol", opacity corresponds to the icon/text opacity\n            (mapbox.layer.paint.text-opacity)\n        source\n            Sets the source data for this layer\n            (mapbox.layer.source). When `sourcetype` is set to\n            "geojson", `source` can be a URL to a GeoJSON or a\n            GeoJSON object. When `sourcetype` is set to "vector" or\n            "raster", `source` can be a URL or an array of tile\n            URLs. When `sourcetype` is set to "image", `source` can\n            be a URL to an image.\n        sourceattribution\n            Sets the attribution for this source.\n        sourcelayer\n            Specifies the layer to use from a vector tile source\n            (mapbox.layer.source-layer). Required for "vector"\n            source type that supports multiple layers.\n        sourcetype\n            Sets the source type for this layer, that is the type\n            of the layer data.\n        symbol\n            :class:`plotly.graph_objects.layout.mapbox.layer.Symbol\n            ` instance or dict with compatible properties\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        type\n            Sets the layer type, that is the how the layer data set\n            in `source` will be rendered With `sourcetype` set to\n            "geojson", the following values are allowed: "circle",\n            "line", "fill" and "symbol". but note that "line" and\n            "fill" are not compatible with Point GeoJSON\n            geometries. With `sourcetype` set to "vector", the\n            following values are allowed:  "circle", "line", "fill"\n            and "symbol". With `sourcetype` set to "raster" or\n            `*image*`, only the "raster" value is allowed.\n        visible\n            Determines whether this layer is displayed\n\n        Returns\n        -------\n        Layer\n        '
        super(Layer, self).__init__('layers')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.mapbox.Layer\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.mapbox.Layer`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('below', None)
        _v = below if below is not None else _v
        if _v is not None:
            self['below'] = _v
        _v = arg.pop('circle', None)
        _v = circle if circle is not None else _v
        if _v is not None:
            self['circle'] = _v
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('coordinates', None)
        _v = coordinates if coordinates is not None else _v
        if _v is not None:
            self['coordinates'] = _v
        _v = arg.pop('fill', None)
        _v = fill if fill is not None else _v
        if _v is not None:
            self['fill'] = _v
        _v = arg.pop('line', None)
        _v = line if line is not None else _v
        if _v is not None:
            self['line'] = _v
        _v = arg.pop('maxzoom', None)
        _v = maxzoom if maxzoom is not None else _v
        if _v is not None:
            self['maxzoom'] = _v
        _v = arg.pop('minzoom', None)
        _v = minzoom if minzoom is not None else _v
        if _v is not None:
            self['minzoom'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        _v = arg.pop('source', None)
        _v = source if source is not None else _v
        if _v is not None:
            self['source'] = _v
        _v = arg.pop('sourceattribution', None)
        _v = sourceattribution if sourceattribution is not None else _v
        if _v is not None:
            self['sourceattribution'] = _v
        _v = arg.pop('sourcelayer', None)
        _v = sourcelayer if sourcelayer is not None else _v
        if _v is not None:
            self['sourcelayer'] = _v
        _v = arg.pop('sourcetype', None)
        _v = sourcetype if sourcetype is not None else _v
        if _v is not None:
            self['sourcetype'] = _v
        _v = arg.pop('symbol', None)
        _v = symbol if symbol is not None else _v
        if _v is not None:
            self['symbol'] = _v
        _v = arg.pop('templateitemname', None)
        _v = templateitemname if templateitemname is not None else _v
        if _v is not None:
            self['templateitemname'] = _v
        _v = arg.pop('type', None)
        _v = type if type is not None else _v
        if _v is not None:
            self['type'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
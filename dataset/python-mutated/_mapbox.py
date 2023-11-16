from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Mapbox(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.mapbox'
    _valid_props = {'accesstoken', 'bearing', 'bounds', 'center', 'domain', 'layerdefaults', 'layers', 'pitch', 'style', 'uirevision', 'zoom'}

    @property
    def accesstoken(self):
        if False:
            return 10
        "\n        Sets the mapbox access token to be used for this mapbox map.\n        Alternatively, the mapbox access token can be set in the\n        configuration options under `mapboxAccessToken`. Note that\n        accessToken are only required when `style` (e.g with values :\n        basic, streets, outdoors, light, dark, satellite, satellite-\n        streets ) and/or a layout layer references the Mapbox server.\n\n        The 'accesstoken' property is a string and must be specified as:\n          - A non-empty string\n\n        Returns\n        -------\n        str\n        "
        return self['accesstoken']

    @accesstoken.setter
    def accesstoken(self, val):
        if False:
            while True:
                i = 10
        self['accesstoken'] = val

    @property
    def bearing(self):
        if False:
            print('Hello World!')
        "\n        Sets the bearing angle of the map in degrees counter-clockwise\n        from North (mapbox.bearing).\n\n        The 'bearing' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['bearing']

    @bearing.setter
    def bearing(self, val):
        if False:
            while True:
                i = 10
        self['bearing'] = val

    @property
    def bounds(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'bounds' property is an instance of Bounds\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.mapbox.Bounds`\n          - A dict of string/value properties that will be passed\n            to the Bounds constructor\n\n            Supported dict properties:\n\n                east\n                    Sets the maximum longitude of the map (in\n                    degrees East) if `west`, `south` and `north`\n                    are declared.\n                north\n                    Sets the maximum latitude of the map (in\n                    degrees North) if `east`, `west` and `south`\n                    are declared.\n                south\n                    Sets the minimum latitude of the map (in\n                    degrees North) if `east`, `west` and `north`\n                    are declared.\n                west\n                    Sets the minimum longitude of the map (in\n                    degrees East) if `east`, `south` and `north`\n                    are declared.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.mapbox.Bounds\n        "
        return self['bounds']

    @bounds.setter
    def bounds(self, val):
        if False:
            return 10
        self['bounds'] = val

    @property
    def center(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'center' property is an instance of Center\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.mapbox.Center`\n          - A dict of string/value properties that will be passed\n            to the Center constructor\n\n            Supported dict properties:\n\n                lat\n                    Sets the latitude of the center of the map (in\n                    degrees North).\n                lon\n                    Sets the longitude of the center of the map (in\n                    degrees East).\n\n        Returns\n        -------\n        plotly.graph_objs.layout.mapbox.Center\n        "
        return self['center']

    @center.setter
    def center(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['center'] = val

    @property
    def domain(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'domain' property is an instance of Domain\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.mapbox.Domain`\n          - A dict of string/value properties that will be passed\n            to the Domain constructor\n\n            Supported dict properties:\n\n                column\n                    If there is a layout grid, use the domain for\n                    this column in the grid for this mapbox subplot\n                    .\n                row\n                    If there is a layout grid, use the domain for\n                    this row in the grid for this mapbox subplot .\n                x\n                    Sets the horizontal domain of this mapbox\n                    subplot (in plot fraction).\n                y\n                    Sets the vertical domain of this mapbox subplot\n                    (in plot fraction).\n\n        Returns\n        -------\n        plotly.graph_objs.layout.mapbox.Domain\n        "
        return self['domain']

    @domain.setter
    def domain(self, val):
        if False:
            return 10
        self['domain'] = val

    @property
    def layers(self):
        if False:
            while True:
                i = 10
        '\n        The \'layers\' property is a tuple of instances of\n        Layer that may be specified as:\n          - A list or tuple of instances of plotly.graph_objs.layout.mapbox.Layer\n          - A list or tuple of dicts of string/value properties that\n            will be passed to the Layer constructor\n\n            Supported dict properties:\n\n                below\n                    Determines if the layer will be inserted before\n                    the layer with the specified ID. If omitted or\n                    set to \'\', the layer will be inserted above\n                    every existing layer.\n                circle\n                    :class:`plotly.graph_objects.layout.mapbox.laye\n                    r.Circle` instance or dict with compatible\n                    properties\n                color\n                    Sets the primary layer color. If `type` is\n                    "circle", color corresponds to the circle color\n                    (mapbox.layer.paint.circle-color) If `type` is\n                    "line", color corresponds to the line color\n                    (mapbox.layer.paint.line-color) If `type` is\n                    "fill", color corresponds to the fill color\n                    (mapbox.layer.paint.fill-color) If `type` is\n                    "symbol", color corresponds to the icon color\n                    (mapbox.layer.paint.icon-color)\n                coordinates\n                    Sets the coordinates array contains [longitude,\n                    latitude] pairs for the image corners listed in\n                    clockwise order: top left, top right, bottom\n                    right, bottom left. Only has an effect for\n                    "image" `sourcetype`.\n                fill\n                    :class:`plotly.graph_objects.layout.mapbox.laye\n                    r.Fill` instance or dict with compatible\n                    properties\n                line\n                    :class:`plotly.graph_objects.layout.mapbox.laye\n                    r.Line` instance or dict with compatible\n                    properties\n                maxzoom\n                    Sets the maximum zoom level\n                    (mapbox.layer.maxzoom). At zoom levels equal to\n                    or greater than the maxzoom, the layer will be\n                    hidden.\n                minzoom\n                    Sets the minimum zoom level\n                    (mapbox.layer.minzoom). At zoom levels less\n                    than the minzoom, the layer will be hidden.\n                name\n                    When used in a template, named items are\n                    created in the output figure in addition to any\n                    items the figure already has in this array. You\n                    can modify these items in the output figure by\n                    making your own item with `templateitemname`\n                    matching this `name` alongside your\n                    modifications (including `visible: false` or\n                    `enabled: false` to hide it). Has no effect\n                    outside of a template.\n                opacity\n                    Sets the opacity of the layer. If `type` is\n                    "circle", opacity corresponds to the circle\n                    opacity (mapbox.layer.paint.circle-opacity) If\n                    `type` is "line", opacity corresponds to the\n                    line opacity (mapbox.layer.paint.line-opacity)\n                    If `type` is "fill", opacity corresponds to the\n                    fill opacity (mapbox.layer.paint.fill-opacity)\n                    If `type` is "symbol", opacity corresponds to\n                    the icon/text opacity (mapbox.layer.paint.text-\n                    opacity)\n                source\n                    Sets the source data for this layer\n                    (mapbox.layer.source). When `sourcetype` is set\n                    to "geojson", `source` can be a URL to a\n                    GeoJSON or a GeoJSON object. When `sourcetype`\n                    is set to "vector" or "raster", `source` can be\n                    a URL or an array of tile URLs. When\n                    `sourcetype` is set to "image", `source` can be\n                    a URL to an image.\n                sourceattribution\n                    Sets the attribution for this source.\n                sourcelayer\n                    Specifies the layer to use from a vector tile\n                    source (mapbox.layer.source-layer). Required\n                    for "vector" source type that supports multiple\n                    layers.\n                sourcetype\n                    Sets the source type for this layer, that is\n                    the type of the layer data.\n                symbol\n                    :class:`plotly.graph_objects.layout.mapbox.laye\n                    r.Symbol` instance or dict with compatible\n                    properties\n                templateitemname\n                    Used to refer to a named item in this array in\n                    the template. Named items from the template\n                    will be created even without a matching item in\n                    the input figure, but you can modify one by\n                    making an item with `templateitemname` matching\n                    its `name`, alongside your modifications\n                    (including `visible: false` or `enabled: false`\n                    to hide it). If there is no template or no\n                    matching item, this item will be hidden unless\n                    you explicitly show it with `visible: true`.\n                type\n                    Sets the layer type, that is the how the layer\n                    data set in `source` will be rendered With\n                    `sourcetype` set to "geojson", the following\n                    values are allowed: "circle", "line", "fill"\n                    and "symbol". but note that "line" and "fill"\n                    are not compatible with Point GeoJSON\n                    geometries. With `sourcetype` set to "vector",\n                    the following values are allowed:  "circle",\n                    "line", "fill" and "symbol". With `sourcetype`\n                    set to "raster" or `*image*`, only the "raster"\n                    value is allowed.\n                visible\n                    Determines whether this layer is displayed\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.layout.mapbox.Layer]\n        '
        return self['layers']

    @layers.setter
    def layers(self, val):
        if False:
            return 10
        self['layers'] = val

    @property
    def layerdefaults(self):
        if False:
            i = 10
            return i + 15
        "\n        When used in a template (as\n        layout.template.layout.mapbox.layerdefaults), sets the default\n        property values to use for elements of layout.mapbox.layers\n\n        The 'layerdefaults' property is an instance of Layer\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.mapbox.Layer`\n          - A dict of string/value properties that will be passed\n            to the Layer constructor\n\n            Supported dict properties:\n\n        Returns\n        -------\n        plotly.graph_objs.layout.mapbox.Layer\n        "
        return self['layerdefaults']

    @layerdefaults.setter
    def layerdefaults(self, val):
        if False:
            while True:
                i = 10
        self['layerdefaults'] = val

    @property
    def pitch(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the pitch angle of the map (in degrees, where 0 means\n        perpendicular to the surface of the map) (mapbox.pitch).\n\n        The 'pitch' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['pitch']

    @pitch.setter
    def pitch(self, val):
        if False:
            while True:
                i = 10
        self['pitch'] = val

    @property
    def style(self):
        if False:
            return 10
        "\n        Defines the map layers that are rendered by default below the\n        trace layers defined in `data`, which are themselves by default\n        rendered below the layers defined in `layout.mapbox.layers`.\n        These layers can be defined either explicitly as a Mapbox Style\n        object which can contain multiple layer definitions that load\n        data from any public or private Tile Map Service (TMS or XYZ)\n        or Web Map Service (WMS) or implicitly by using one of the\n        built-in style objects which use WMSes which do not require any\n        access tokens, or by using a default Mapbox style or custom\n        Mapbox style URL, both of which require a Mapbox access token\n        Note that Mapbox access token can be set in the `accesstoken`\n        attribute or in the `mapboxAccessToken` config option.  Mapbox\n        Style objects are of the form described in the Mapbox GL JS\n        documentation available at https://docs.mapbox.com/mapbox-gl-\n        js/style-spec  The built-in plotly.js styles objects are:\n        carto-darkmatter, carto-positron, open-street-map, stamen-\n        terrain, stamen-toner, stamen-watercolor, white-bg  The built-\n        in Mapbox styles are: basic, streets, outdoors, light, dark,\n        satellite, satellite-streets  Mapbox style URLs are of the\n        form: mapbox://mapbox.mapbox-<name>-<version>\n\n        The 'style' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['style']

    @style.setter
    def style(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['style'] = val

    @property
    def uirevision(self):
        if False:
            print('Hello World!')
        "\n        Controls persistence of user-driven changes in the view:\n        `center`, `zoom`, `bearing`, `pitch`. Defaults to\n        `layout.uirevision`.\n\n        The 'uirevision' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['uirevision'] = val

    @property
    def zoom(self):
        if False:
            print('Hello World!')
        "\n        Sets the zoom level of the map (mapbox.zoom).\n\n        The 'zoom' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['zoom']

    @zoom.setter
    def zoom(self, val):
        if False:
            print('Hello World!')
        self['zoom'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        accesstoken\n            Sets the mapbox access token to be used for this mapbox\n            map. Alternatively, the mapbox access token can be set\n            in the configuration options under `mapboxAccessToken`.\n            Note that accessToken are only required when `style`\n            (e.g with values : basic, streets, outdoors, light,\n            dark, satellite, satellite-streets ) and/or a layout\n            layer references the Mapbox server.\n        bearing\n            Sets the bearing angle of the map in degrees counter-\n            clockwise from North (mapbox.bearing).\n        bounds\n            :class:`plotly.graph_objects.layout.mapbox.Bounds`\n            instance or dict with compatible properties\n        center\n            :class:`plotly.graph_objects.layout.mapbox.Center`\n            instance or dict with compatible properties\n        domain\n            :class:`plotly.graph_objects.layout.mapbox.Domain`\n            instance or dict with compatible properties\n        layers\n            A tuple of\n            :class:`plotly.graph_objects.layout.mapbox.Layer`\n            instances or dicts with compatible properties\n        layerdefaults\n            When used in a template (as\n            layout.template.layout.mapbox.layerdefaults), sets the\n            default property values to use for elements of\n            layout.mapbox.layers\n        pitch\n            Sets the pitch angle of the map (in degrees, where 0\n            means perpendicular to the surface of the map)\n            (mapbox.pitch).\n        style\n            Defines the map layers that are rendered by default\n            below the trace layers defined in `data`, which are\n            themselves by default rendered below the layers defined\n            in `layout.mapbox.layers`.  These layers can be defined\n            either explicitly as a Mapbox Style object which can\n            contain multiple layer definitions that load data from\n            any public or private Tile Map Service (TMS or XYZ) or\n            Web Map Service (WMS) or implicitly by using one of the\n            built-in style objects which use WMSes which do not\n            require any access tokens, or by using a default Mapbox\n            style or custom Mapbox style URL, both of which require\n            a Mapbox access token  Note that Mapbox access token\n            can be set in the `accesstoken` attribute or in the\n            `mapboxAccessToken` config option.  Mapbox Style\n            objects are of the form described in the Mapbox GL JS\n            documentation available at\n            https://docs.mapbox.com/mapbox-gl-js/style-spec  The\n            built-in plotly.js styles objects are: carto-\n            darkmatter, carto-positron, open-street-map, stamen-\n            terrain, stamen-toner, stamen-watercolor, white-bg  The\n            built-in Mapbox styles are: basic, streets, outdoors,\n            light, dark, satellite, satellite-streets  Mapbox style\n            URLs are of the form:\n            mapbox://mapbox.mapbox-<name>-<version>\n        uirevision\n            Controls persistence of user-driven changes in the\n            view: `center`, `zoom`, `bearing`, `pitch`. Defaults to\n            `layout.uirevision`.\n        zoom\n            Sets the zoom level of the map (mapbox.zoom).\n        '

    def __init__(self, arg=None, accesstoken=None, bearing=None, bounds=None, center=None, domain=None, layers=None, layerdefaults=None, pitch=None, style=None, uirevision=None, zoom=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Mapbox object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.layout.Mapbox`\n        accesstoken\n            Sets the mapbox access token to be used for this mapbox\n            map. Alternatively, the mapbox access token can be set\n            in the configuration options under `mapboxAccessToken`.\n            Note that accessToken are only required when `style`\n            (e.g with values : basic, streets, outdoors, light,\n            dark, satellite, satellite-streets ) and/or a layout\n            layer references the Mapbox server.\n        bearing\n            Sets the bearing angle of the map in degrees counter-\n            clockwise from North (mapbox.bearing).\n        bounds\n            :class:`plotly.graph_objects.layout.mapbox.Bounds`\n            instance or dict with compatible properties\n        center\n            :class:`plotly.graph_objects.layout.mapbox.Center`\n            instance or dict with compatible properties\n        domain\n            :class:`plotly.graph_objects.layout.mapbox.Domain`\n            instance or dict with compatible properties\n        layers\n            A tuple of\n            :class:`plotly.graph_objects.layout.mapbox.Layer`\n            instances or dicts with compatible properties\n        layerdefaults\n            When used in a template (as\n            layout.template.layout.mapbox.layerdefaults), sets the\n            default property values to use for elements of\n            layout.mapbox.layers\n        pitch\n            Sets the pitch angle of the map (in degrees, where 0\n            means perpendicular to the surface of the map)\n            (mapbox.pitch).\n        style\n            Defines the map layers that are rendered by default\n            below the trace layers defined in `data`, which are\n            themselves by default rendered below the layers defined\n            in `layout.mapbox.layers`.  These layers can be defined\n            either explicitly as a Mapbox Style object which can\n            contain multiple layer definitions that load data from\n            any public or private Tile Map Service (TMS or XYZ) or\n            Web Map Service (WMS) or implicitly by using one of the\n            built-in style objects which use WMSes which do not\n            require any access tokens, or by using a default Mapbox\n            style or custom Mapbox style URL, both of which require\n            a Mapbox access token  Note that Mapbox access token\n            can be set in the `accesstoken` attribute or in the\n            `mapboxAccessToken` config option.  Mapbox Style\n            objects are of the form described in the Mapbox GL JS\n            documentation available at\n            https://docs.mapbox.com/mapbox-gl-js/style-spec  The\n            built-in plotly.js styles objects are: carto-\n            darkmatter, carto-positron, open-street-map, stamen-\n            terrain, stamen-toner, stamen-watercolor, white-bg  The\n            built-in Mapbox styles are: basic, streets, outdoors,\n            light, dark, satellite, satellite-streets  Mapbox style\n            URLs are of the form:\n            mapbox://mapbox.mapbox-<name>-<version>\n        uirevision\n            Controls persistence of user-driven changes in the\n            view: `center`, `zoom`, `bearing`, `pitch`. Defaults to\n            `layout.uirevision`.\n        zoom\n            Sets the zoom level of the map (mapbox.zoom).\n\n        Returns\n        -------\n        Mapbox\n        '
        super(Mapbox, self).__init__('mapbox')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Mapbox\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Mapbox`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('accesstoken', None)
        _v = accesstoken if accesstoken is not None else _v
        if _v is not None:
            self['accesstoken'] = _v
        _v = arg.pop('bearing', None)
        _v = bearing if bearing is not None else _v
        if _v is not None:
            self['bearing'] = _v
        _v = arg.pop('bounds', None)
        _v = bounds if bounds is not None else _v
        if _v is not None:
            self['bounds'] = _v
        _v = arg.pop('center', None)
        _v = center if center is not None else _v
        if _v is not None:
            self['center'] = _v
        _v = arg.pop('domain', None)
        _v = domain if domain is not None else _v
        if _v is not None:
            self['domain'] = _v
        _v = arg.pop('layers', None)
        _v = layers if layers is not None else _v
        if _v is not None:
            self['layers'] = _v
        _v = arg.pop('layerdefaults', None)
        _v = layerdefaults if layerdefaults is not None else _v
        if _v is not None:
            self['layerdefaults'] = _v
        _v = arg.pop('pitch', None)
        _v = pitch if pitch is not None else _v
        if _v is not None:
            self['pitch'] = _v
        _v = arg.pop('style', None)
        _v = style if style is not None else _v
        if _v is not None:
            self['style'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('zoom', None)
        _v = zoom if zoom is not None else _v
        if _v is not None:
            self['zoom'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
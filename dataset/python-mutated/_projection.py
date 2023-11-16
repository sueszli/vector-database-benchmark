from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Projection(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.geo'
    _path_str = 'layout.geo.projection'
    _valid_props = {'distance', 'parallels', 'rotation', 'scale', 'tilt', 'type'}

    @property
    def distance(self):
        if False:
            return 10
        "\n        For satellite projection type only. Sets the distance from the\n        center of the sphere to the point of view as a proportion of\n        the sphere’s radius.\n\n        The 'distance' property is a number and may be specified as:\n          - An int or float in the interval [1.001, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['distance']

    @distance.setter
    def distance(self, val):
        if False:
            i = 10
            return i + 15
        self['distance'] = val

    @property
    def parallels(self):
        if False:
            i = 10
            return i + 15
        "\n            For conic projection types only. Sets the parallels (tangent,\n            secant) where the cone intersects the sphere.\n\n            The 'parallels' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The 'parallels[0]' property is a number and may be specified as:\n              - An int or float\n        (1) The 'parallels[1]' property is a number and may be specified as:\n              - An int or float\n\n            Returns\n            -------\n            list\n        "
        return self['parallels']

    @parallels.setter
    def parallels(self, val):
        if False:
            i = 10
            return i + 15
        self['parallels'] = val

    @property
    def rotation(self):
        if False:
            print('Hello World!')
        "\n        The 'rotation' property is an instance of Rotation\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.geo.projection.Rotation`\n          - A dict of string/value properties that will be passed\n            to the Rotation constructor\n\n            Supported dict properties:\n\n                lat\n                    Rotates the map along meridians (in degrees\n                    North).\n                lon\n                    Rotates the map along parallels (in degrees\n                    East). Defaults to the center of the\n                    `lonaxis.range` values.\n                roll\n                    Roll the map (in degrees) For example, a roll\n                    of 180 makes the map appear upside down.\n\n        Returns\n        -------\n        plotly.graph_objs.layout.geo.projection.Rotation\n        "
        return self['rotation']

    @rotation.setter
    def rotation(self, val):
        if False:
            while True:
                i = 10
        self['rotation'] = val

    @property
    def scale(self):
        if False:
            print('Hello World!')
        "\n        Zooms in or out on the map view. A scale of 1 corresponds to\n        the largest zoom level that fits the map's lon and lat ranges.\n\n        The 'scale' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['scale']

    @scale.setter
    def scale(self, val):
        if False:
            i = 10
            return i + 15
        self['scale'] = val

    @property
    def tilt(self):
        if False:
            i = 10
            return i + 15
        "\n        For satellite projection type only. Sets the tilt angle of\n        perspective projection.\n\n        The 'tilt' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['tilt']

    @tilt.setter
    def tilt(self, val):
        if False:
            print('Hello World!')
        self['tilt'] = val

    @property
    def type(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the projection type.\n\n        The 'type' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['airy', 'aitoff', 'albers', 'albers usa', 'august',\n                'azimuthal equal area', 'azimuthal equidistant', 'baker',\n                'bertin1953', 'boggs', 'bonne', 'bottomley', 'bromley',\n                'collignon', 'conic conformal', 'conic equal area', 'conic\n                equidistant', 'craig', 'craster', 'cylindrical equal\n                area', 'cylindrical stereographic', 'eckert1', 'eckert2',\n                'eckert3', 'eckert4', 'eckert5', 'eckert6', 'eisenlohr',\n                'equal earth', 'equirectangular', 'fahey', 'foucaut',\n                'foucaut sinusoidal', 'ginzburg4', 'ginzburg5',\n                'ginzburg6', 'ginzburg8', 'ginzburg9', 'gnomonic',\n                'gringorten', 'gringorten quincuncial', 'guyou', 'hammer',\n                'hill', 'homolosine', 'hufnagel', 'hyperelliptical',\n                'kavrayskiy7', 'lagrange', 'larrivee', 'laskowski',\n                'loximuthal', 'mercator', 'miller', 'mollweide', 'mt flat\n                polar parabolic', 'mt flat polar quartic', 'mt flat polar\n                sinusoidal', 'natural earth', 'natural earth1', 'natural\n                earth2', 'nell hammer', 'nicolosi', 'orthographic',\n                'patterson', 'peirce quincuncial', 'polyconic',\n                'rectangular polyconic', 'robinson', 'satellite', 'sinu\n                mollweide', 'sinusoidal', 'stereographic', 'times',\n                'transverse mercator', 'van der grinten', 'van der\n                grinten2', 'van der grinten3', 'van der grinten4',\n                'wagner4', 'wagner6', 'wiechel', 'winkel tripel',\n                'winkel3']\n\n        Returns\n        -------\n        Any\n        "
        return self['type']

    @type.setter
    def type(self, val):
        if False:
            return 10
        self['type'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return "        distance\n            For satellite projection type only. Sets the distance\n            from the center of the sphere to the point of view as a\n            proportion of the sphere’s radius.\n        parallels\n            For conic projection types only. Sets the parallels\n            (tangent, secant) where the cone intersects the sphere.\n        rotation\n            :class:`plotly.graph_objects.layout.geo.projection.Rota\n            tion` instance or dict with compatible properties\n        scale\n            Zooms in or out on the map view. A scale of 1\n            corresponds to the largest zoom level that fits the\n            map's lon and lat ranges.\n        tilt\n            For satellite projection type only. Sets the tilt angle\n            of perspective projection.\n        type\n            Sets the projection type.\n        "

    def __init__(self, arg=None, distance=None, parallels=None, rotation=None, scale=None, tilt=None, type=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Construct a new Projection object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.geo.Projection`\n        distance\n            For satellite projection type only. Sets the distance\n            from the center of the sphere to the point of view as a\n            proportion of the sphere’s radius.\n        parallels\n            For conic projection types only. Sets the parallels\n            (tangent, secant) where the cone intersects the sphere.\n        rotation\n            :class:`plotly.graph_objects.layout.geo.projection.Rota\n            tion` instance or dict with compatible properties\n        scale\n            Zooms in or out on the map view. A scale of 1\n            corresponds to the largest zoom level that fits the\n            map's lon and lat ranges.\n        tilt\n            For satellite projection type only. Sets the tilt angle\n            of perspective projection.\n        type\n            Sets the projection type.\n\n        Returns\n        -------\n        Projection\n        "
        super(Projection, self).__init__('projection')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.geo.Projection\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.geo.Projection`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('distance', None)
        _v = distance if distance is not None else _v
        if _v is not None:
            self['distance'] = _v
        _v = arg.pop('parallels', None)
        _v = parallels if parallels is not None else _v
        if _v is not None:
            self['parallels'] = _v
        _v = arg.pop('rotation', None)
        _v = rotation if rotation is not None else _v
        if _v is not None:
            self['rotation'] = _v
        _v = arg.pop('scale', None)
        _v = scale if scale is not None else _v
        if _v is not None:
            self['scale'] = _v
        _v = arg.pop('tilt', None)
        _v = tilt if tilt is not None else _v
        if _v is not None:
            self['tilt'] = _v
        _v = arg.pop('type', None)
        _v = type if type is not None else _v
        if _v is not None:
            self['type'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
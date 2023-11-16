from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Rotation(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.geo.projection'
    _path_str = 'layout.geo.projection.rotation'
    _valid_props = {'lat', 'lon', 'roll'}

    @property
    def lat(self):
        if False:
            while True:
                i = 10
        "\n        Rotates the map along meridians (in degrees North).\n\n        The 'lat' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['lat']

    @lat.setter
    def lat(self, val):
        if False:
            while True:
                i = 10
        self['lat'] = val

    @property
    def lon(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Rotates the map along parallels (in degrees East). Defaults to\n        the center of the `lonaxis.range` values.\n\n        The 'lon' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['lon']

    @lon.setter
    def lon(self, val):
        if False:
            print('Hello World!')
        self['lon'] = val

    @property
    def roll(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Roll the map (in degrees) For example, a roll of 180 makes the\n        map appear upside down.\n\n        The 'roll' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['roll']

    @roll.setter
    def roll(self, val):
        if False:
            while True:
                i = 10
        self['roll'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        lat\n            Rotates the map along meridians (in degrees North).\n        lon\n            Rotates the map along parallels (in degrees East).\n            Defaults to the center of the `lonaxis.range` values.\n        roll\n            Roll the map (in degrees) For example, a roll of 180\n            makes the map appear upside down.\n        '

    def __init__(self, arg=None, lat=None, lon=None, roll=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Rotation object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.layout.geo.pro\n            jection.Rotation`\n        lat\n            Rotates the map along meridians (in degrees North).\n        lon\n            Rotates the map along parallels (in degrees East).\n            Defaults to the center of the `lonaxis.range` values.\n        roll\n            Roll the map (in degrees) For example, a roll of 180\n            makes the map appear upside down.\n\n        Returns\n        -------\n        Rotation\n        '
        super(Rotation, self).__init__('rotation')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.geo.projection.Rotation\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.geo.projection.Rotation`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('lat', None)
        _v = lat if lat is not None else _v
        if _v is not None:
            self['lat'] = _v
        _v = arg.pop('lon', None)
        _v = lon if lon is not None else _v
        if _v is not None:
            self['lon'] = _v
        _v = arg.pop('roll', None)
        _v = roll if roll is not None else _v
        if _v is not None:
            self['roll'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
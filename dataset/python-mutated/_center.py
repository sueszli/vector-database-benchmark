from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Center(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.mapbox'
    _path_str = 'layout.mapbox.center'
    _valid_props = {'lat', 'lon'}

    @property
    def lat(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the latitude of the center of the map (in degrees North).\n\n        The 'lat' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
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
            while True:
                i = 10
        "\n        Sets the longitude of the center of the map (in degrees East).\n\n        The 'lon' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['lon']

    @lon.setter
    def lon(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['lon'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        lat\n            Sets the latitude of the center of the map (in degrees\n            North).\n        lon\n            Sets the longitude of the center of the map (in degrees\n            East).\n        '

    def __init__(self, arg=None, lat=None, lon=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Center object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.mapbox.Center`\n        lat\n            Sets the latitude of the center of the map (in degrees\n            North).\n        lon\n            Sets the longitude of the center of the map (in degrees\n            East).\n\n        Returns\n        -------\n        Center\n        '
        super(Center, self).__init__('center')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.mapbox.Center\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.mapbox.Center`')
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
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
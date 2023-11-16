from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Circle(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.mapbox.layer'
    _path_str = 'layout.mapbox.layer.circle'
    _valid_props = {'radius'}

    @property
    def radius(self):
        if False:
            while True:
                i = 10
        '\n        Sets the circle radius (mapbox.layer.paint.circle-radius). Has\n        an effect only when `type` is set to "circle".\n\n        The \'radius\' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        '
        return self['radius']

    @radius.setter
    def radius(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['radius'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        radius\n            Sets the circle radius (mapbox.layer.paint.circle-\n            radius). Has an effect only when `type` is set to\n            "circle".\n        '

    def __init__(self, arg=None, radius=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Circle object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.mapbox.layer.Circle`\n        radius\n            Sets the circle radius (mapbox.layer.paint.circle-\n            radius). Has an effect only when `type` is set to\n            "circle".\n\n        Returns\n        -------\n        Circle\n        '
        super(Circle, self).__init__('circle')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.mapbox.layer.Circle\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.mapbox.layer.Circle`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('radius', None)
        _v = radius if radius is not None else _v
        if _v is not None:
            self['radius'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
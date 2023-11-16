from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Decreasing(_BaseTraceHierarchyType):
    _parent_path_str = 'waterfall'
    _path_str = 'waterfall.decreasing'
    _valid_props = {'marker'}

    @property
    def marker(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'marker' property is an instance of Marker\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.waterfall.decreasing.Marker`\n          - A dict of string/value properties that will be passed\n            to the Marker constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the marker color of all decreasing values.\n                line\n                    :class:`plotly.graph_objects.waterfall.decreasi\n                    ng.marker.Line` instance or dict with\n                    compatible properties\n\n        Returns\n        -------\n        plotly.graph_objs.waterfall.decreasing.Marker\n        "
        return self['marker']

    @marker.setter
    def marker(self, val):
        if False:
            while True:
                i = 10
        self['marker'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        marker\n            :class:`plotly.graph_objects.waterfall.decreasing.Marke\n            r` instance or dict with compatible properties\n        '

    def __init__(self, arg=None, marker=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Decreasing object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.waterfall.Decreasing`\n        marker\n            :class:`plotly.graph_objects.waterfall.decreasing.Marke\n            r` instance or dict with compatible properties\n\n        Returns\n        -------\n        Decreasing\n        '
        super(Decreasing, self).__init__('decreasing')
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
            raise ValueError('The first argument to the plotly.graph_objs.waterfall.Decreasing\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.waterfall.Decreasing`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('marker', None)
        _v = marker if marker is not None else _v
        if _v is not None:
            self['marker'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
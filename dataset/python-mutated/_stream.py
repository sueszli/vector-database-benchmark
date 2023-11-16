from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Stream(_BaseTraceHierarchyType):
    _parent_path_str = 'pointcloud'
    _path_str = 'pointcloud.stream'
    _valid_props = {'maxpoints', 'token'}

    @property
    def maxpoints(self):
        if False:
            return 10
        "\n        Sets the maximum number of points to keep on the plots from an\n        incoming stream. If `maxpoints` is set to 50, only the newest\n        50 points will be displayed on the plot.\n\n        The 'maxpoints' property is a number and may be specified as:\n          - An int or float in the interval [0, 10000]\n\n        Returns\n        -------\n        int|float\n        "
        return self['maxpoints']

    @maxpoints.setter
    def maxpoints(self, val):
        if False:
            print('Hello World!')
        self['maxpoints'] = val

    @property
    def token(self):
        if False:
            print('Hello World!')
        "\n        The stream id number links a data trace on a plot with a\n        stream. See https://chart-studio.plotly.com/settings for more\n        details.\n\n        The 'token' property is a string and must be specified as:\n          - A non-empty string\n\n        Returns\n        -------\n        str\n        "
        return self['token']

    @token.setter
    def token(self, val):
        if False:
            while True:
                i = 10
        self['token'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        maxpoints\n            Sets the maximum number of points to keep on the plots\n            from an incoming stream. If `maxpoints` is set to 50,\n            only the newest 50 points will be displayed on the\n            plot.\n        token\n            The stream id number links a data trace on a plot with\n            a stream. See https://chart-studio.plotly.com/settings\n            for more details.\n        '

    def __init__(self, arg=None, maxpoints=None, token=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Stream object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.pointcloud.Stream`\n        maxpoints\n            Sets the maximum number of points to keep on the plots\n            from an incoming stream. If `maxpoints` is set to 50,\n            only the newest 50 points will be displayed on the\n            plot.\n        token\n            The stream id number links a data trace on a plot with\n            a stream. See https://chart-studio.plotly.com/settings\n            for more details.\n\n        Returns\n        -------\n        Stream\n        '
        super(Stream, self).__init__('stream')
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
            raise ValueError('The first argument to the plotly.graph_objs.pointcloud.Stream\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.pointcloud.Stream`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('maxpoints', None)
        _v = maxpoints if maxpoints is not None else _v
        if _v is not None:
            self['maxpoints'] = _v
        _v = arg.pop('token', None)
        _v = token if token is not None else _v
        if _v is not None:
            self['token'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
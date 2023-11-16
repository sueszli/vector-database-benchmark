from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Starts(_BaseTraceHierarchyType):
    _parent_path_str = 'streamtube'
    _path_str = 'streamtube.starts'
    _valid_props = {'x', 'xsrc', 'y', 'ysrc', 'z', 'zsrc'}

    @property
    def x(self):
        if False:
            return 10
        "\n        Sets the x components of the starting position of the\n        streamtubes\n\n        The 'x' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['x'] = val

    @property
    def xsrc(self):
        if False:
            return 10
        "\n        Sets the source reference on Chart Studio Cloud for `x`.\n\n        The 'xsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['xsrc']

    @xsrc.setter
    def xsrc(self, val):
        if False:
            while True:
                i = 10
        self['xsrc'] = val

    @property
    def y(self):
        if False:
            while True:
                i = 10
        "\n        Sets the y components of the starting position of the\n        streamtubes\n\n        The 'y' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            i = 10
            return i + 15
        self['y'] = val

    @property
    def ysrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `y`.\n\n        The 'ysrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['ysrc']

    @ysrc.setter
    def ysrc(self, val):
        if False:
            print('Hello World!')
        self['ysrc'] = val

    @property
    def z(self):
        if False:
            while True:
                i = 10
        "\n        Sets the z components of the starting position of the\n        streamtubes\n\n        The 'z' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['z']

    @z.setter
    def z(self, val):
        if False:
            print('Hello World!')
        self['z'] = val

    @property
    def zsrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for `z`.\n\n        The 'zsrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['zsrc']

    @zsrc.setter
    def zsrc(self, val):
        if False:
            i = 10
            return i + 15
        self['zsrc'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        x\n            Sets the x components of the starting position of the\n            streamtubes\n        xsrc\n            Sets the source reference on Chart Studio Cloud for\n            `x`.\n        y\n            Sets the y components of the starting position of the\n            streamtubes\n        ysrc\n            Sets the source reference on Chart Studio Cloud for\n            `y`.\n        z\n            Sets the z components of the starting position of the\n            streamtubes\n        zsrc\n            Sets the source reference on Chart Studio Cloud for\n            `z`.\n        '

    def __init__(self, arg=None, x=None, xsrc=None, y=None, ysrc=None, z=None, zsrc=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Starts object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.streamtube.Starts`\n        x\n            Sets the x components of the starting position of the\n            streamtubes\n        xsrc\n            Sets the source reference on Chart Studio Cloud for\n            `x`.\n        y\n            Sets the y components of the starting position of the\n            streamtubes\n        ysrc\n            Sets the source reference on Chart Studio Cloud for\n            `y`.\n        z\n            Sets the z components of the starting position of the\n            streamtubes\n        zsrc\n            Sets the source reference on Chart Studio Cloud for\n            `z`.\n\n        Returns\n        -------\n        Starts\n        '
        super(Starts, self).__init__('starts')
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
            raise ValueError('The first argument to the plotly.graph_objs.streamtube.Starts\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.streamtube.Starts`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('xsrc', None)
        _v = xsrc if xsrc is not None else _v
        if _v is not None:
            self['xsrc'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('ysrc', None)
        _v = ysrc if ysrc is not None else _v
        if _v is not None:
            self['ysrc'] = _v
        _v = arg.pop('z', None)
        _v = z if z is not None else _v
        if _v is not None:
            self['z'] = _v
        _v = arg.pop('zsrc', None)
        _v = zsrc if zsrc is not None else _v
        if _v is not None:
            self['zsrc'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
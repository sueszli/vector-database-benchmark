from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Up(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.scene.camera'
    _path_str = 'layout.scene.camera.up'
    _valid_props = {'x', 'y', 'z'}

    @property
    def x(self):
        if False:
            return 10
        "\n        The 'x' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            while True:
                i = 10
        self['x'] = val

    @property
    def y(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'y' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            return 10
        self['y'] = val

    @property
    def z(self):
        if False:
            i = 10
            return i + 15
        "\n        The 'z' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['z']

    @z.setter
    def z(self, val):
        if False:
            print('Hello World!')
        self['z'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        x\n\n        y\n\n        z\n\n        '

    def __init__(self, arg=None, x=None, y=None, z=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a new Up object\n\n        Sets the (x,y,z) components of the 'up' camera vector. This\n        vector determines the up direction of this scene with respect\n        to the page. The default is *{x: 0, y: 0, z: 1}* which means\n        that the z axis points up.\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.scene.camera.Up`\n        x\n\n        y\n\n        z\n\n\n        Returns\n        -------\n        Up\n        "
        super(Up, self).__init__('up')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.scene.camera.Up\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.scene.camera.Up`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        _v = arg.pop('z', None)
        _v = z if z is not None else _v
        if _v is not None:
            self['z'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
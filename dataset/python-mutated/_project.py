from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Project(_BaseTraceHierarchyType):
    _parent_path_str = 'surface.contours.z'
    _path_str = 'surface.contours.z.project'
    _valid_props = {'x', 'y', 'z'}

    @property
    def x(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not these contour lines are projected on\n        the x plane. If `highlight` is set to True (the default), the\n        projected lines are shown on hover. If `show` is set to True,\n        the projected lines are shown in permanence.\n\n        The 'x' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['x'] = val

    @property
    def y(self):
        if False:
            print('Hello World!')
        "\n        Determines whether or not these contour lines are projected on\n        the y plane. If `highlight` is set to True (the default), the\n        projected lines are shown on hover. If `show` is set to True,\n        the projected lines are shown in permanence.\n\n        The 'y' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            return 10
        self['y'] = val

    @property
    def z(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not these contour lines are projected on\n        the z plane. If `highlight` is set to True (the default), the\n        projected lines are shown on hover. If `show` is set to True,\n        the projected lines are shown in permanence.\n\n        The 'z' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['z']

    @z.setter
    def z(self, val):
        if False:
            while True:
                i = 10
        self['z'] = val

    @property
    def _prop_descriptions(self):
        if False:
            i = 10
            return i + 15
        return '        x\n            Determines whether or not these contour lines are\n            projected on the x plane. If `highlight` is set to True\n            (the default), the projected lines are shown on hover.\n            If `show` is set to True, the projected lines are shown\n            in permanence.\n        y\n            Determines whether or not these contour lines are\n            projected on the y plane. If `highlight` is set to True\n            (the default), the projected lines are shown on hover.\n            If `show` is set to True, the projected lines are shown\n            in permanence.\n        z\n            Determines whether or not these contour lines are\n            projected on the z plane. If `highlight` is set to True\n            (the default), the projected lines are shown on hover.\n            If `show` is set to True, the projected lines are shown\n            in permanence.\n        '

    def __init__(self, arg=None, x=None, y=None, z=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Project object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.surface.contours.z.Project`\n        x\n            Determines whether or not these contour lines are\n            projected on the x plane. If `highlight` is set to True\n            (the default), the projected lines are shown on hover.\n            If `show` is set to True, the projected lines are shown\n            in permanence.\n        y\n            Determines whether or not these contour lines are\n            projected on the y plane. If `highlight` is set to True\n            (the default), the projected lines are shown on hover.\n            If `show` is set to True, the projected lines are shown\n            in permanence.\n        z\n            Determines whether or not these contour lines are\n            projected on the z plane. If `highlight` is set to True\n            (the default), the projected lines are shown on hover.\n            If `show` is set to True, the projected lines are shown\n            in permanence.\n\n        Returns\n        -------\n        Project\n        '
        super(Project, self).__init__('project')
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
            raise ValueError('The first argument to the plotly.graph_objs.surface.contours.z.Project\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.surface.contours.z.Project`')
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
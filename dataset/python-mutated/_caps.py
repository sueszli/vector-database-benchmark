from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Caps(_BaseTraceHierarchyType):
    _parent_path_str = 'isosurface'
    _path_str = 'isosurface.caps'
    _valid_props = {'x', 'y', 'z'}

    @property
    def x(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'x' property is an instance of X\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.isosurface.caps.X`\n          - A dict of string/value properties that will be passed\n            to the X constructor\n\n            Supported dict properties:\n\n                fill\n                    Sets the fill ratio of the `caps`. The default\n                    fill value of the `caps` is 1 meaning that they\n                    are entirely shaded. On the other hand Applying\n                    a `fill` ratio less than one would allow the\n                    creation of openings parallel to the edges.\n                show\n                    Sets the fill ratio of the `slices`. The\n                    default fill value of the x `slices` is 1\n                    meaning that they are entirely shaded. On the\n                    other hand Applying a `fill` ratio less than\n                    one would allow the creation of openings\n                    parallel to the edges.\n\n        Returns\n        -------\n        plotly.graph_objs.isosurface.caps.X\n        "
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
            for i in range(10):
                print('nop')
        "\n        The 'y' property is an instance of Y\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.isosurface.caps.Y`\n          - A dict of string/value properties that will be passed\n            to the Y constructor\n\n            Supported dict properties:\n\n                fill\n                    Sets the fill ratio of the `caps`. The default\n                    fill value of the `caps` is 1 meaning that they\n                    are entirely shaded. On the other hand Applying\n                    a `fill` ratio less than one would allow the\n                    creation of openings parallel to the edges.\n                show\n                    Sets the fill ratio of the `slices`. The\n                    default fill value of the y `slices` is 1\n                    meaning that they are entirely shaded. On the\n                    other hand Applying a `fill` ratio less than\n                    one would allow the creation of openings\n                    parallel to the edges.\n\n        Returns\n        -------\n        plotly.graph_objs.isosurface.caps.Y\n        "
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
        "\n        The 'z' property is an instance of Z\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.isosurface.caps.Z`\n          - A dict of string/value properties that will be passed\n            to the Z constructor\n\n            Supported dict properties:\n\n                fill\n                    Sets the fill ratio of the `caps`. The default\n                    fill value of the `caps` is 1 meaning that they\n                    are entirely shaded. On the other hand Applying\n                    a `fill` ratio less than one would allow the\n                    creation of openings parallel to the edges.\n                show\n                    Sets the fill ratio of the `slices`. The\n                    default fill value of the z `slices` is 1\n                    meaning that they are entirely shaded. On the\n                    other hand Applying a `fill` ratio less than\n                    one would allow the creation of openings\n                    parallel to the edges.\n\n        Returns\n        -------\n        plotly.graph_objs.isosurface.caps.Z\n        "
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
        return '        x\n            :class:`plotly.graph_objects.isosurface.caps.X`\n            instance or dict with compatible properties\n        y\n            :class:`plotly.graph_objects.isosurface.caps.Y`\n            instance or dict with compatible properties\n        z\n            :class:`plotly.graph_objects.isosurface.caps.Z`\n            instance or dict with compatible properties\n        '

    def __init__(self, arg=None, x=None, y=None, z=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Caps object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.isosurface.Caps`\n        x\n            :class:`plotly.graph_objects.isosurface.caps.X`\n            instance or dict with compatible properties\n        y\n            :class:`plotly.graph_objects.isosurface.caps.Y`\n            instance or dict with compatible properties\n        z\n            :class:`plotly.graph_objects.isosurface.caps.Z`\n            instance or dict with compatible properties\n\n        Returns\n        -------\n        Caps\n        '
        super(Caps, self).__init__('caps')
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
            raise ValueError('The first argument to the plotly.graph_objs.isosurface.Caps\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.isosurface.Caps`')
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
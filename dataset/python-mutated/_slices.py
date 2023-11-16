from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Slices(_BaseTraceHierarchyType):
    _parent_path_str = 'isosurface'
    _path_str = 'isosurface.slices'
    _valid_props = {'x', 'y', 'z'}

    @property
    def x(self):
        if False:
            print('Hello World!')
        "\n        The 'x' property is an instance of X\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.isosurface.slices.X`\n          - A dict of string/value properties that will be passed\n            to the X constructor\n\n            Supported dict properties:\n\n                fill\n                    Sets the fill ratio of the `slices`. The\n                    default fill value of the `slices` is 1 meaning\n                    that they are entirely shaded. On the other\n                    hand Applying a `fill` ratio less than one\n                    would allow the creation of openings parallel\n                    to the edges.\n                locations\n                    Specifies the location(s) of slices on the\n                    axis. When not specified slices would be\n                    created for all points of the axis x except\n                    start and end.\n                locationssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `locations`.\n                show\n                    Determines whether or not slice planes about\n                    the x dimension are drawn.\n\n        Returns\n        -------\n        plotly.graph_objs.isosurface.slices.X\n        "
        return self['x']

    @x.setter
    def x(self, val):
        if False:
            return 10
        self['x'] = val

    @property
    def y(self):
        if False:
            return 10
        "\n        The 'y' property is an instance of Y\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.isosurface.slices.Y`\n          - A dict of string/value properties that will be passed\n            to the Y constructor\n\n            Supported dict properties:\n\n                fill\n                    Sets the fill ratio of the `slices`. The\n                    default fill value of the `slices` is 1 meaning\n                    that they are entirely shaded. On the other\n                    hand Applying a `fill` ratio less than one\n                    would allow the creation of openings parallel\n                    to the edges.\n                locations\n                    Specifies the location(s) of slices on the\n                    axis. When not specified slices would be\n                    created for all points of the axis y except\n                    start and end.\n                locationssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `locations`.\n                show\n                    Determines whether or not slice planes about\n                    the y dimension are drawn.\n\n        Returns\n        -------\n        plotly.graph_objs.isosurface.slices.Y\n        "
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['y'] = val

    @property
    def z(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'z' property is an instance of Z\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.isosurface.slices.Z`\n          - A dict of string/value properties that will be passed\n            to the Z constructor\n\n            Supported dict properties:\n\n                fill\n                    Sets the fill ratio of the `slices`. The\n                    default fill value of the `slices` is 1 meaning\n                    that they are entirely shaded. On the other\n                    hand Applying a `fill` ratio less than one\n                    would allow the creation of openings parallel\n                    to the edges.\n                locations\n                    Specifies the location(s) of slices on the\n                    axis. When not specified slices would be\n                    created for all points of the axis z except\n                    start and end.\n                locationssrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `locations`.\n                show\n                    Determines whether or not slice planes about\n                    the z dimension are drawn.\n\n        Returns\n        -------\n        plotly.graph_objs.isosurface.slices.Z\n        "
        return self['z']

    @z.setter
    def z(self, val):
        if False:
            print('Hello World!')
        self['z'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        x\n            :class:`plotly.graph_objects.isosurface.slices.X`\n            instance or dict with compatible properties\n        y\n            :class:`plotly.graph_objects.isosurface.slices.Y`\n            instance or dict with compatible properties\n        z\n            :class:`plotly.graph_objects.isosurface.slices.Z`\n            instance or dict with compatible properties\n        '

    def __init__(self, arg=None, x=None, y=None, z=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Slices object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.isosurface.Slices`\n        x\n            :class:`plotly.graph_objects.isosurface.slices.X`\n            instance or dict with compatible properties\n        y\n            :class:`plotly.graph_objects.isosurface.slices.Y`\n            instance or dict with compatible properties\n        z\n            :class:`plotly.graph_objects.isosurface.slices.Z`\n            instance or dict with compatible properties\n\n        Returns\n        -------\n        Slices\n        '
        super(Slices, self).__init__('slices')
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
            raise ValueError('The first argument to the plotly.graph_objs.isosurface.Slices\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.isosurface.Slices`')
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
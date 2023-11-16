from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Contours(_BaseTraceHierarchyType):
    _parent_path_str = 'surface'
    _path_str = 'surface.contours'
    _valid_props = {'x', 'y', 'z'}

    @property
    def x(self):
        if False:
            return 10
        '\n        The \'x\' property is an instance of X\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.surface.contours.X`\n          - A dict of string/value properties that will be passed\n            to the X constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the color of the contour lines.\n                end\n                    Sets the end contour level value. Must be more\n                    than `contours.start`\n                highlight\n                    Determines whether or not contour lines about\n                    the x dimension are highlighted on hover.\n                highlightcolor\n                    Sets the color of the highlighted contour\n                    lines.\n                highlightwidth\n                    Sets the width of the highlighted contour\n                    lines.\n                project\n                    :class:`plotly.graph_objects.surface.contours.x\n                    .Project` instance or dict with compatible\n                    properties\n                show\n                    Determines whether or not contour lines about\n                    the x dimension are drawn.\n                size\n                    Sets the step between each contour level. Must\n                    be positive.\n                start\n                    Sets the starting contour level value. Must be\n                    less than `contours.end`\n                usecolormap\n                    An alternate to "color". Determines whether or\n                    not the contour lines are colored using the\n                    trace "colorscale".\n                width\n                    Sets the width of the contour lines.\n\n        Returns\n        -------\n        plotly.graph_objs.surface.contours.X\n        '
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
            while True:
                i = 10
        '\n        The \'y\' property is an instance of Y\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.surface.contours.Y`\n          - A dict of string/value properties that will be passed\n            to the Y constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the color of the contour lines.\n                end\n                    Sets the end contour level value. Must be more\n                    than `contours.start`\n                highlight\n                    Determines whether or not contour lines about\n                    the y dimension are highlighted on hover.\n                highlightcolor\n                    Sets the color of the highlighted contour\n                    lines.\n                highlightwidth\n                    Sets the width of the highlighted contour\n                    lines.\n                project\n                    :class:`plotly.graph_objects.surface.contours.y\n                    .Project` instance or dict with compatible\n                    properties\n                show\n                    Determines whether or not contour lines about\n                    the y dimension are drawn.\n                size\n                    Sets the step between each contour level. Must\n                    be positive.\n                start\n                    Sets the starting contour level value. Must be\n                    less than `contours.end`\n                usecolormap\n                    An alternate to "color". Determines whether or\n                    not the contour lines are colored using the\n                    trace "colorscale".\n                width\n                    Sets the width of the contour lines.\n\n        Returns\n        -------\n        plotly.graph_objs.surface.contours.Y\n        '
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
        '\n        The \'z\' property is an instance of Z\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.surface.contours.Z`\n          - A dict of string/value properties that will be passed\n            to the Z constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the color of the contour lines.\n                end\n                    Sets the end contour level value. Must be more\n                    than `contours.start`\n                highlight\n                    Determines whether or not contour lines about\n                    the z dimension are highlighted on hover.\n                highlightcolor\n                    Sets the color of the highlighted contour\n                    lines.\n                highlightwidth\n                    Sets the width of the highlighted contour\n                    lines.\n                project\n                    :class:`plotly.graph_objects.surface.contours.z\n                    .Project` instance or dict with compatible\n                    properties\n                show\n                    Determines whether or not contour lines about\n                    the z dimension are drawn.\n                size\n                    Sets the step between each contour level. Must\n                    be positive.\n                start\n                    Sets the starting contour level value. Must be\n                    less than `contours.end`\n                usecolormap\n                    An alternate to "color". Determines whether or\n                    not the contour lines are colored using the\n                    trace "colorscale".\n                width\n                    Sets the width of the contour lines.\n\n        Returns\n        -------\n        plotly.graph_objs.surface.contours.Z\n        '
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
            return 10
        return '        x\n            :class:`plotly.graph_objects.surface.contours.X`\n            instance or dict with compatible properties\n        y\n            :class:`plotly.graph_objects.surface.contours.Y`\n            instance or dict with compatible properties\n        z\n            :class:`plotly.graph_objects.surface.contours.Z`\n            instance or dict with compatible properties\n        '

    def __init__(self, arg=None, x=None, y=None, z=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Contours object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.surface.Contours`\n        x\n            :class:`plotly.graph_objects.surface.contours.X`\n            instance or dict with compatible properties\n        y\n            :class:`plotly.graph_objects.surface.contours.Y`\n            instance or dict with compatible properties\n        z\n            :class:`plotly.graph_objects.surface.contours.Z`\n            instance or dict with compatible properties\n\n        Returns\n        -------\n        Contours\n        '
        super(Contours, self).__init__('contours')
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
            raise ValueError('The first argument to the plotly.graph_objs.surface.Contours\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.surface.Contours`')
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
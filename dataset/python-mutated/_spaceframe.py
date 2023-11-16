from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Spaceframe(_BaseTraceHierarchyType):
    _parent_path_str = 'isosurface'
    _path_str = 'isosurface.spaceframe'
    _valid_props = {'fill', 'show'}

    @property
    def fill(self):
        if False:
            while True:
                i = 10
        "\n        Sets the fill ratio of the `spaceframe` elements. The default\n        fill value is 0.15 meaning that only 15% of the area of every\n        faces of tetras would be shaded. Applying a greater `fill`\n        ratio would allow the creation of stronger elements or could be\n        sued to have entirely closed areas (in case of using 1).\n\n        The 'fill' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['fill']

    @fill.setter
    def fill(self, val):
        if False:
            i = 10
            return i + 15
        self['fill'] = val

    @property
    def show(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Displays/hides tetrahedron shapes between minimum and maximum\n        iso-values. Often useful when either caps or surfaces are\n        disabled or filled with values less than 1.\n\n        The 'show' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['show']

    @show.setter
    def show(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['show'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        fill\n            Sets the fill ratio of the `spaceframe` elements. The\n            default fill value is 0.15 meaning that only 15% of the\n            area of every faces of tetras would be shaded. Applying\n            a greater `fill` ratio would allow the creation of\n            stronger elements or could be sued to have entirely\n            closed areas (in case of using 1).\n        show\n            Displays/hides tetrahedron shapes between minimum and\n            maximum iso-values. Often useful when either caps or\n            surfaces are disabled or filled with values less than\n            1.\n        '

    def __init__(self, arg=None, fill=None, show=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Spaceframe object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.isosurface.Spaceframe`\n        fill\n            Sets the fill ratio of the `spaceframe` elements. The\n            default fill value is 0.15 meaning that only 15% of the\n            area of every faces of tetras would be shaded. Applying\n            a greater `fill` ratio would allow the creation of\n            stronger elements or could be sued to have entirely\n            closed areas (in case of using 1).\n        show\n            Displays/hides tetrahedron shapes between minimum and\n            maximum iso-values. Often useful when either caps or\n            surfaces are disabled or filled with values less than\n            1.\n\n        Returns\n        -------\n        Spaceframe\n        '
        super(Spaceframe, self).__init__('spaceframe')
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
            raise ValueError('The first argument to the plotly.graph_objs.isosurface.Spaceframe\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.isosurface.Spaceframe`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('fill', None)
        _v = fill if fill is not None else _v
        if _v is not None:
            self['fill'] = _v
        _v = arg.pop('show', None)
        _v = show if show is not None else _v
        if _v is not None:
            self['show'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
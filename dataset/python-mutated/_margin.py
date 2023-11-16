from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Margin(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.margin'
    _valid_props = {'autoexpand', 'b', 'l', 'pad', 'r', 't'}

    @property
    def autoexpand(self):
        if False:
            return 10
        "\n        Turns on/off margin expansion computations. Legends, colorbars,\n        updatemenus, sliders, axis rangeselector and rangeslider are\n        allowed to push the margins by defaults.\n\n        The 'autoexpand' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['autoexpand']

    @autoexpand.setter
    def autoexpand(self, val):
        if False:
            return 10
        self['autoexpand'] = val

    @property
    def b(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the bottom margin (in px).\n\n        The 'b' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['b']

    @b.setter
    def b(self, val):
        if False:
            return 10
        self['b'] = val

    @property
    def l(self):
        if False:
            while True:
                i = 10
        "\n        Sets the left margin (in px).\n\n        The 'l' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['l']

    @l.setter
    def l(self, val):
        if False:
            return 10
        self['l'] = val

    @property
    def pad(self):
        if False:
            while True:
                i = 10
        "\n        Sets the amount of padding (in px) between the plotting area\n        and the axis lines\n\n        The 'pad' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['pad']

    @pad.setter
    def pad(self, val):
        if False:
            while True:
                i = 10
        self['pad'] = val

    @property
    def r(self):
        if False:
            print('Hello World!')
        "\n        Sets the right margin (in px).\n\n        The 'r' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['r']

    @r.setter
    def r(self, val):
        if False:
            while True:
                i = 10
        self['r'] = val

    @property
    def t(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the top margin (in px).\n\n        The 't' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['t']

    @t.setter
    def t(self, val):
        if False:
            return 10
        self['t'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        autoexpand\n            Turns on/off margin expansion computations. Legends,\n            colorbars, updatemenus, sliders, axis rangeselector and\n            rangeslider are allowed to push the margins by\n            defaults.\n        b\n            Sets the bottom margin (in px).\n        l\n            Sets the left margin (in px).\n        pad\n            Sets the amount of padding (in px) between the plotting\n            area and the axis lines\n        r\n            Sets the right margin (in px).\n        t\n            Sets the top margin (in px).\n        '

    def __init__(self, arg=None, autoexpand=None, b=None, l=None, pad=None, r=None, t=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Margin object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.layout.Margin`\n        autoexpand\n            Turns on/off margin expansion computations. Legends,\n            colorbars, updatemenus, sliders, axis rangeselector and\n            rangeslider are allowed to push the margins by\n            defaults.\n        b\n            Sets the bottom margin (in px).\n        l\n            Sets the left margin (in px).\n        pad\n            Sets the amount of padding (in px) between the plotting\n            area and the axis lines\n        r\n            Sets the right margin (in px).\n        t\n            Sets the top margin (in px).\n\n        Returns\n        -------\n        Margin\n        '
        super(Margin, self).__init__('margin')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Margin\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Margin`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('autoexpand', None)
        _v = autoexpand if autoexpand is not None else _v
        if _v is not None:
            self['autoexpand'] = _v
        _v = arg.pop('b', None)
        _v = b if b is not None else _v
        if _v is not None:
            self['b'] = _v
        _v = arg.pop('l', None)
        _v = l if l is not None else _v
        if _v is not None:
            self['l'] = _v
        _v = arg.pop('pad', None)
        _v = pad if pad is not None else _v
        if _v is not None:
            self['pad'] = _v
        _v = arg.pop('r', None)
        _v = r if r is not None else _v
        if _v is not None:
            self['r'] = _v
        _v = arg.pop('t', None)
        _v = t if t is not None else _v
        if _v is not None:
            self['t'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
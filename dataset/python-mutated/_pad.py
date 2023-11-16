from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Pad(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.title'
    _path_str = 'layout.title.pad'
    _valid_props = {'b', 'l', 'r', 't'}

    @property
    def b(self):
        if False:
            print('Hello World!')
        "\n        The amount of padding (in px) along the bottom of the\n        component.\n\n        The 'b' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['b']

    @b.setter
    def b(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['b'] = val

    @property
    def l(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The amount of padding (in px) on the left side of the\n        component.\n\n        The 'l' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['l']

    @l.setter
    def l(self, val):
        if False:
            return 10
        self['l'] = val

    @property
    def r(self):
        if False:
            while True:
                i = 10
        "\n        The amount of padding (in px) on the right side of the\n        component.\n\n        The 'r' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['r']

    @r.setter
    def r(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['r'] = val

    @property
    def t(self):
        if False:
            return 10
        "\n        The amount of padding (in px) along the top of the component.\n\n        The 't' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['t']

    @t.setter
    def t(self, val):
        if False:
            while True:
                i = 10
        self['t'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        b\n            The amount of padding (in px) along the bottom of the\n            component.\n        l\n            The amount of padding (in px) on the left side of the\n            component.\n        r\n            The amount of padding (in px) on the right side of the\n            component.\n        t\n            The amount of padding (in px) along the top of the\n            component.\n        '

    def __init__(self, arg=None, b=None, l=None, r=None, t=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Pad object\n\n        Sets the padding of the title. Each padding value only applies\n        when the corresponding `xanchor`/`yanchor` value is set\n        accordingly. E.g. for left padding to take effect, `xanchor`\n        must be set to "left". The same rule applies if\n        `xanchor`/`yanchor` is determined automatically. Padding is\n        muted if the respective anchor value is "middle*/*center".\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.title.Pad`\n        b\n            The amount of padding (in px) along the bottom of the\n            component.\n        l\n            The amount of padding (in px) on the left side of the\n            component.\n        r\n            The amount of padding (in px) on the right side of the\n            component.\n        t\n            The amount of padding (in px) along the top of the\n            component.\n\n        Returns\n        -------\n        Pad\n        '
        super(Pad, self).__init__('pad')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.title.Pad\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.title.Pad`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('b', None)
        _v = b if b is not None else _v
        if _v is not None:
            self['b'] = _v
        _v = arg.pop('l', None)
        _v = l if l is not None else _v
        if _v is not None:
            self['l'] = _v
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
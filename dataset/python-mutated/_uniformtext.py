from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Uniformtext(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.uniformtext'
    _valid_props = {'minsize', 'mode'}

    @property
    def minsize(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the minimum text size between traces of the same type.\n\n        The 'minsize' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['minsize']

    @minsize.setter
    def minsize(self, val):
        if False:
            while True:
                i = 10
        self['minsize'] = val

    @property
    def mode(self):
        if False:
            i = 10
            return i + 15
        '\n        Determines how the font size for various text elements are\n        uniformed between each trace type. If the computed text sizes\n        were smaller than the minimum size defined by\n        `uniformtext.minsize` using "hide" option hides the text; and\n        using "show" option shows the text without further downscaling.\n        Please note that if the size defined by `minsize` is greater\n        than the font size defined by trace, then the `minsize` is\n        used.\n\n        The \'mode\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [False, \'hide\', \'show\']\n\n        Returns\n        -------\n        Any\n        '
        return self['mode']

    @mode.setter
    def mode(self, val):
        if False:
            print('Hello World!')
        self['mode'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return '        minsize\n            Sets the minimum text size between traces of the same\n            type.\n        mode\n            Determines how the font size for various text elements\n            are uniformed between each trace type. If the computed\n            text sizes were smaller than the minimum size defined\n            by `uniformtext.minsize` using "hide" option hides the\n            text; and using "show" option shows the text without\n            further downscaling. Please note that if the size\n            defined by `minsize` is greater than the font size\n            defined by trace, then the `minsize` is used.\n        '

    def __init__(self, arg=None, minsize=None, mode=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Uniformtext object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Uniformtext`\n        minsize\n            Sets the minimum text size between traces of the same\n            type.\n        mode\n            Determines how the font size for various text elements\n            are uniformed between each trace type. If the computed\n            text sizes were smaller than the minimum size defined\n            by `uniformtext.minsize` using "hide" option hides the\n            text; and using "show" option shows the text without\n            further downscaling. Please note that if the size\n            defined by `minsize` is greater than the font size\n            defined by trace, then the `minsize` is used.\n\n        Returns\n        -------\n        Uniformtext\n        '
        super(Uniformtext, self).__init__('uniformtext')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Uniformtext\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Uniformtext`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('minsize', None)
        _v = minsize if minsize is not None else _v
        if _v is not None:
            self['minsize'] = _v
        _v = arg.pop('mode', None)
        _v = mode if mode is not None else _v
        if _v is not None:
            self['mode'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
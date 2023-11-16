from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Autorangeoptions(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.xaxis'
    _path_str = 'layout.xaxis.autorangeoptions'
    _valid_props = {'clipmax', 'clipmin', 'include', 'includesrc', 'maxallowed', 'minallowed'}

    @property
    def clipmax(self):
        if False:
            while True:
                i = 10
        "\n        Clip autorange maximum if it goes beyond this value. Has no\n        effect when `autorangeoptions.maxallowed` is provided.\n\n        The 'clipmax' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['clipmax']

    @clipmax.setter
    def clipmax(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['clipmax'] = val

    @property
    def clipmin(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Clip autorange minimum if it goes beyond this value. Has no\n        effect when `autorangeoptions.minallowed` is provided.\n\n        The 'clipmin' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['clipmin']

    @clipmin.setter
    def clipmin(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['clipmin'] = val

    @property
    def include(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Ensure this value is included in autorange.\n\n        The 'include' property accepts values of any type\n\n        Returns\n        -------\n        Any|numpy.ndarray\n        "
        return self['include']

    @include.setter
    def include(self, val):
        if False:
            while True:
                i = 10
        self['include'] = val

    @property
    def includesrc(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the source reference on Chart Studio Cloud for `include`.\n\n        The 'includesrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['includesrc']

    @includesrc.setter
    def includesrc(self, val):
        if False:
            return 10
        self['includesrc'] = val

    @property
    def maxallowed(self):
        if False:
            return 10
        "\n        Use this value exactly as autorange maximum.\n\n        The 'maxallowed' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['maxallowed']

    @maxallowed.setter
    def maxallowed(self, val):
        if False:
            while True:
                i = 10
        self['maxallowed'] = val

    @property
    def minallowed(self):
        if False:
            i = 10
            return i + 15
        "\n        Use this value exactly as autorange minimum.\n\n        The 'minallowed' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['minallowed']

    @minallowed.setter
    def minallowed(self, val):
        if False:
            print('Hello World!')
        self['minallowed'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        clipmax\n            Clip autorange maximum if it goes beyond this value.\n            Has no effect when `autorangeoptions.maxallowed` is\n            provided.\n        clipmin\n            Clip autorange minimum if it goes beyond this value.\n            Has no effect when `autorangeoptions.minallowed` is\n            provided.\n        include\n            Ensure this value is included in autorange.\n        includesrc\n            Sets the source reference on Chart Studio Cloud for\n            `include`.\n        maxallowed\n            Use this value exactly as autorange maximum.\n        minallowed\n            Use this value exactly as autorange minimum.\n        '

    def __init__(self, arg=None, clipmax=None, clipmin=None, include=None, includesrc=None, maxallowed=None, minallowed=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Autorangeoptions object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.layout.xaxis.A\n            utorangeoptions`\n        clipmax\n            Clip autorange maximum if it goes beyond this value.\n            Has no effect when `autorangeoptions.maxallowed` is\n            provided.\n        clipmin\n            Clip autorange minimum if it goes beyond this value.\n            Has no effect when `autorangeoptions.minallowed` is\n            provided.\n        include\n            Ensure this value is included in autorange.\n        includesrc\n            Sets the source reference on Chart Studio Cloud for\n            `include`.\n        maxallowed\n            Use this value exactly as autorange maximum.\n        minallowed\n            Use this value exactly as autorange minimum.\n\n        Returns\n        -------\n        Autorangeoptions\n        '
        super(Autorangeoptions, self).__init__('autorangeoptions')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.xaxis.Autorangeoptions\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.xaxis.Autorangeoptions`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('clipmax', None)
        _v = clipmax if clipmax is not None else _v
        if _v is not None:
            self['clipmax'] = _v
        _v = arg.pop('clipmin', None)
        _v = clipmin if clipmin is not None else _v
        if _v is not None:
            self['clipmin'] = _v
        _v = arg.pop('include', None)
        _v = include if include is not None else _v
        if _v is not None:
            self['include'] = _v
        _v = arg.pop('includesrc', None)
        _v = includesrc if includesrc is not None else _v
        if _v is not None:
            self['includesrc'] = _v
        _v = arg.pop('maxallowed', None)
        _v = maxallowed if maxallowed is not None else _v
        if _v is not None:
            self['maxallowed'] = _v
        _v = arg.pop('minallowed', None)
        _v = minallowed if minallowed is not None else _v
        if _v is not None:
            self['minallowed'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
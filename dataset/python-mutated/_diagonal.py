from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Diagonal(_BaseTraceHierarchyType):
    _parent_path_str = 'splom'
    _path_str = 'splom.diagonal'
    _valid_props = {'visible'}

    @property
    def visible(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not subplots on the diagonal are\n        displayed.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            print('Hello World!')
        self['visible'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        visible\n            Determines whether or not subplots on the diagonal are\n            displayed.\n        '

    def __init__(self, arg=None, visible=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Diagonal object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.splom.Diagonal`\n        visible\n            Determines whether or not subplots on the diagonal are\n            displayed.\n\n        Returns\n        -------\n        Diagonal\n        '
        super(Diagonal, self).__init__('diagonal')
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
            raise ValueError('The first argument to the plotly.graph_objs.splom.Diagonal\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.splom.Diagonal`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
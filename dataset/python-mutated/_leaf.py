from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Leaf(_BaseTraceHierarchyType):
    _parent_path_str = 'sunburst'
    _path_str = 'sunburst.leaf'
    _valid_props = {'opacity'}

    @property
    def opacity(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the opacity of the leaves. With colorscale it is defaulted\n        to 1; otherwise it is defaulted to 0.7\n\n        The 'opacity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            i = 10
            return i + 15
        self['opacity'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        opacity\n            Sets the opacity of the leaves. With colorscale it is\n            defaulted to 1; otherwise it is defaulted to 0.7\n        '

    def __init__(self, arg=None, opacity=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new Leaf object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.sunburst.Leaf`\n        opacity\n            Sets the opacity of the leaves. With colorscale it is\n            defaulted to 1; otherwise it is defaulted to 0.7\n\n        Returns\n        -------\n        Leaf\n        '
        super(Leaf, self).__init__('leaf')
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
            raise ValueError('The first argument to the plotly.graph_objs.sunburst.Leaf\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.sunburst.Leaf`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
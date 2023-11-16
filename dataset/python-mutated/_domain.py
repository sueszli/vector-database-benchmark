from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Domain(_BaseTraceHierarchyType):
    _parent_path_str = 'funnelarea'
    _path_str = 'funnelarea.domain'
    _valid_props = {'column', 'row', 'x', 'y'}

    @property
    def column(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If there is a layout grid, use the domain for this column in\n        the grid for this funnelarea trace .\n\n        The 'column' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['column']

    @column.setter
    def column(self, val):
        if False:
            print('Hello World!')
        self['column'] = val

    @property
    def row(self):
        if False:
            print('Hello World!')
        "\n        If there is a layout grid, use the domain for this row in the\n        grid for this funnelarea trace .\n\n        The 'row' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['row']

    @row.setter
    def row(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['row'] = val

    @property
    def x(self):
        if False:
            for i in range(10):
                print('nop')
        "\n            Sets the horizontal domain of this funnelarea trace (in plot\n            fraction).\n\n            The 'x' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The 'x[0]' property is a number and may be specified as:\n              - An int or float in the interval [0, 1]\n        (1) The 'x[1]' property is a number and may be specified as:\n              - An int or float in the interval [0, 1]\n\n            Returns\n            -------\n            list\n        "
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
        "\n            Sets the vertical domain of this funnelarea trace (in plot\n            fraction).\n\n            The 'y' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The 'y[0]' property is a number and may be specified as:\n              - An int or float in the interval [0, 1]\n        (1) The 'y[1]' property is a number and may be specified as:\n              - An int or float in the interval [0, 1]\n\n            Returns\n            -------\n            list\n        "
        return self['y']

    @y.setter
    def y(self, val):
        if False:
            print('Hello World!')
        self['y'] = val

    @property
    def _prop_descriptions(self):
        if False:
            i = 10
            return i + 15
        return '        column\n            If there is a layout grid, use the domain for this\n            column in the grid for this funnelarea trace .\n        row\n            If there is a layout grid, use the domain for this row\n            in the grid for this funnelarea trace .\n        x\n            Sets the horizontal domain of this funnelarea trace (in\n            plot fraction).\n        y\n            Sets the vertical domain of this funnelarea trace (in\n            plot fraction).\n        '

    def __init__(self, arg=None, column=None, row=None, x=None, y=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Domain object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.funnelarea.Domain`\n        column\n            If there is a layout grid, use the domain for this\n            column in the grid for this funnelarea trace .\n        row\n            If there is a layout grid, use the domain for this row\n            in the grid for this funnelarea trace .\n        x\n            Sets the horizontal domain of this funnelarea trace (in\n            plot fraction).\n        y\n            Sets the vertical domain of this funnelarea trace (in\n            plot fraction).\n\n        Returns\n        -------\n        Domain\n        '
        super(Domain, self).__init__('domain')
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
            raise ValueError('The first argument to the plotly.graph_objs.funnelarea.Domain\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.funnelarea.Domain`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('column', None)
        _v = column if column is not None else _v
        if _v is not None:
            self['column'] = _v
        _v = arg.pop('row', None)
        _v = row if row is not None else _v
        if _v is not None:
            self['row'] = _v
        _v = arg.pop('x', None)
        _v = x if x is not None else _v
        if _v is not None:
            self['x'] = _v
        _v = arg.pop('y', None)
        _v = y if y is not None else _v
        if _v is not None:
            self['y'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
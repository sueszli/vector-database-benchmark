from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Tiling(_BaseTraceHierarchyType):
    _parent_path_str = 'treemap'
    _path_str = 'treemap.tiling'
    _valid_props = {'flip', 'packing', 'pad', 'squarifyratio'}

    @property
    def flip(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines if the positions obtained from solver are flipped on\n        each axis.\n\n        The 'flip' property is a flaglist and may be specified\n        as a string containing:\n          - Any combination of ['x', 'y'] joined with '+' characters\n            (e.g. 'x+y')\n\n        Returns\n        -------\n        Any\n        "
        return self['flip']

    @flip.setter
    def flip(self, val):
        if False:
            return 10
        self['flip'] = val

    @property
    def packing(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Determines d3 treemap solver. For more info please refer to\n        https://github.com/d3/d3-hierarchy#treemap-tiling\n\n        The 'packing' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['squarify', 'binary', 'dice', 'slice', 'slice-dice',\n                'dice-slice']\n\n        Returns\n        -------\n        Any\n        "
        return self['packing']

    @packing.setter
    def packing(self, val):
        if False:
            while True:
                i = 10
        self['packing'] = val

    @property
    def pad(self):
        if False:
            return 10
        "\n        Sets the inner padding (in px).\n\n        The 'pad' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['pad']

    @pad.setter
    def pad(self, val):
        if False:
            return 10
        self['pad'] = val

    @property
    def squarifyratio(self):
        if False:
            i = 10
            return i + 15
        '\n        When using "squarify" `packing` algorithm, according to https:/\n        /github.com/d3/d3-\n        hierarchy/blob/v3.1.1/README.md#squarify_ratio this option\n        specifies the desired aspect ratio of the generated rectangles.\n        The ratio must be specified as a number greater than or equal\n        to one. Note that the orientation of the generated rectangles\n        (tall or wide) is not implied by the ratio; for example, a\n        ratio of two will attempt to produce a mixture of rectangles\n        whose width:height ratio is either 2:1 or 1:2. When using\n        "squarify", unlike d3 which uses the Golden Ratio i.e.\n        1.618034, Plotly applies 1 to increase squares in treemap\n        layouts.\n\n        The \'squarifyratio\' property is a number and may be specified as:\n          - An int or float in the interval [1, inf]\n\n        Returns\n        -------\n        int|float\n        '
        return self['squarifyratio']

    @squarifyratio.setter
    def squarifyratio(self, val):
        if False:
            return 10
        self['squarifyratio'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        flip\n            Determines if the positions obtained from solver are\n            flipped on each axis.\n        packing\n            Determines d3 treemap solver. For more info please\n            refer to https://github.com/d3/d3-hierarchy#treemap-\n            tiling\n        pad\n            Sets the inner padding (in px).\n        squarifyratio\n            When using "squarify" `packing` algorithm, according to\n            https://github.com/d3/d3-\n            hierarchy/blob/v3.1.1/README.md#squarify_ratio this\n            option specifies the desired aspect ratio of the\n            generated rectangles. The ratio must be specified as a\n            number greater than or equal to one. Note that the\n            orientation of the generated rectangles (tall or wide)\n            is not implied by the ratio; for example, a ratio of\n            two will attempt to produce a mixture of rectangles\n            whose width:height ratio is either 2:1 or 1:2. When\n            using "squarify", unlike d3 which uses the Golden Ratio\n            i.e. 1.618034, Plotly applies 1 to increase squares in\n            treemap layouts.\n        '

    def __init__(self, arg=None, flip=None, packing=None, pad=None, squarifyratio=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Tiling object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.treemap.Tiling`\n        flip\n            Determines if the positions obtained from solver are\n            flipped on each axis.\n        packing\n            Determines d3 treemap solver. For more info please\n            refer to https://github.com/d3/d3-hierarchy#treemap-\n            tiling\n        pad\n            Sets the inner padding (in px).\n        squarifyratio\n            When using "squarify" `packing` algorithm, according to\n            https://github.com/d3/d3-\n            hierarchy/blob/v3.1.1/README.md#squarify_ratio this\n            option specifies the desired aspect ratio of the\n            generated rectangles. The ratio must be specified as a\n            number greater than or equal to one. Note that the\n            orientation of the generated rectangles (tall or wide)\n            is not implied by the ratio; for example, a ratio of\n            two will attempt to produce a mixture of rectangles\n            whose width:height ratio is either 2:1 or 1:2. When\n            using "squarify", unlike d3 which uses the Golden Ratio\n            i.e. 1.618034, Plotly applies 1 to increase squares in\n            treemap layouts.\n\n        Returns\n        -------\n        Tiling\n        '
        super(Tiling, self).__init__('tiling')
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
            raise ValueError('The first argument to the plotly.graph_objs.treemap.Tiling\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.treemap.Tiling`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('flip', None)
        _v = flip if flip is not None else _v
        if _v is not None:
            self['flip'] = _v
        _v = arg.pop('packing', None)
        _v = packing if packing is not None else _v
        if _v is not None:
            self['packing'] = _v
        _v = arg.pop('pad', None)
        _v = pad if pad is not None else _v
        if _v is not None:
            self['pad'] = _v
        _v = arg.pop('squarifyratio', None)
        _v = squarifyratio if squarifyratio is not None else _v
        if _v is not None:
            self['squarifyratio'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class ErrorX(_BaseTraceHierarchyType):
    _parent_path_str = 'bar'
    _path_str = 'bar.error_x'
    _valid_props = {'array', 'arrayminus', 'arrayminussrc', 'arraysrc', 'color', 'copy_ystyle', 'symmetric', 'thickness', 'traceref', 'tracerefminus', 'type', 'value', 'valueminus', 'visible', 'width'}

    @property
    def array(self):
        if False:
            print('Hello World!')
        "\n        Sets the data corresponding the length of each error bar.\n        Values are plotted relative to the underlying data.\n\n        The 'array' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['array']

    @array.setter
    def array(self, val):
        if False:
            print('Hello World!')
        self['array'] = val

    @property
    def arrayminus(self):
        if False:
            print('Hello World!')
        "\n        Sets the data corresponding the length of each error bar in the\n        bottom (left) direction for vertical (horizontal) bars Values\n        are plotted relative to the underlying data.\n\n        The 'arrayminus' property is an array that may be specified as a tuple,\n        list, numpy array, or pandas Series\n\n        Returns\n        -------\n        numpy.ndarray\n        "
        return self['arrayminus']

    @arrayminus.setter
    def arrayminus(self, val):
        if False:
            while True:
                i = 10
        self['arrayminus'] = val

    @property
    def arrayminussrc(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the source reference on Chart Studio Cloud for\n        `arrayminus`.\n\n        The 'arrayminussrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['arrayminussrc']

    @arrayminussrc.setter
    def arrayminussrc(self, val):
        if False:
            while True:
                i = 10
        self['arrayminussrc'] = val

    @property
    def arraysrc(self):
        if False:
            while True:
                i = 10
        "\n        Sets the source reference on Chart Studio Cloud for `array`.\n\n        The 'arraysrc' property must be specified as a string or\n        as a plotly.grid_objs.Column object\n\n        Returns\n        -------\n        str\n        "
        return self['arraysrc']

    @arraysrc.setter
    def arraysrc(self, val):
        if False:
            while True:
                i = 10
        self['arraysrc'] = val

    @property
    def color(self):
        if False:
            while True:
                i = 10
        "\n        Sets the stoke color of the error bars.\n\n        The 'color' property is a color and may be specified as:\n          - A hex string (e.g. '#ff0000')\n          - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n          - A named CSS color:\n                aliceblue, antiquewhite, aqua, aquamarine, azure,\n                beige, bisque, black, blanchedalmond, blue,\n                blueviolet, brown, burlywood, cadetblue,\n                chartreuse, chocolate, coral, cornflowerblue,\n                cornsilk, crimson, cyan, darkblue, darkcyan,\n                darkgoldenrod, darkgray, darkgrey, darkgreen,\n                darkkhaki, darkmagenta, darkolivegreen, darkorange,\n                darkorchid, darkred, darksalmon, darkseagreen,\n                darkslateblue, darkslategray, darkslategrey,\n                darkturquoise, darkviolet, deeppink, deepskyblue,\n                dimgray, dimgrey, dodgerblue, firebrick,\n                floralwhite, forestgreen, fuchsia, gainsboro,\n                ghostwhite, gold, goldenrod, gray, grey, green,\n                greenyellow, honeydew, hotpink, indianred, indigo,\n                ivory, khaki, lavender, lavenderblush, lawngreen,\n                lemonchiffon, lightblue, lightcoral, lightcyan,\n                lightgoldenrodyellow, lightgray, lightgrey,\n                lightgreen, lightpink, lightsalmon, lightseagreen,\n                lightskyblue, lightslategray, lightslategrey,\n                lightsteelblue, lightyellow, lime, limegreen,\n                linen, magenta, maroon, mediumaquamarine,\n                mediumblue, mediumorchid, mediumpurple,\n                mediumseagreen, mediumslateblue, mediumspringgreen,\n                mediumturquoise, mediumvioletred, midnightblue,\n                mintcream, mistyrose, moccasin, navajowhite, navy,\n                oldlace, olive, olivedrab, orange, orangered,\n                orchid, palegoldenrod, palegreen, paleturquoise,\n                palevioletred, papayawhip, peachpuff, peru, pink,\n                plum, powderblue, purple, red, rosybrown,\n                royalblue, rebeccapurple, saddlebrown, salmon,\n                sandybrown, seagreen, seashell, sienna, silver,\n                skyblue, slateblue, slategray, slategrey, snow,\n                springgreen, steelblue, tan, teal, thistle, tomato,\n                turquoise, violet, wheat, white, whitesmoke,\n                yellow, yellowgreen\n\n        Returns\n        -------\n        str\n        "
        return self['color']

    @color.setter
    def color(self, val):
        if False:
            print('Hello World!')
        self['color'] = val

    @property
    def copy_ystyle(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'copy_ystyle' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['copy_ystyle']

    @copy_ystyle.setter
    def copy_ystyle(self, val):
        if False:
            print('Hello World!')
        self['copy_ystyle'] = val

    @property
    def symmetric(self):
        if False:
            i = 10
            return i + 15
        "\n        Determines whether or not the error bars have the same length\n        in both direction (top/bottom for vertical bars, left/right for\n        horizontal bars.\n\n        The 'symmetric' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['symmetric']

    @symmetric.setter
    def symmetric(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['symmetric'] = val

    @property
    def thickness(self):
        if False:
            return 10
        "\n        Sets the thickness (in px) of the error bars.\n\n        The 'thickness' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['thickness']

    @thickness.setter
    def thickness(self, val):
        if False:
            while True:
                i = 10
        self['thickness'] = val

    @property
    def traceref(self):
        if False:
            return 10
        "\n        The 'traceref' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['traceref']

    @traceref.setter
    def traceref(self, val):
        if False:
            return 10
        self['traceref'] = val

    @property
    def tracerefminus(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'tracerefminus' property is a integer and may be specified as:\n          - An int (or float that will be cast to an int)\n            in the interval [0, 9223372036854775807]\n\n        Returns\n        -------\n        int\n        "
        return self['tracerefminus']

    @tracerefminus.setter
    def tracerefminus(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['tracerefminus'] = val

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        '\n        Determines the rule used to generate the error bars. If\n        *constant`, the bar lengths are of a constant value. Set this\n        constant in `value`. If "percent", the bar lengths correspond\n        to a percentage of underlying data. Set this percentage in\n        `value`. If "sqrt", the bar lengths correspond to the square of\n        the underlying data. If "data", the bar lengths are set with\n        data set `array`.\n\n        The \'type\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'percent\', \'constant\', \'sqrt\', \'data\']\n\n        Returns\n        -------\n        Any\n        '
        return self['type']

    @type.setter
    def type(self, val):
        if False:
            return 10
        self['type'] = val

    @property
    def value(self):
        if False:
            while True:
                i = 10
        '\n        Sets the value of either the percentage (if `type` is set to\n        "percent") or the constant (if `type` is set to "constant")\n        corresponding to the lengths of the error bars.\n\n        The \'value\' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        '
        return self['value']

    @value.setter
    def value(self, val):
        if False:
            i = 10
            return i + 15
        self['value'] = val

    @property
    def valueminus(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the value of either the percentage (if `type` is set to\n        "percent") or the constant (if `type` is set to "constant")\n        corresponding to the lengths of the error bars in the bottom\n        (left) direction for vertical (horizontal) bars\n\n        The \'valueminus\' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        '
        return self['valueminus']

    @valueminus.setter
    def valueminus(self, val):
        if False:
            return 10
        self['valueminus'] = val

    @property
    def visible(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not this set of error bars is visible.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            print('Hello World!')
        self['visible'] = val

    @property
    def width(self):
        if False:
            print('Hello World!')
        "\n        Sets the width (in px) of the cross-bar at both ends of the\n        error bars.\n\n        The 'width' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['width']

    @width.setter
    def width(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['width'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        array\n            Sets the data corresponding the length of each error\n            bar. Values are plotted relative to the underlying\n            data.\n        arrayminus\n            Sets the data corresponding the length of each error\n            bar in the bottom (left) direction for vertical\n            (horizontal) bars Values are plotted relative to the\n            underlying data.\n        arrayminussrc\n            Sets the source reference on Chart Studio Cloud for\n            `arrayminus`.\n        arraysrc\n            Sets the source reference on Chart Studio Cloud for\n            `array`.\n        color\n            Sets the stoke color of the error bars.\n        copy_ystyle\n\n        symmetric\n            Determines whether or not the error bars have the same\n            length in both direction (top/bottom for vertical bars,\n            left/right for horizontal bars.\n        thickness\n            Sets the thickness (in px) of the error bars.\n        traceref\n\n        tracerefminus\n\n        type\n            Determines the rule used to generate the error bars. If\n            *constant`, the bar lengths are of a constant value.\n            Set this constant in `value`. If "percent", the bar\n            lengths correspond to a percentage of underlying data.\n            Set this percentage in `value`. If "sqrt", the bar\n            lengths correspond to the square of the underlying\n            data. If "data", the bar lengths are set with data set\n            `array`.\n        value\n            Sets the value of either the percentage (if `type` is\n            set to "percent") or the constant (if `type` is set to\n            "constant") corresponding to the lengths of the error\n            bars.\n        valueminus\n            Sets the value of either the percentage (if `type` is\n            set to "percent") or the constant (if `type` is set to\n            "constant") corresponding to the lengths of the error\n            bars in the bottom (left) direction for vertical\n            (horizontal) bars\n        visible\n            Determines whether or not this set of error bars is\n            visible.\n        width\n            Sets the width (in px) of the cross-bar at both ends of\n            the error bars.\n        '

    def __init__(self, arg=None, array=None, arrayminus=None, arrayminussrc=None, arraysrc=None, color=None, copy_ystyle=None, symmetric=None, thickness=None, traceref=None, tracerefminus=None, type=None, value=None, valueminus=None, visible=None, width=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new ErrorX object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.bar.ErrorX`\n        array\n            Sets the data corresponding the length of each error\n            bar. Values are plotted relative to the underlying\n            data.\n        arrayminus\n            Sets the data corresponding the length of each error\n            bar in the bottom (left) direction for vertical\n            (horizontal) bars Values are plotted relative to the\n            underlying data.\n        arrayminussrc\n            Sets the source reference on Chart Studio Cloud for\n            `arrayminus`.\n        arraysrc\n            Sets the source reference on Chart Studio Cloud for\n            `array`.\n        color\n            Sets the stoke color of the error bars.\n        copy_ystyle\n\n        symmetric\n            Determines whether or not the error bars have the same\n            length in both direction (top/bottom for vertical bars,\n            left/right for horizontal bars.\n        thickness\n            Sets the thickness (in px) of the error bars.\n        traceref\n\n        tracerefminus\n\n        type\n            Determines the rule used to generate the error bars. If\n            *constant`, the bar lengths are of a constant value.\n            Set this constant in `value`. If "percent", the bar\n            lengths correspond to a percentage of underlying data.\n            Set this percentage in `value`. If "sqrt", the bar\n            lengths correspond to the square of the underlying\n            data. If "data", the bar lengths are set with data set\n            `array`.\n        value\n            Sets the value of either the percentage (if `type` is\n            set to "percent") or the constant (if `type` is set to\n            "constant") corresponding to the lengths of the error\n            bars.\n        valueminus\n            Sets the value of either the percentage (if `type` is\n            set to "percent") or the constant (if `type` is set to\n            "constant") corresponding to the lengths of the error\n            bars in the bottom (left) direction for vertical\n            (horizontal) bars\n        visible\n            Determines whether or not this set of error bars is\n            visible.\n        width\n            Sets the width (in px) of the cross-bar at both ends of\n            the error bars.\n\n        Returns\n        -------\n        ErrorX\n        '
        super(ErrorX, self).__init__('error_x')
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
            raise ValueError('The first argument to the plotly.graph_objs.bar.ErrorX\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.bar.ErrorX`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('array', None)
        _v = array if array is not None else _v
        if _v is not None:
            self['array'] = _v
        _v = arg.pop('arrayminus', None)
        _v = arrayminus if arrayminus is not None else _v
        if _v is not None:
            self['arrayminus'] = _v
        _v = arg.pop('arrayminussrc', None)
        _v = arrayminussrc if arrayminussrc is not None else _v
        if _v is not None:
            self['arrayminussrc'] = _v
        _v = arg.pop('arraysrc', None)
        _v = arraysrc if arraysrc is not None else _v
        if _v is not None:
            self['arraysrc'] = _v
        _v = arg.pop('color', None)
        _v = color if color is not None else _v
        if _v is not None:
            self['color'] = _v
        _v = arg.pop('copy_ystyle', None)
        _v = copy_ystyle if copy_ystyle is not None else _v
        if _v is not None:
            self['copy_ystyle'] = _v
        _v = arg.pop('symmetric', None)
        _v = symmetric if symmetric is not None else _v
        if _v is not None:
            self['symmetric'] = _v
        _v = arg.pop('thickness', None)
        _v = thickness if thickness is not None else _v
        if _v is not None:
            self['thickness'] = _v
        _v = arg.pop('traceref', None)
        _v = traceref if traceref is not None else _v
        if _v is not None:
            self['traceref'] = _v
        _v = arg.pop('tracerefminus', None)
        _v = tracerefminus if tracerefminus is not None else _v
        if _v is not None:
            self['tracerefminus'] = _v
        _v = arg.pop('type', None)
        _v = type if type is not None else _v
        if _v is not None:
            self['type'] = _v
        _v = arg.pop('value', None)
        _v = value if value is not None else _v
        if _v is not None:
            self['value'] = _v
        _v = arg.pop('valueminus', None)
        _v = valueminus if valueminus is not None else _v
        if _v is not None:
            self['valueminus'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        _v = arg.pop('width', None)
        _v = width if width is not None else _v
        if _v is not None:
            self['width'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
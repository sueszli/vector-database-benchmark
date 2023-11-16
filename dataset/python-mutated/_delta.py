from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Delta(_BaseTraceHierarchyType):
    _parent_path_str = 'indicator'
    _path_str = 'indicator.delta'
    _valid_props = {'decreasing', 'font', 'increasing', 'position', 'prefix', 'reference', 'relative', 'suffix', 'valueformat'}

    @property
    def decreasing(self):
        if False:
            while True:
                i = 10
        "\n        The 'decreasing' property is an instance of Decreasing\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.delta.Decreasing`\n          - A dict of string/value properties that will be passed\n            to the Decreasing constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the color for increasing value.\n                symbol\n                    Sets the symbol to display for increasing value\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.delta.Decreasing\n        "
        return self['decreasing']

    @decreasing.setter
    def decreasing(self, val):
        if False:
            print('Hello World!')
        self['decreasing'] = val

    @property
    def font(self):
        if False:
            while True:
                i = 10
        '\n        Set the font used to display the delta\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.delta.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.delta.Font\n        '
        return self['font']

    @font.setter
    def font(self, val):
        if False:
            i = 10
            return i + 15
        self['font'] = val

    @property
    def increasing(self):
        if False:
            while True:
                i = 10
        "\n        The 'increasing' property is an instance of Increasing\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.delta.Increasing`\n          - A dict of string/value properties that will be passed\n            to the Increasing constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the color for increasing value.\n                symbol\n                    Sets the symbol to display for increasing value\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.delta.Increasing\n        "
        return self['increasing']

    @increasing.setter
    def increasing(self, val):
        if False:
            i = 10
            return i + 15
        self['increasing'] = val

    @property
    def position(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the position of delta with respect to the number.\n\n        The 'position' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['top', 'bottom', 'left', 'right']\n\n        Returns\n        -------\n        Any\n        "
        return self['position']

    @position.setter
    def position(self, val):
        if False:
            return 10
        self['position'] = val

    @property
    def prefix(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets a prefix appearing before the delta.\n\n        The 'prefix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['prefix']

    @prefix.setter
    def prefix(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['prefix'] = val

    @property
    def reference(self):
        if False:
            while True:
                i = 10
        "\n        Sets the reference value to compute the delta. By default, it\n        is set to the current value.\n\n        The 'reference' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['reference']

    @reference.setter
    def reference(self, val):
        if False:
            print('Hello World!')
        self['reference'] = val

    @property
    def relative(self):
        if False:
            return 10
        "\n        Show relative change\n\n        The 'relative' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['relative']

    @relative.setter
    def relative(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['relative'] = val

    @property
    def suffix(self):
        if False:
            return 10
        "\n        Sets a suffix appearing next to the delta.\n\n        The 'suffix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['suffix']

    @suffix.setter
    def suffix(self, val):
        if False:
            return 10
        self['suffix'] = val

    @property
    def valueformat(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the value formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n\n        The 'valueformat' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['valueformat']

    @valueformat.setter
    def valueformat(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['valueformat'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        decreasing\n            :class:`plotly.graph_objects.indicator.delta.Decreasing\n            ` instance or dict with compatible properties\n        font\n            Set the font used to display the delta\n        increasing\n            :class:`plotly.graph_objects.indicator.delta.Increasing\n            ` instance or dict with compatible properties\n        position\n            Sets the position of delta with respect to the number.\n        prefix\n            Sets a prefix appearing before the delta.\n        reference\n            Sets the reference value to compute the delta. By\n            default, it is set to the current value.\n        relative\n            Show relative change\n        suffix\n            Sets a suffix appearing next to the delta.\n        valueformat\n            Sets the value formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n        '

    def __init__(self, arg=None, decreasing=None, font=None, increasing=None, position=None, prefix=None, reference=None, relative=None, suffix=None, valueformat=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Delta object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.indicator.Delta`\n        decreasing\n            :class:`plotly.graph_objects.indicator.delta.Decreasing\n            ` instance or dict with compatible properties\n        font\n            Set the font used to display the delta\n        increasing\n            :class:`plotly.graph_objects.indicator.delta.Increasing\n            ` instance or dict with compatible properties\n        position\n            Sets the position of delta with respect to the number.\n        prefix\n            Sets a prefix appearing before the delta.\n        reference\n            Sets the reference value to compute the delta. By\n            default, it is set to the current value.\n        relative\n            Show relative change\n        suffix\n            Sets a suffix appearing next to the delta.\n        valueformat\n            Sets the value formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n\n        Returns\n        -------\n        Delta\n        '
        super(Delta, self).__init__('delta')
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
            raise ValueError('The first argument to the plotly.graph_objs.indicator.Delta\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.indicator.Delta`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('decreasing', None)
        _v = decreasing if decreasing is not None else _v
        if _v is not None:
            self['decreasing'] = _v
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('increasing', None)
        _v = increasing if increasing is not None else _v
        if _v is not None:
            self['increasing'] = _v
        _v = arg.pop('position', None)
        _v = position if position is not None else _v
        if _v is not None:
            self['position'] = _v
        _v = arg.pop('prefix', None)
        _v = prefix if prefix is not None else _v
        if _v is not None:
            self['prefix'] = _v
        _v = arg.pop('reference', None)
        _v = reference if reference is not None else _v
        if _v is not None:
            self['reference'] = _v
        _v = arg.pop('relative', None)
        _v = relative if relative is not None else _v
        if _v is not None:
            self['relative'] = _v
        _v = arg.pop('suffix', None)
        _v = suffix if suffix is not None else _v
        if _v is not None:
            self['suffix'] = _v
        _v = arg.pop('valueformat', None)
        _v = valueformat if valueformat is not None else _v
        if _v is not None:
            self['valueformat'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
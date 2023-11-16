from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Number(_BaseTraceHierarchyType):
    _parent_path_str = 'indicator'
    _path_str = 'indicator.number'
    _valid_props = {'font', 'prefix', 'suffix', 'valueformat'}

    @property
    def font(self):
        if False:
            while True:
                i = 10
        '\n        Set the font used to display main number\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.indicator.number.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.indicator.number.Font\n        '
        return self['font']

    @font.setter
    def font(self, val):
        if False:
            i = 10
            return i + 15
        self['font'] = val

    @property
    def prefix(self):
        if False:
            print('Hello World!')
        "\n        Sets a prefix appearing before the number.\n\n        The 'prefix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['prefix']

    @prefix.setter
    def prefix(self, val):
        if False:
            i = 10
            return i + 15
        self['prefix'] = val

    @property
    def suffix(self):
        if False:
            return 10
        "\n        Sets a suffix appearing next to the number.\n\n        The 'suffix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['suffix']

    @suffix.setter
    def suffix(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['suffix'] = val

    @property
    def valueformat(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the value formatting rule using d3 formatting mini-\n        languages which are very similar to those in Python. For\n        numbers, see:\n        https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n\n        The 'valueformat' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['valueformat']

    @valueformat.setter
    def valueformat(self, val):
        if False:
            while True:
                i = 10
        self['valueformat'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        font\n            Set the font used to display main number\n        prefix\n            Sets a prefix appearing before the number.\n        suffix\n            Sets a suffix appearing next to the number.\n        valueformat\n            Sets the value formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n        '

    def __init__(self, arg=None, font=None, prefix=None, suffix=None, valueformat=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Number object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.indicator.Number`\n        font\n            Set the font used to display main number\n        prefix\n            Sets a prefix appearing before the number.\n        suffix\n            Sets a suffix appearing next to the number.\n        valueformat\n            Sets the value formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n\n        Returns\n        -------\n        Number\n        '
        super(Number, self).__init__('number')
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
            raise ValueError('The first argument to the plotly.graph_objs.indicator.Number\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.indicator.Number`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('prefix', None)
        _v = prefix if prefix is not None else _v
        if _v is not None:
            self['prefix'] = _v
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
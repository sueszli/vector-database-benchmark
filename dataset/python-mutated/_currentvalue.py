from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Currentvalue(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.slider'
    _path_str = 'layout.slider.currentvalue'
    _valid_props = {'font', 'offset', 'prefix', 'suffix', 'visible', 'xanchor'}

    @property
    def font(self):
        if False:
            while True:
                i = 10
        '\n        Sets the font of the current value label text.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.slider.currentvalue.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.layout.slider.currentvalue.Font\n        '
        return self['font']

    @font.setter
    def font(self, val):
        if False:
            while True:
                i = 10
        self['font'] = val

    @property
    def offset(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The amount of space, in pixels, between the current value label\n        and the slider.\n\n        The 'offset' property is a number and may be specified as:\n          - An int or float\n\n        Returns\n        -------\n        int|float\n        "
        return self['offset']

    @offset.setter
    def offset(self, val):
        if False:
            print('Hello World!')
        self['offset'] = val

    @property
    def prefix(self):
        if False:
            print('Hello World!')
        "\n        When currentvalue.visible is true, this sets the prefix of the\n        label.\n\n        The 'prefix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['prefix']

    @prefix.setter
    def prefix(self, val):
        if False:
            print('Hello World!')
        self['prefix'] = val

    @property
    def suffix(self):
        if False:
            while True:
                i = 10
        "\n        When currentvalue.visible is true, this sets the suffix of the\n        label.\n\n        The 'suffix' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['suffix']

    @suffix.setter
    def suffix(self, val):
        if False:
            i = 10
            return i + 15
        self['suffix'] = val

    @property
    def visible(self):
        if False:
            return 10
        "\n        Shows the currently-selected value above the slider.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            print('Hello World!')
        self['visible'] = val

    @property
    def xanchor(self):
        if False:
            while True:
                i = 10
        "\n        The alignment of the value readout relative to the length of\n        the slider.\n\n        The 'xanchor' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['left', 'center', 'right']\n\n        Returns\n        -------\n        Any\n        "
        return self['xanchor']

    @xanchor.setter
    def xanchor(self, val):
        if False:
            print('Hello World!')
        self['xanchor'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        font\n            Sets the font of the current value label text.\n        offset\n            The amount of space, in pixels, between the current\n            value label and the slider.\n        prefix\n            When currentvalue.visible is true, this sets the prefix\n            of the label.\n        suffix\n            When currentvalue.visible is true, this sets the suffix\n            of the label.\n        visible\n            Shows the currently-selected value above the slider.\n        xanchor\n            The alignment of the value readout relative to the\n            length of the slider.\n        '

    def __init__(self, arg=None, font=None, offset=None, prefix=None, suffix=None, visible=None, xanchor=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Currentvalue object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.slider.Currentvalue`\n        font\n            Sets the font of the current value label text.\n        offset\n            The amount of space, in pixels, between the current\n            value label and the slider.\n        prefix\n            When currentvalue.visible is true, this sets the prefix\n            of the label.\n        suffix\n            When currentvalue.visible is true, this sets the suffix\n            of the label.\n        visible\n            Shows the currently-selected value above the slider.\n        xanchor\n            The alignment of the value readout relative to the\n            length of the slider.\n\n        Returns\n        -------\n        Currentvalue\n        '
        super(Currentvalue, self).__init__('currentvalue')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.slider.Currentvalue\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.slider.Currentvalue`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('offset', None)
        _v = offset if offset is not None else _v
        if _v is not None:
            self['offset'] = _v
        _v = arg.pop('prefix', None)
        _v = prefix if prefix is not None else _v
        if _v is not None:
            self['prefix'] = _v
        _v = arg.pop('suffix', None)
        _v = suffix if suffix is not None else _v
        if _v is not None:
            self['suffix'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        _v = arg.pop('xanchor', None)
        _v = xanchor if xanchor is not None else _v
        if _v is not None:
            self['xanchor'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Pathbar(_BaseTraceHierarchyType):
    _parent_path_str = 'treemap'
    _path_str = 'treemap.pathbar'
    _valid_props = {'edgeshape', 'side', 'textfont', 'thickness', 'visible'}

    @property
    def edgeshape(self):
        if False:
            print('Hello World!')
        "\n        Determines which shape is used for edges between `barpath`\n        labels.\n\n        The 'edgeshape' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['>', '<', '|', '/', '\\']\n\n        Returns\n        -------\n        Any\n        "
        return self['edgeshape']

    @edgeshape.setter
    def edgeshape(self, val):
        if False:
            return 10
        self['edgeshape'] = val

    @property
    def side(self):
        if False:
            print('Hello World!')
        "\n        Determines on which side of the the treemap the `pathbar`\n        should be presented.\n\n        The 'side' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['top', 'bottom']\n\n        Returns\n        -------\n        Any\n        "
        return self['side']

    @side.setter
    def side(self, val):
        if False:
            i = 10
            return i + 15
        self['side'] = val

    @property
    def textfont(self):
        if False:
            return 10
        '\n        Sets the font used inside `pathbar`.\n\n        The \'textfont\' property is an instance of Textfont\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.treemap.pathbar.Textfont`\n          - A dict of string/value properties that will be passed\n            to the Textfont constructor\n\n            Supported dict properties:\n\n                color\n\n                colorsrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `color`.\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                familysrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `family`.\n                size\n\n                sizesrc\n                    Sets the source reference on Chart Studio Cloud\n                    for `size`.\n\n        Returns\n        -------\n        plotly.graph_objs.treemap.pathbar.Textfont\n        '
        return self['textfont']

    @textfont.setter
    def textfont(self, val):
        if False:
            while True:
                i = 10
        self['textfont'] = val

    @property
    def thickness(self):
        if False:
            return 10
        "\n        Sets the thickness of `pathbar` (in px). If not specified the\n        `pathbar.textfont.size` is used with 3 pixles extra padding on\n        each side.\n\n        The 'thickness' property is a number and may be specified as:\n          - An int or float in the interval [12, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['thickness']

    @thickness.setter
    def thickness(self, val):
        if False:
            print('Hello World!')
        self['thickness'] = val

    @property
    def visible(self):
        if False:
            while True:
                i = 10
        "\n        Determines if the path bar is drawn i.e. outside the trace\n        `domain` and with one pixel gap.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['visible'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        edgeshape\n            Determines which shape is used for edges between\n            `barpath` labels.\n        side\n            Determines on which side of the the treemap the\n            `pathbar` should be presented.\n        textfont\n            Sets the font used inside `pathbar`.\n        thickness\n            Sets the thickness of `pathbar` (in px). If not\n            specified the `pathbar.textfont.size` is used with 3\n            pixles extra padding on each side.\n        visible\n            Determines if the path bar is drawn i.e. outside the\n            trace `domain` and with one pixel gap.\n        '

    def __init__(self, arg=None, edgeshape=None, side=None, textfont=None, thickness=None, visible=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Pathbar object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.treemap.Pathbar`\n        edgeshape\n            Determines which shape is used for edges between\n            `barpath` labels.\n        side\n            Determines on which side of the the treemap the\n            `pathbar` should be presented.\n        textfont\n            Sets the font used inside `pathbar`.\n        thickness\n            Sets the thickness of `pathbar` (in px). If not\n            specified the `pathbar.textfont.size` is used with 3\n            pixles extra padding on each side.\n        visible\n            Determines if the path bar is drawn i.e. outside the\n            trace `domain` and with one pixel gap.\n\n        Returns\n        -------\n        Pathbar\n        '
        super(Pathbar, self).__init__('pathbar')
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
            raise ValueError('The first argument to the plotly.graph_objs.treemap.Pathbar\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.treemap.Pathbar`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('edgeshape', None)
        _v = edgeshape if edgeshape is not None else _v
        if _v is not None:
            self['edgeshape'] = _v
        _v = arg.pop('side', None)
        _v = side if side is not None else _v
        if _v is not None:
            self['side'] = _v
        _v = arg.pop('textfont', None)
        _v = textfont if textfont is not None else _v
        if _v is not None:
            self['textfont'] = _v
        _v = arg.pop('thickness', None)
        _v = thickness if thickness is not None else _v
        if _v is not None:
            self['thickness'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
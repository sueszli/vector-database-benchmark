from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Legendgrouptitle(_BaseTraceHierarchyType):
    _parent_path_str = 'pointcloud'
    _path_str = 'pointcloud.legendgrouptitle'
    _valid_props = {'font', 'text'}

    @property
    def font(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets this legend group\'s title font.\n\n        The \'font\' property is an instance of Font\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.pointcloud.legendgrouptitle.Font`\n          - A dict of string/value properties that will be passed\n            to the Font constructor\n\n            Supported dict properties:\n\n                color\n\n                family\n                    HTML font family - the typeface that will be\n                    applied by the web browser. The web browser\n                    will only be able to apply a font if it is\n                    available on the system which it operates.\n                    Provide multiple font families, separated by\n                    commas, to indicate the preference in which to\n                    apply fonts if they aren\'t available on the\n                    system. The Chart Studio Cloud (at\n                    https://chart-studio.plotly.com or on-premise)\n                    generates images on a server, where only a\n                    select number of fonts are installed and\n                    supported. These include "Arial", "Balto",\n                    "Courier New", "Droid Sans",, "Droid Serif",\n                    "Droid Sans Mono", "Gravitas One", "Old\n                    Standard TT", "Open Sans", "Overpass", "PT Sans\n                    Narrow", "Raleway", "Times New Roman".\n                size\n\n        Returns\n        -------\n        plotly.graph_objs.pointcloud.legendgrouptitle.Font\n        '
        return self['font']

    @font.setter
    def font(self, val):
        if False:
            print('Hello World!')
        self['font'] = val

    @property
    def text(self):
        if False:
            while True:
                i = 10
        "\n        Sets the title of the legend group.\n\n        The 'text' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['text']

    @text.setter
    def text(self, val):
        if False:
            i = 10
            return i + 15
        self['text'] = val

    @property
    def _prop_descriptions(self):
        if False:
            i = 10
            return i + 15
        return "        font\n            Sets this legend group's title font.\n        text\n            Sets the title of the legend group.\n        "

    def __init__(self, arg=None, font=None, text=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Construct a new Legendgrouptitle object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.pointcloud.Legendgrouptitle`\n        font\n            Sets this legend group's title font.\n        text\n            Sets the title of the legend group.\n\n        Returns\n        -------\n        Legendgrouptitle\n        "
        super(Legendgrouptitle, self).__init__('legendgrouptitle')
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
            raise ValueError('The first argument to the plotly.graph_objs.pointcloud.Legendgrouptitle\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.pointcloud.Legendgrouptitle`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('font', None)
        _v = font if font is not None else _v
        if _v is not None:
            self['font'] = _v
        _v = arg.pop('text', None)
        _v = text if text is not None else _v
        if _v is not None:
            self['text'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
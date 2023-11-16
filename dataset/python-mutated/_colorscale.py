from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Colorscale(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.colorscale'
    _valid_props = {'diverging', 'sequential', 'sequentialminus'}

    @property
    def diverging(self):
        if False:
            print('Hello World!')
        "\n        Sets the default diverging colorscale. Note that\n        `autocolorscale` must be true for this attribute to work.\n\n        The 'diverging' property is a colorscale and may be\n        specified as:\n          - A list of colors that will be spaced evenly to create the colorscale.\n            Many predefined colorscale lists are included in the sequential, diverging,\n            and cyclical modules in the plotly.colors package.\n          - A list of 2-element lists where the first element is the\n            normalized color level value (starting at 0 and ending at 1),\n            and the second item is a valid color string.\n            (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])\n          - One of the following named colorscales:\n                ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',\n                 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',\n                 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',\n                 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',\n                 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',\n                 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',\n                 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',\n                 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',\n                 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',\n                 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',\n                 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',\n                 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',\n                 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',\n                 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',\n                 'ylorrd'].\n            Appending '_r' to a named colorscale reverses it.\n\n        Returns\n        -------\n        str\n        "
        return self['diverging']

    @diverging.setter
    def diverging(self, val):
        if False:
            print('Hello World!')
        self['diverging'] = val

    @property
    def sequential(self):
        if False:
            while True:
                i = 10
        "\n        Sets the default sequential colorscale for positive values.\n        Note that `autocolorscale` must be true for this attribute to\n        work.\n\n        The 'sequential' property is a colorscale and may be\n        specified as:\n          - A list of colors that will be spaced evenly to create the colorscale.\n            Many predefined colorscale lists are included in the sequential, diverging,\n            and cyclical modules in the plotly.colors package.\n          - A list of 2-element lists where the first element is the\n            normalized color level value (starting at 0 and ending at 1),\n            and the second item is a valid color string.\n            (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])\n          - One of the following named colorscales:\n                ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',\n                 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',\n                 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',\n                 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',\n                 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',\n                 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',\n                 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',\n                 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',\n                 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',\n                 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',\n                 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',\n                 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',\n                 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',\n                 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',\n                 'ylorrd'].\n            Appending '_r' to a named colorscale reverses it.\n\n        Returns\n        -------\n        str\n        "
        return self['sequential']

    @sequential.setter
    def sequential(self, val):
        if False:
            return 10
        self['sequential'] = val

    @property
    def sequentialminus(self):
        if False:
            return 10
        "\n        Sets the default sequential colorscale for negative values.\n        Note that `autocolorscale` must be true for this attribute to\n        work.\n\n        The 'sequentialminus' property is a colorscale and may be\n        specified as:\n          - A list of colors that will be spaced evenly to create the colorscale.\n            Many predefined colorscale lists are included in the sequential, diverging,\n            and cyclical modules in the plotly.colors package.\n          - A list of 2-element lists where the first element is the\n            normalized color level value (starting at 0 and ending at 1),\n            and the second item is a valid color string.\n            (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])\n          - One of the following named colorscales:\n                ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',\n                 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',\n                 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',\n                 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',\n                 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',\n                 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',\n                 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',\n                 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',\n                 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',\n                 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',\n                 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',\n                 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',\n                 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',\n                 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',\n                 'ylorrd'].\n            Appending '_r' to a named colorscale reverses it.\n\n        Returns\n        -------\n        str\n        "
        return self['sequentialminus']

    @sequentialminus.setter
    def sequentialminus(self, val):
        if False:
            return 10
        self['sequentialminus'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        diverging\n            Sets the default diverging colorscale. Note that\n            `autocolorscale` must be true for this attribute to\n            work.\n        sequential\n            Sets the default sequential colorscale for positive\n            values. Note that `autocolorscale` must be true for\n            this attribute to work.\n        sequentialminus\n            Sets the default sequential colorscale for negative\n            values. Note that `autocolorscale` must be true for\n            this attribute to work.\n        '

    def __init__(self, arg=None, diverging=None, sequential=None, sequentialminus=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Colorscale object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Colorscale`\n        diverging\n            Sets the default diverging colorscale. Note that\n            `autocolorscale` must be true for this attribute to\n            work.\n        sequential\n            Sets the default sequential colorscale for positive\n            values. Note that `autocolorscale` must be true for\n            this attribute to work.\n        sequentialminus\n            Sets the default sequential colorscale for negative\n            values. Note that `autocolorscale` must be true for\n            this attribute to work.\n\n        Returns\n        -------\n        Colorscale\n        '
        super(Colorscale, self).__init__('colorscale')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Colorscale\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Colorscale`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('diverging', None)
        _v = diverging if diverging is not None else _v
        if _v is not None:
            self['diverging'] = _v
        _v = arg.pop('sequential', None)
        _v = sequential if sequential is not None else _v
        if _v is not None:
            self['sequential'] = _v
        _v = arg.pop('sequentialminus', None)
        _v = sequentialminus if sequentialminus is not None else _v
        if _v is not None:
            self['sequentialminus'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
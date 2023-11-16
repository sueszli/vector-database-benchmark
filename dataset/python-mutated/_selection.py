from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Selection(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.selection'
    _valid_props = {'line', 'name', 'opacity', 'path', 'templateitemname', 'type', 'x0', 'x1', 'xref', 'y0', 'y1', 'yref'}

    @property
    def line(self):
        if False:
            print('Hello World!')
        '\n        The \'line\' property is an instance of Line\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.selection.Line`\n          - A dict of string/value properties that will be passed\n            to the Line constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the line color.\n                dash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                width\n                    Sets the line width (in px).\n\n        Returns\n        -------\n        plotly.graph_objs.layout.selection.Line\n        '
        return self['line']

    @line.setter
    def line(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['line'] = val

    @property
    def name(self):
        if False:
            return 10
        "\n        When used in a template, named items are created in the output\n        figure in addition to any items the figure already has in this\n        array. You can modify these items in the output figure by\n        making your own item with `templateitemname` matching this\n        `name` alongside your modifications (including `visible: false`\n        or `enabled: false` to hide it). Has no effect outside of a\n        template.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            while True:
                i = 10
        self['name'] = val

    @property
    def opacity(self):
        if False:
            while True:
                i = 10
        "\n        Sets the opacity of the selection.\n\n        The 'opacity' property is a number and may be specified as:\n          - An int or float in the interval [0, 1]\n\n        Returns\n        -------\n        int|float\n        "
        return self['opacity']

    @opacity.setter
    def opacity(self, val):
        if False:
            i = 10
            return i + 15
        self['opacity'] = val

    @property
    def path(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        For `type` "path" - a valid SVG path similar to `shapes.path`\n        in data coordinates. Allowed segments are: M, L and Z.\n\n        The \'path\' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        '
        return self['path']

    @path.setter
    def path(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['path'] = val

    @property
    def templateitemname(self):
        if False:
            while True:
                i = 10
        "\n        Used to refer to a named item in this array in the template.\n        Named items from the template will be created even without a\n        matching item in the input figure, but you can modify one by\n        making an item with `templateitemname` matching its `name`,\n        alongside your modifications (including `visible: false` or\n        `enabled: false` to hide it). If there is no template or no\n        matching item, this item will be hidden unless you explicitly\n        show it with `visible: true`.\n\n        The 'templateitemname' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['templateitemname']

    @templateitemname.setter
    def templateitemname(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['templateitemname'] = val

    @property
    def type(self):
        if False:
            print('Hello World!')
        '\n        Specifies the selection type to be drawn. If "rect", a\n        rectangle is drawn linking (`x0`,`y0`), (`x1`,`y0`),\n        (`x1`,`y1`) and (`x0`,`y1`). If "path", draw a custom SVG path\n        using `path`.\n\n        The \'type\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'rect\', \'path\']\n\n        Returns\n        -------\n        Any\n        '
        return self['type']

    @type.setter
    def type(self, val):
        if False:
            return 10
        self['type'] = val

    @property
    def x0(self):
        if False:
            print('Hello World!')
        "\n        Sets the selection's starting x position.\n\n        The 'x0' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['x0']

    @x0.setter
    def x0(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['x0'] = val

    @property
    def x1(self):
        if False:
            i = 10
            return i + 15
        "\n        Sets the selection's end x position.\n\n        The 'x1' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['x1']

    @x1.setter
    def x1(self, val):
        if False:
            return 10
        self['x1'] = val

    @property
    def xref(self):
        if False:
            return 10
        '\n        Sets the selection\'s x coordinate axis. If set to a x axis id\n        (e.g. "x" or "x2"), the `x` position refers to a x coordinate.\n        If set to "paper", the `x` position refers to the distance from\n        the left of the plotting area in normalized coordinates where 0\n        (1) corresponds to the left (right). If set to a x axis ID\n        followed by "domain" (separated by a space), the position\n        behaves like for "paper", but refers to the distance in\n        fractions of the domain length from the left of the domain of\n        that axis: e.g., *x2 domain* refers to the domain of the second\n        x  axis and a x position of 0.5 refers to the point between the\n        left and the right of the domain of the second x axis.\n\n        The \'xref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'paper\']\n          - A string that matches one of the following regular expressions:\n                [\'^x([2-9]|[1-9][0-9]+)?( domain)?$\']\n\n        Returns\n        -------\n        Any\n        '
        return self['xref']

    @xref.setter
    def xref(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['xref'] = val

    @property
    def y0(self):
        if False:
            print('Hello World!')
        "\n        Sets the selection's starting y position.\n\n        The 'y0' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['y0']

    @y0.setter
    def y0(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['y0'] = val

    @property
    def y1(self):
        if False:
            while True:
                i = 10
        "\n        Sets the selection's end y position.\n\n        The 'y1' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['y1']

    @y1.setter
    def y1(self, val):
        if False:
            while True:
                i = 10
        self['y1'] = val

    @property
    def yref(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the selection\'s x coordinate axis. If set to a y axis id\n        (e.g. "y" or "y2"), the `y` position refers to a y coordinate.\n        If set to "paper", the `y` position refers to the distance from\n        the bottom of the plotting area in normalized coordinates where\n        0 (1) corresponds to the bottom (top). If set to a y axis ID\n        followed by "domain" (separated by a space), the position\n        behaves like for "paper", but refers to the distance in\n        fractions of the domain length from the bottom of the domain of\n        that axis: e.g., *y2 domain* refers to the domain of the second\n        y  axis and a y position of 0.5 refers to the point between the\n        bottom and the top of the domain of the second y axis.\n\n        The \'yref\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'paper\']\n          - A string that matches one of the following regular expressions:\n                [\'^y([2-9]|[1-9][0-9]+)?( domain)?$\']\n\n        Returns\n        -------\n        Any\n        '
        return self['yref']

    @yref.setter
    def yref(self, val):
        if False:
            return 10
        self['yref'] = val

    @property
    def _prop_descriptions(self):
        if False:
            return 10
        return '        line\n            :class:`plotly.graph_objects.layout.selection.Line`\n            instance or dict with compatible properties\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        opacity\n            Sets the opacity of the selection.\n        path\n            For `type` "path" - a valid SVG path similar to\n            `shapes.path` in data coordinates. Allowed segments\n            are: M, L and Z.\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        type\n            Specifies the selection type to be drawn. If "rect", a\n            rectangle is drawn linking (`x0`,`y0`), (`x1`,`y0`),\n            (`x1`,`y1`) and (`x0`,`y1`). If "path", draw a custom\n            SVG path using `path`.\n        x0\n            Sets the selection\'s starting x position.\n        x1\n            Sets the selection\'s end x position.\n        xref\n            Sets the selection\'s x coordinate axis. If set to a x\n            axis id (e.g. "x" or "x2"), the `x` position refers to\n            a x coordinate. If set to "paper", the `x` position\n            refers to the distance from the left of the plotting\n            area in normalized coordinates where 0 (1) corresponds\n            to the left (right). If set to a x axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the left of the\n            domain of that axis: e.g., *x2 domain* refers to the\n            domain of the second x  axis and a x position of 0.5\n            refers to the point between the left and the right of\n            the domain of the second x axis.\n        y0\n            Sets the selection\'s starting y position.\n        y1\n            Sets the selection\'s end y position.\n        yref\n            Sets the selection\'s x coordinate axis. If set to a y\n            axis id (e.g. "y" or "y2"), the `y` position refers to\n            a y coordinate. If set to "paper", the `y` position\n            refers to the distance from the bottom of the plotting\n            area in normalized coordinates where 0 (1) corresponds\n            to the bottom (top). If set to a y axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the bottom of the\n            domain of that axis: e.g., *y2 domain* refers to the\n            domain of the second y  axis and a y position of 0.5\n            refers to the point between the bottom and the top of\n            the domain of the second y axis.\n        '

    def __init__(self, arg=None, line=None, name=None, opacity=None, path=None, templateitemname=None, type=None, x0=None, x1=None, xref=None, y0=None, y1=None, yref=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Selection object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Selection`\n        line\n            :class:`plotly.graph_objects.layout.selection.Line`\n            instance or dict with compatible properties\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        opacity\n            Sets the opacity of the selection.\n        path\n            For `type` "path" - a valid SVG path similar to\n            `shapes.path` in data coordinates. Allowed segments\n            are: M, L and Z.\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        type\n            Specifies the selection type to be drawn. If "rect", a\n            rectangle is drawn linking (`x0`,`y0`), (`x1`,`y0`),\n            (`x1`,`y1`) and (`x0`,`y1`). If "path", draw a custom\n            SVG path using `path`.\n        x0\n            Sets the selection\'s starting x position.\n        x1\n            Sets the selection\'s end x position.\n        xref\n            Sets the selection\'s x coordinate axis. If set to a x\n            axis id (e.g. "x" or "x2"), the `x` position refers to\n            a x coordinate. If set to "paper", the `x` position\n            refers to the distance from the left of the plotting\n            area in normalized coordinates where 0 (1) corresponds\n            to the left (right). If set to a x axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the left of the\n            domain of that axis: e.g., *x2 domain* refers to the\n            domain of the second x  axis and a x position of 0.5\n            refers to the point between the left and the right of\n            the domain of the second x axis.\n        y0\n            Sets the selection\'s starting y position.\n        y1\n            Sets the selection\'s end y position.\n        yref\n            Sets the selection\'s x coordinate axis. If set to a y\n            axis id (e.g. "y" or "y2"), the `y` position refers to\n            a y coordinate. If set to "paper", the `y` position\n            refers to the distance from the bottom of the plotting\n            area in normalized coordinates where 0 (1) corresponds\n            to the bottom (top). If set to a y axis ID followed by\n            "domain" (separated by a space), the position behaves\n            like for "paper", but refers to the distance in\n            fractions of the domain length from the bottom of the\n            domain of that axis: e.g., *y2 domain* refers to the\n            domain of the second y  axis and a y position of 0.5\n            refers to the point between the bottom and the top of\n            the domain of the second y axis.\n\n        Returns\n        -------\n        Selection\n        '
        super(Selection, self).__init__('selections')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Selection\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Selection`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('line', None)
        _v = line if line is not None else _v
        if _v is not None:
            self['line'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('opacity', None)
        _v = opacity if opacity is not None else _v
        if _v is not None:
            self['opacity'] = _v
        _v = arg.pop('path', None)
        _v = path if path is not None else _v
        if _v is not None:
            self['path'] = _v
        _v = arg.pop('templateitemname', None)
        _v = templateitemname if templateitemname is not None else _v
        if _v is not None:
            self['templateitemname'] = _v
        _v = arg.pop('type', None)
        _v = type if type is not None else _v
        if _v is not None:
            self['type'] = _v
        _v = arg.pop('x0', None)
        _v = x0 if x0 is not None else _v
        if _v is not None:
            self['x0'] = _v
        _v = arg.pop('x1', None)
        _v = x1 if x1 is not None else _v
        if _v is not None:
            self['x1'] = _v
        _v = arg.pop('xref', None)
        _v = xref if xref is not None else _v
        if _v is not None:
            self['xref'] = _v
        _v = arg.pop('y0', None)
        _v = y0 if y0 is not None else _v
        if _v is not None:
            self['y0'] = _v
        _v = arg.pop('y1', None)
        _v = y1 if y1 is not None else _v
        if _v is not None:
            self['y1'] = _v
        _v = arg.pop('yref', None)
        _v = yref if yref is not None else _v
        if _v is not None:
            self['yref'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Newselection(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout'
    _path_str = 'layout.newselection'
    _valid_props = {'line', 'mode'}

    @property
    def line(self):
        if False:
            print('Hello World!')
        '\n        The \'line\' property is an instance of Line\n        that may be specified as:\n          - An instance of :class:`plotly.graph_objs.layout.newselection.Line`\n          - A dict of string/value properties that will be passed\n            to the Line constructor\n\n            Supported dict properties:\n\n                color\n                    Sets the line color. By default uses either\n                    dark grey or white to increase contrast with\n                    background color.\n                dash\n                    Sets the dash style of lines. Set to a dash\n                    type string ("solid", "dot", "dash",\n                    "longdash", "dashdot", or "longdashdot") or a\n                    dash length list in px (eg "5px,10px,2px,2px").\n                width\n                    Sets the line width (in px).\n\n        Returns\n        -------\n        plotly.graph_objs.layout.newselection.Line\n        '
        return self['line']

    @line.setter
    def line(self, val):
        if False:
            while True:
                i = 10
        self['line'] = val

    @property
    def mode(self):
        if False:
            i = 10
            return i + 15
        "\n        Describes how a new selection is created. If `immediate`, a new\n        selection is created after first mouse up. If `gradual`, a new\n        selection is not created after first mouse. By adding to and\n        subtracting from the initial selection, this option allows\n        declaring extra outlines of the selection.\n\n        The 'mode' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['immediate', 'gradual']\n\n        Returns\n        -------\n        Any\n        "
        return self['mode']

    @mode.setter
    def mode(self, val):
        if False:
            while True:
                i = 10
        self['mode'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        line\n            :class:`plotly.graph_objects.layout.newselection.Line`\n            instance or dict with compatible properties\n        mode\n            Describes how a new selection is created. If\n            `immediate`, a new selection is created after first\n            mouse up. If `gradual`, a new selection is not created\n            after first mouse. By adding to and subtracting from\n            the initial selection, this option allows declaring\n            extra outlines of the selection.\n        '

    def __init__(self, arg=None, line=None, mode=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Newselection object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.Newselection`\n        line\n            :class:`plotly.graph_objects.layout.newselection.Line`\n            instance or dict with compatible properties\n        mode\n            Describes how a new selection is created. If\n            `immediate`, a new selection is created after first\n            mouse up. If `gradual`, a new selection is not created\n            after first mouse. By adding to and subtracting from\n            the initial selection, this option allows declaring\n            extra outlines of the selection.\n\n        Returns\n        -------\n        Newselection\n        '
        super(Newselection, self).__init__('newselection')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.Newselection\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.Newselection`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('line', None)
        _v = line if line is not None else _v
        if _v is not None:
            self['line'] = _v
        _v = arg.pop('mode', None)
        _v = mode if mode is not None else _v
        if _v is not None:
            self['mode'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
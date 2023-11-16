from plotly.basedatatypes import BaseFrameHierarchyType as _BaseFrameHierarchyType
import copy as _copy

class Frame(_BaseFrameHierarchyType):
    _parent_path_str = ''
    _path_str = 'frame'
    _valid_props = {'baseframe', 'data', 'group', 'layout', 'name', 'traces'}

    @property
    def baseframe(self):
        if False:
            while True:
                i = 10
        "\n        The name of the frame into which this frame's properties are\n        merged before applying. This is used to unify properties and\n        avoid needing to specify the same values for the same\n        properties in multiple frames.\n\n        The 'baseframe' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['baseframe']

    @baseframe.setter
    def baseframe(self, val):
        if False:
            i = 10
            return i + 15
        self['baseframe'] = val

    @property
    def data(self):
        if False:
            return 10
        '\n        A list of traces this frame modifies. The format is identical\n        to the normal trace definition.\n\n        Returns\n        -------\n        Any\n        '
        return self['data']

    @data.setter
    def data(self, val):
        if False:
            while True:
                i = 10
        self['data'] = val

    @property
    def group(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        An identifier that specifies the group to which the frame\n        belongs, used by animate to select a subset of frames.\n\n        The 'group' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['group']

    @group.setter
    def group(self, val):
        if False:
            return 10
        self['group'] = val

    @property
    def layout(self):
        if False:
            i = 10
            return i + 15
        '\n        Layout properties which this frame modifies. The format is\n        identical to the normal layout definition.\n\n        Returns\n        -------\n        Any\n        '
        return self['layout']

    @layout.setter
    def layout(self, val):
        if False:
            while True:
                i = 10
        self['layout'] = val

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        A label by which to identify the frame\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            i = 10
            return i + 15
        self['name'] = val

    @property
    def traces(self):
        if False:
            return 10
        "\n        A list of trace indices that identify the respective traces in\n        the data attribute\n\n        The 'traces' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['traces']

    @traces.setter
    def traces(self, val):
        if False:
            print('Hello World!')
        self['traces'] = val

    @property
    def _prop_descriptions(self):
        if False:
            print('Hello World!')
        return "        baseframe\n            The name of the frame into which this frame's\n            properties are merged before applying. This is used to\n            unify properties and avoid needing to specify the same\n            values for the same properties in multiple frames.\n        data\n            A list of traces this frame modifies. The format is\n            identical to the normal trace definition.\n        group\n            An identifier that specifies the group to which the\n            frame belongs, used by animate to select a subset of\n            frames.\n        layout\n            Layout properties which this frame modifies. The format\n            is identical to the normal layout definition.\n        name\n            A label by which to identify the frame\n        traces\n            A list of trace indices that identify the respective\n            traces in the data attribute\n        "

    def __init__(self, arg=None, baseframe=None, data=None, group=None, layout=None, name=None, traces=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Construct a new Frame object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of :class:`plotly.graph_objs.Frame`\n        baseframe\n            The name of the frame into which this frame's\n            properties are merged before applying. This is used to\n            unify properties and avoid needing to specify the same\n            values for the same properties in multiple frames.\n        data\n            A list of traces this frame modifies. The format is\n            identical to the normal trace definition.\n        group\n            An identifier that specifies the group to which the\n            frame belongs, used by animate to select a subset of\n            frames.\n        layout\n            Layout properties which this frame modifies. The format\n            is identical to the normal layout definition.\n        name\n            A label by which to identify the frame\n        traces\n            A list of trace indices that identify the respective\n            traces in the data attribute\n\n        Returns\n        -------\n        Frame\n        "
        super(Frame, self).__init__('frames')
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
            raise ValueError('The first argument to the plotly.graph_objs.Frame\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Frame`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('baseframe', None)
        _v = baseframe if baseframe is not None else _v
        if _v is not None:
            self['baseframe'] = _v
        _v = arg.pop('data', None)
        _v = data if data is not None else _v
        if _v is not None:
            self['data'] = _v
        _v = arg.pop('group', None)
        _v = group if group is not None else _v
        if _v is not None:
            self['group'] = _v
        _v = arg.pop('layout', None)
        _v = layout if layout is not None else _v
        if _v is not None:
            self['layout'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('traces', None)
        _v = traces if traces is not None else _v
        if _v is not None:
            self['traces'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Transition(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.slider'
    _path_str = 'layout.slider.transition'
    _valid_props = {'duration', 'easing'}

    @property
    def duration(self):
        if False:
            return 10
        "\n        Sets the duration of the slider transition\n\n        The 'duration' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['duration']

    @duration.setter
    def duration(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['duration'] = val

    @property
    def easing(self):
        if False:
            return 10
        "\n        Sets the easing function of the slider transition\n\n        The 'easing' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['linear', 'quad', 'cubic', 'sin', 'exp', 'circle',\n                'elastic', 'back', 'bounce', 'linear-in', 'quad-in',\n                'cubic-in', 'sin-in', 'exp-in', 'circle-in', 'elastic-in',\n                'back-in', 'bounce-in', 'linear-out', 'quad-out',\n                'cubic-out', 'sin-out', 'exp-out', 'circle-out',\n                'elastic-out', 'back-out', 'bounce-out', 'linear-in-out',\n                'quad-in-out', 'cubic-in-out', 'sin-in-out', 'exp-in-out',\n                'circle-in-out', 'elastic-in-out', 'back-in-out',\n                'bounce-in-out']\n\n        Returns\n        -------\n        Any\n        "
        return self['easing']

    @easing.setter
    def easing(self, val):
        if False:
            return 10
        self['easing'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        duration\n            Sets the duration of the slider transition\n        easing\n            Sets the easing function of the slider transition\n        '

    def __init__(self, arg=None, duration=None, easing=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Transition object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.slider.Transition`\n        duration\n            Sets the duration of the slider transition\n        easing\n            Sets the easing function of the slider transition\n\n        Returns\n        -------\n        Transition\n        '
        super(Transition, self).__init__('transition')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.slider.Transition\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.slider.Transition`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('duration', None)
        _v = duration if duration is not None else _v
        if _v is not None:
            self['duration'] = _v
        _v = arg.pop('easing', None)
        _v = easing if easing is not None else _v
        if _v is not None:
            self['easing'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class Cumulative(_BaseTraceHierarchyType):
    _parent_path_str = 'histogram'
    _path_str = 'histogram.cumulative'
    _valid_props = {'currentbin', 'direction', 'enabled'}

    @property
    def currentbin(self):
        if False:
            while True:
                i = 10
        '\n        Only applies if cumulative is enabled. Sets whether the current\n        bin is included, excluded, or has half of its value included in\n        the current cumulative value. "include" is the default for\n        compatibility with various other tools, however it introduces a\n        half-bin bias to the results. "exclude" makes the opposite\n        half-bin bias, and "half" removes it.\n\n        The \'currentbin\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'include\', \'exclude\', \'half\']\n\n        Returns\n        -------\n        Any\n        '
        return self['currentbin']

    @currentbin.setter
    def currentbin(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['currentbin'] = val

    @property
    def direction(self):
        if False:
            while True:
                i = 10
        '\n        Only applies if cumulative is enabled. If "increasing"\n        (default) we sum all prior bins, so the result increases from\n        left to right. If "decreasing" we sum later bins so the result\n        decreases from left to right.\n\n        The \'direction\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'increasing\', \'decreasing\']\n\n        Returns\n        -------\n        Any\n        '
        return self['direction']

    @direction.setter
    def direction(self, val):
        if False:
            return 10
        self['direction'] = val

    @property
    def enabled(self):
        if False:
            while True:
                i = 10
        '\n        If true, display the cumulative distribution by summing the\n        binned values. Use the `direction` and `centralbin` attributes\n        to tune the accumulation method. Note: in this mode, the\n        "density" `histnorm` settings behave the same as their\n        equivalents without "density": "" and "density" both rise to\n        the number of data points, and "probability" and *probability\n        density* both rise to the number of sample points.\n\n        The \'enabled\' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        '
        return self['enabled']

    @enabled.setter
    def enabled(self, val):
        if False:
            while True:
                i = 10
        self['enabled'] = val

    @property
    def _prop_descriptions(self):
        if False:
            i = 10
            return i + 15
        return '        currentbin\n            Only applies if cumulative is enabled. Sets whether the\n            current bin is included, excluded, or has half of its\n            value included in the current cumulative value.\n            "include" is the default for compatibility with various\n            other tools, however it introduces a half-bin bias to\n            the results. "exclude" makes the opposite half-bin\n            bias, and "half" removes it.\n        direction\n            Only applies if cumulative is enabled. If "increasing"\n            (default) we sum all prior bins, so the result\n            increases from left to right. If "decreasing" we sum\n            later bins so the result decreases from left to right.\n        enabled\n            If true, display the cumulative distribution by summing\n            the binned values. Use the `direction` and `centralbin`\n            attributes to tune the accumulation method. Note: in\n            this mode, the "density" `histnorm` settings behave the\n            same as their equivalents without "density": "" and\n            "density" both rise to the number of data points, and\n            "probability" and *probability density* both rise to\n            the number of sample points.\n        '

    def __init__(self, arg=None, currentbin=None, direction=None, enabled=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new Cumulative object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.histogram.Cumulative`\n        currentbin\n            Only applies if cumulative is enabled. Sets whether the\n            current bin is included, excluded, or has half of its\n            value included in the current cumulative value.\n            "include" is the default for compatibility with various\n            other tools, however it introduces a half-bin bias to\n            the results. "exclude" makes the opposite half-bin\n            bias, and "half" removes it.\n        direction\n            Only applies if cumulative is enabled. If "increasing"\n            (default) we sum all prior bins, so the result\n            increases from left to right. If "decreasing" we sum\n            later bins so the result decreases from left to right.\n        enabled\n            If true, display the cumulative distribution by summing\n            the binned values. Use the `direction` and `centralbin`\n            attributes to tune the accumulation method. Note: in\n            this mode, the "density" `histnorm` settings behave the\n            same as their equivalents without "density": "" and\n            "density" both rise to the number of data points, and\n            "probability" and *probability density* both rise to\n            the number of sample points.\n\n        Returns\n        -------\n        Cumulative\n        '
        super(Cumulative, self).__init__('cumulative')
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
            raise ValueError('The first argument to the plotly.graph_objs.histogram.Cumulative\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.histogram.Cumulative`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('currentbin', None)
        _v = currentbin if currentbin is not None else _v
        if _v is not None:
            self['currentbin'] = _v
        _v = arg.pop('direction', None)
        _v = direction if direction is not None else _v
        if _v is not None:
            self['direction'] = _v
        _v = arg.pop('enabled', None)
        _v = enabled if enabled is not None else _v
        if _v is not None:
            self['enabled'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
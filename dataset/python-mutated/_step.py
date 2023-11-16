from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Step(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.slider'
    _path_str = 'layout.slider.step'
    _valid_props = {'args', 'execute', 'label', 'method', 'name', 'templateitemname', 'value', 'visible'}

    @property
    def args(self):
        if False:
            return 10
        "\n            Sets the arguments values to be passed to the Plotly method set\n            in `method` on slide.\n\n            The 'args' property is an info array that may be specified as:\n\n            * a list or tuple of up to 3 elements where:\n        (0) The 'args[0]' property accepts values of any type\n        (1) The 'args[1]' property accepts values of any type\n        (2) The 'args[2]' property accepts values of any type\n\n            Returns\n            -------\n            list\n        "
        return self['args']

    @args.setter
    def args(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['args'] = val

    @property
    def execute(self):
        if False:
            i = 10
            return i + 15
        "\n        When true, the API method is executed. When false, all other\n        behaviors are the same and command execution is skipped. This\n        may be useful when hooking into, for example, the\n        `plotly_sliderchange` method and executing the API command\n        manually without losing the benefit of the slider automatically\n        binding to the state of the plot through the specification of\n        `method` and `args`.\n\n        The 'execute' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['execute']

    @execute.setter
    def execute(self, val):
        if False:
            print('Hello World!')
        self['execute'] = val

    @property
    def label(self):
        if False:
            print('Hello World!')
        "\n        Sets the text label to appear on the slider\n\n        The 'label' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['label']

    @label.setter
    def label(self, val):
        if False:
            print('Hello World!')
        self['label'] = val

    @property
    def method(self):
        if False:
            while True:
                i = 10
        "\n        Sets the Plotly method to be called when the slider value is\n        changed. If the `skip` method is used, the API slider will\n        function as normal but will perform no API calls and will not\n        bind automatically to state updates. This may be used to create\n        a component interface and attach to slider events manually via\n        JavaScript.\n\n        The 'method' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                ['restyle', 'relayout', 'animate', 'update', 'skip']\n\n        Returns\n        -------\n        Any\n        "
        return self['method']

    @method.setter
    def method(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['method'] = val

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        When used in a template, named items are created in the output\n        figure in addition to any items the figure already has in this\n        array. You can modify these items in the output figure by\n        making your own item with `templateitemname` matching this\n        `name` alongside your modifications (including `visible: false`\n        or `enabled: false` to hide it). Has no effect outside of a\n        template.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['name'] = val

    @property
    def templateitemname(self):
        if False:
            return 10
        "\n        Used to refer to a named item in this array in the template.\n        Named items from the template will be created even without a\n        matching item in the input figure, but you can modify one by\n        making an item with `templateitemname` matching its `name`,\n        alongside your modifications (including `visible: false` or\n        `enabled: false` to hide it). If there is no template or no\n        matching item, this item will be hidden unless you explicitly\n        show it with `visible: true`.\n\n        The 'templateitemname' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['templateitemname']

    @templateitemname.setter
    def templateitemname(self, val):
        if False:
            print('Hello World!')
        self['templateitemname'] = val

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the value of the slider step, used to refer to the step\n        programatically. Defaults to the slider label if not provided.\n\n        The 'value' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['value']

    @value.setter
    def value(self, val):
        if False:
            while True:
                i = 10
        self['value'] = val

    @property
    def visible(self):
        if False:
            while True:
                i = 10
        "\n        Determines whether or not this step is included in the slider.\n\n        The 'visible' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        "
        return self['visible']

    @visible.setter
    def visible(self, val):
        if False:
            while True:
                i = 10
        self['visible'] = val

    @property
    def _prop_descriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return '        args\n            Sets the arguments values to be passed to the Plotly\n            method set in `method` on slide.\n        execute\n            When true, the API method is executed. When false, all\n            other behaviors are the same and command execution is\n            skipped. This may be useful when hooking into, for\n            example, the `plotly_sliderchange` method and executing\n            the API command manually without losing the benefit of\n            the slider automatically binding to the state of the\n            plot through the specification of `method` and `args`.\n        label\n            Sets the text label to appear on the slider\n        method\n            Sets the Plotly method to be called when the slider\n            value is changed. If the `skip` method is used, the API\n            slider will function as normal but will perform no API\n            calls and will not bind automatically to state updates.\n            This may be used to create a component interface and\n            attach to slider events manually via JavaScript.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        value\n            Sets the value of the slider step, used to refer to the\n            step programatically. Defaults to the slider label if\n            not provided.\n        visible\n            Determines whether or not this step is included in the\n            slider.\n        '

    def __init__(self, arg=None, args=None, execute=None, label=None, method=None, name=None, templateitemname=None, value=None, visible=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new Step object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.slider.Step`\n        args\n            Sets the arguments values to be passed to the Plotly\n            method set in `method` on slide.\n        execute\n            When true, the API method is executed. When false, all\n            other behaviors are the same and command execution is\n            skipped. This may be useful when hooking into, for\n            example, the `plotly_sliderchange` method and executing\n            the API command manually without losing the benefit of\n            the slider automatically binding to the state of the\n            plot through the specification of `method` and `args`.\n        label\n            Sets the text label to appear on the slider\n        method\n            Sets the Plotly method to be called when the slider\n            value is changed. If the `skip` method is used, the API\n            slider will function as normal but will perform no API\n            calls and will not bind automatically to state updates.\n            This may be used to create a component interface and\n            attach to slider events manually via JavaScript.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        value\n            Sets the value of the slider step, used to refer to the\n            step programatically. Defaults to the slider label if\n            not provided.\n        visible\n            Determines whether or not this step is included in the\n            slider.\n\n        Returns\n        -------\n        Step\n        '
        super(Step, self).__init__('steps')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.slider.Step\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.slider.Step`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('args', None)
        _v = args if args is not None else _v
        if _v is not None:
            self['args'] = _v
        _v = arg.pop('execute', None)
        _v = execute if execute is not None else _v
        if _v is not None:
            self['execute'] = _v
        _v = arg.pop('label', None)
        _v = label if label is not None else _v
        if _v is not None:
            self['label'] = _v
        _v = arg.pop('method', None)
        _v = method if method is not None else _v
        if _v is not None:
            self['method'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('templateitemname', None)
        _v = templateitemname if templateitemname is not None else _v
        if _v is not None:
            self['templateitemname'] = _v
        _v = arg.pop('value', None)
        _v = value if value is not None else _v
        if _v is not None:
            self['value'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
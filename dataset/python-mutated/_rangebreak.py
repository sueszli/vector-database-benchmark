from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
import copy as _copy

class Rangebreak(_BaseLayoutHierarchyType):
    _parent_path_str = 'layout.xaxis'
    _path_str = 'layout.xaxis.rangebreak'
    _valid_props = {'bounds', 'dvalue', 'enabled', 'name', 'pattern', 'templateitemname', 'values'}

    @property
    def bounds(self):
        if False:
            while True:
                i = 10
        "\n            Sets the lower and upper bounds of this axis rangebreak. Can be\n            used with `pattern`.\n\n            The 'bounds' property is an info array that may be specified as:\n\n            * a list or tuple of 2 elements where:\n        (0) The 'bounds[0]' property accepts values of any type\n        (1) The 'bounds[1]' property accepts values of any type\n\n            Returns\n            -------\n            list\n        "
        return self['bounds']

    @bounds.setter
    def bounds(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['bounds'] = val

    @property
    def dvalue(self):
        if False:
            while True:
                i = 10
        "\n        Sets the size of each `values` item. The default is one day in\n        milliseconds.\n\n        The 'dvalue' property is a number and may be specified as:\n          - An int or float in the interval [0, inf]\n\n        Returns\n        -------\n        int|float\n        "
        return self['dvalue']

    @dvalue.setter
    def dvalue(self, val):
        if False:
            print('Hello World!')
        self['dvalue'] = val

    @property
    def enabled(self):
        if False:
            return 10
        '\n        Determines whether this axis rangebreak is enabled or disabled.\n        Please note that `rangebreaks` only work for "date" axis type.\n\n        The \'enabled\' property must be specified as a bool\n        (either True, or False)\n\n        Returns\n        -------\n        bool\n        '
        return self['enabled']

    @enabled.setter
    def enabled(self, val):
        if False:
            while True:
                i = 10
        self['enabled'] = val

    @property
    def name(self):
        if False:
            while True:
                i = 10
        "\n        When used in a template, named items are created in the output\n        figure in addition to any items the figure already has in this\n        array. You can modify these items in the output figure by\n        making your own item with `templateitemname` matching this\n        `name` alongside your modifications (including `visible: false`\n        or `enabled: false` to hide it). Has no effect outside of a\n        template.\n\n        The 'name' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['name']

    @name.setter
    def name(self, val):
        if False:
            return 10
        self['name'] = val

    @property
    def pattern(self):
        if False:
            print('Hello World!')
        '\n        Determines a pattern on the time line that generates breaks. If\n        *day of week* - days of the week in English e.g. \'Sunday\' or\n        `sun` (matching is case-insensitive and considers only the\n        first three characters), as well as Sunday-based integers\n        between 0 and 6. If "hour" - hour (24-hour clock) as decimal\n        numbers between 0 and 24. for more info. Examples: - { pattern:\n        \'day of week\', bounds: [6, 1] }  or simply { bounds: [\'sat\',\n        \'mon\'] }   breaks from Saturday to Monday (i.e. skips the\n        weekends). - { pattern: \'hour\', bounds: [17, 8] }   breaks from\n        5pm to 8am (i.e. skips non-work hours).\n\n        The \'pattern\' property is an enumeration that may be specified as:\n          - One of the following enumeration values:\n                [\'day of week\', \'hour\', \'\']\n\n        Returns\n        -------\n        Any\n        '
        return self['pattern']

    @pattern.setter
    def pattern(self, val):
        if False:
            i = 10
            return i + 15
        self['pattern'] = val

    @property
    def templateitemname(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Used to refer to a named item in this array in the template.\n        Named items from the template will be created even without a\n        matching item in the input figure, but you can modify one by\n        making an item with `templateitemname` matching its `name`,\n        alongside your modifications (including `visible: false` or\n        `enabled: false` to hide it). If there is no template or no\n        matching item, this item will be hidden unless you explicitly\n        show it with `visible: true`.\n\n        The 'templateitemname' property is a string and must be specified as:\n          - A string\n          - A number that will be converted to a string\n\n        Returns\n        -------\n        str\n        "
        return self['templateitemname']

    @templateitemname.setter
    def templateitemname(self, val):
        if False:
            for i in range(10):
                print('nop')
        self['templateitemname'] = val

    @property
    def values(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the coordinate values corresponding to the rangebreaks. An\n        alternative to `bounds`. Use `dvalue` to set the size of the\n        values along the axis.\n\n        The 'values' property is an info array that may be specified as:\n        * a list of elements where:\n          The 'values[i]' property accepts values of any type\n\n        Returns\n        -------\n        list\n        "
        return self['values']

    @values.setter
    def values(self, val):
        if False:
            i = 10
            return i + 15
        self['values'] = val

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        return '        bounds\n            Sets the lower and upper bounds of this axis\n            rangebreak. Can be used with `pattern`.\n        dvalue\n            Sets the size of each `values` item. The default is one\n            day in milliseconds.\n        enabled\n            Determines whether this axis rangebreak is enabled or\n            disabled. Please note that `rangebreaks` only work for\n            "date" axis type.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        pattern\n            Determines a pattern on the time line that generates\n            breaks. If *day of week* - days of the week in English\n            e.g. \'Sunday\' or `sun` (matching is case-insensitive\n            and considers only the first three characters), as well\n            as Sunday-based integers between 0 and 6. If "hour" -\n            hour (24-hour clock) as decimal numbers between 0 and\n            24. for more info. Examples: - { pattern: \'day of\n            week\', bounds: [6, 1] }  or simply { bounds: [\'sat\',\n            \'mon\'] }   breaks from Saturday to Monday (i.e. skips\n            the weekends). - { pattern: \'hour\', bounds: [17, 8] }\n            breaks from 5pm to 8am (i.e. skips non-work hours).\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        values\n            Sets the coordinate values corresponding to the\n            rangebreaks. An alternative to `bounds`. Use `dvalue`\n            to set the size of the values along the axis.\n        '

    def __init__(self, arg=None, bounds=None, dvalue=None, enabled=None, name=None, pattern=None, templateitemname=None, values=None, **kwargs):
        if False:
            return 10
        '\n        Construct a new Rangebreak object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.layout.xaxis.Rangebreak`\n        bounds\n            Sets the lower and upper bounds of this axis\n            rangebreak. Can be used with `pattern`.\n        dvalue\n            Sets the size of each `values` item. The default is one\n            day in milliseconds.\n        enabled\n            Determines whether this axis rangebreak is enabled or\n            disabled. Please note that `rangebreaks` only work for\n            "date" axis type.\n        name\n            When used in a template, named items are created in the\n            output figure in addition to any items the figure\n            already has in this array. You can modify these items\n            in the output figure by making your own item with\n            `templateitemname` matching this `name` alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). Has no effect outside of a\n            template.\n        pattern\n            Determines a pattern on the time line that generates\n            breaks. If *day of week* - days of the week in English\n            e.g. \'Sunday\' or `sun` (matching is case-insensitive\n            and considers only the first three characters), as well\n            as Sunday-based integers between 0 and 6. If "hour" -\n            hour (24-hour clock) as decimal numbers between 0 and\n            24. for more info. Examples: - { pattern: \'day of\n            week\', bounds: [6, 1] }  or simply { bounds: [\'sat\',\n            \'mon\'] }   breaks from Saturday to Monday (i.e. skips\n            the weekends). - { pattern: \'hour\', bounds: [17, 8] }\n            breaks from 5pm to 8am (i.e. skips non-work hours).\n        templateitemname\n            Used to refer to a named item in this array in the\n            template. Named items from the template will be created\n            even without a matching item in the input figure, but\n            you can modify one by making an item with\n            `templateitemname` matching its `name`, alongside your\n            modifications (including `visible: false` or `enabled:\n            false` to hide it). If there is no template or no\n            matching item, this item will be hidden unless you\n            explicitly show it with `visible: true`.\n        values\n            Sets the coordinate values corresponding to the\n            rangebreaks. An alternative to `bounds`. Use `dvalue`\n            to set the size of the values along the axis.\n\n        Returns\n        -------\n        Rangebreak\n        '
        super(Rangebreak, self).__init__('rangebreaks')
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
            raise ValueError('The first argument to the plotly.graph_objs.layout.xaxis.Rangebreak\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.layout.xaxis.Rangebreak`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('bounds', None)
        _v = bounds if bounds is not None else _v
        if _v is not None:
            self['bounds'] = _v
        _v = arg.pop('dvalue', None)
        _v = dvalue if dvalue is not None else _v
        if _v is not None:
            self['dvalue'] = _v
        _v = arg.pop('enabled', None)
        _v = enabled if enabled is not None else _v
        if _v is not None:
            self['enabled'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('pattern', None)
        _v = pattern if pattern is not None else _v
        if _v is not None:
            self['pattern'] = _v
        _v = arg.pop('templateitemname', None)
        _v = templateitemname if templateitemname is not None else _v
        if _v is not None:
            self['templateitemname'] = _v
        _v = arg.pop('values', None)
        _v = values if values is not None else _v
        if _v is not None:
            self['values'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
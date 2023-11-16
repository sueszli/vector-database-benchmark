from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy

class YBins(_BaseTraceHierarchyType):
    _parent_path_str = 'histogram2d'
    _path_str = 'histogram2d.ybins'
    _valid_props = {'end', 'size', 'start'}

    @property
    def end(self):
        if False:
            print('Hello World!')
        "\n        Sets the end value for the y axis bins. The last bin may not\n        end exactly at this value, we increment the bin edge by `size`\n        from `start` until we reach or exceed `end`. Defaults to the\n        maximum data value. Like `start`, for dates use a date string,\n        and for category data `end` is based on the category serial\n        numbers.\n\n        The 'end' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['end']

    @end.setter
    def end(self, val):
        if False:
            i = 10
            return i + 15
        self['end'] = val

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        '\n        Sets the size of each y axis bin. Default behavior: If `nbinsy`\n        is 0 or omitted, we choose a nice round bin size such that the\n        number of bins is about the same as the typical number of\n        samples in each bin. If `nbinsy` is provided, we choose a nice\n        round bin size giving no more than that many bins. For date\n        data, use milliseconds or "M<n>" for months, as in\n        `axis.dtick`. For category data, the number of categories to\n        bin together (always defaults to 1).\n\n        The \'size\' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        '
        return self['size']

    @size.setter
    def size(self, val):
        if False:
            return 10
        self['size'] = val

    @property
    def start(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the starting value for the y axis bins. Defaults to the\n        minimum data value, shifted down if necessary to make nice\n        round values and to remove ambiguous bin edges. For example, if\n        most of the data is integers we shift the bin edges 0.5 down,\n        so a `size` of 5 would have a default `start` of -0.5, so it is\n        clear that 0-4 are in the first bin, 5-9 in the second, but\n        continuous data gets a start of 0 and bins [0,5), [5,10) etc.\n        Dates behave similarly, and `start` should be a date string.\n        For category data, `start` is based on the category serial\n        numbers, and defaults to -0.5.\n\n        The 'start' property accepts values of any type\n\n        Returns\n        -------\n        Any\n        "
        return self['start']

    @start.setter
    def start(self, val):
        if False:
            print('Hello World!')
        self['start'] = val

    @property
    def _prop_descriptions(self):
        if False:
            i = 10
            return i + 15
        return '        end\n            Sets the end value for the y axis bins. The last bin\n            may not end exactly at this value, we increment the bin\n            edge by `size` from `start` until we reach or exceed\n            `end`. Defaults to the maximum data value. Like\n            `start`, for dates use a date string, and for category\n            data `end` is based on the category serial numbers.\n        size\n            Sets the size of each y axis bin. Default behavior: If\n            `nbinsy` is 0 or omitted, we choose a nice round bin\n            size such that the number of bins is about the same as\n            the typical number of samples in each bin. If `nbinsy`\n            is provided, we choose a nice round bin size giving no\n            more than that many bins. For date data, use\n            milliseconds or "M<n>" for months, as in `axis.dtick`.\n            For category data, the number of categories to bin\n            together (always defaults to 1).\n        start\n            Sets the starting value for the y axis bins. Defaults\n            to the minimum data value, shifted down if necessary to\n            make nice round values and to remove ambiguous bin\n            edges. For example, if most of the data is integers we\n            shift the bin edges 0.5 down, so a `size` of 5 would\n            have a default `start` of -0.5, so it is clear that 0-4\n            are in the first bin, 5-9 in the second, but continuous\n            data gets a start of 0 and bins [0,5), [5,10) etc.\n            Dates behave similarly, and `start` should be a date\n            string. For category data, `start` is based on the\n            category serial numbers, and defaults to -0.5.\n        '

    def __init__(self, arg=None, end=None, size=None, start=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Construct a new YBins object\n\n        Parameters\n        ----------\n        arg\n            dict of properties compatible with this constructor or\n            an instance of\n            :class:`plotly.graph_objs.histogram2d.YBins`\n        end\n            Sets the end value for the y axis bins. The last bin\n            may not end exactly at this value, we increment the bin\n            edge by `size` from `start` until we reach or exceed\n            `end`. Defaults to the maximum data value. Like\n            `start`, for dates use a date string, and for category\n            data `end` is based on the category serial numbers.\n        size\n            Sets the size of each y axis bin. Default behavior: If\n            `nbinsy` is 0 or omitted, we choose a nice round bin\n            size such that the number of bins is about the same as\n            the typical number of samples in each bin. If `nbinsy`\n            is provided, we choose a nice round bin size giving no\n            more than that many bins. For date data, use\n            milliseconds or "M<n>" for months, as in `axis.dtick`.\n            For category data, the number of categories to bin\n            together (always defaults to 1).\n        start\n            Sets the starting value for the y axis bins. Defaults\n            to the minimum data value, shifted down if necessary to\n            make nice round values and to remove ambiguous bin\n            edges. For example, if most of the data is integers we\n            shift the bin edges 0.5 down, so a `size` of 5 would\n            have a default `start` of -0.5, so it is clear that 0-4\n            are in the first bin, 5-9 in the second, but continuous\n            data gets a start of 0 and bins [0,5), [5,10) etc.\n            Dates behave similarly, and `start` should be a date\n            string. For category data, `start` is based on the\n            category serial numbers, and defaults to -0.5.\n\n        Returns\n        -------\n        YBins\n        '
        super(YBins, self).__init__('ybins')
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
            raise ValueError('The first argument to the plotly.graph_objs.histogram2d.YBins\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.histogram2d.YBins`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('end', None)
        _v = end if end is not None else _v
        if _v is not None:
            self['end'] = _v
        _v = arg.pop('size', None)
        _v = size if size is not None else _v
        if _v is not None:
            self['size'] = _v
        _v = arg.pop('start', None)
        _v = start if start is not None else _v
        if _v is not None:
            self['start'] = _v
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False
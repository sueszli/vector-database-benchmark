class ViewConfig(object):
    """Defines the configuration for a View object."""

    def __init__(self, **config):
        if False:
            while True:
                i = 10
        'Receives a user-provided config dict and standardizes it for\n        reference.\n\n        Keyword Arguments:\n            columns (:obj:`list` of :obj:`str`): A list of column names to be\n                visible to the user.\n            group_by (:obj:`list` of :obj:`str`): A list of column names to\n                use as group by.\n            split_by (:obj:`list` of :obj:`str`): A list of column names\n                to use as split by.\n            aggregates (:obj:`dict` of :obj:`str` to :obj:`str`):  A dictionary\n                of column names to aggregate types, which specify aggregates\n                for individual columns.\n            sort (:obj:`list` of :obj:`list` of :obj:`str`): A list of lists,\n                each list containing a column name and a sort direction\n                (``asc``, ``desc``, ``asc abs``, ``desc abs``, ``col asc``,\n                ``col desc``, ``col asc abs``, ``col desc abs``).\n            filter (:obj:`list` of :obj:`list` of :obj:`str`):  A list of lists,\n                each list containing a column name, a filter comparator, and a\n                value to filter by.\n            expressions (:obj:`list` of :obj:`str`):  A list of string\n                expressions which will be calculated by the view.\n        '
        self._config = config
        self._group_by = self._config.get('group_by', [])
        self._split_by = self._config.get('split_by', [])
        self._aggregates = self._config.get('aggregates', {})
        self._columns = self._config.get('columns', [])
        self._sort = self._config.get('sort', [])
        self._filter = self._config.get('filter', [])
        self._expressions = self._config.get('expressions', [])
        self._filter_op = self._config.get('filter_op', 'and')
        self.group_by_depth = self._config.get('group_by_depth', None)
        self.split_by_depth = self._config.get('split_by_depth', None)

    def get_group_by(self):
        if False:
            return 10
        'The columns used as\n        [group by](https://en.wikipedia.org/wiki/Pivot_table#Row_labels)\n\n        Returns:\n            list : the columns used as group by\n        '
        return self._group_by

    def get_split_by(self):
        if False:
            while True:
                i = 10
        'The columns used as\n        [split by](https://en.wikipedia.org/wiki/Pivot_table#Column_labels)\n\n        Returns:\n            list : the columns used as split by\n        '
        return self._split_by

    def get_aggregates(self):
        if False:
            while True:
                i = 10
        'Defines the grouping of data within columns.\n\n        Returns:\n            dict[str:str]  a vector of string vectors in which the first value\n                is the column name, and the second value is the string\n                representation of an aggregate\n        '
        return self._aggregates

    def get_columns(self):
        if False:
            print('Hello World!')
        'The columns that will be shown to the user in the view. If left\n        empty, the view shows all columns in the dataset by default.\n\n        Returns:\n            `list` : the columns shown to the user\n        '
        return self._columns

    def get_sort(self):
        if False:
            for i in range(10):
                print('nop')
        'The columns that should be sorted, and the direction to sort.\n\n        A sort configuration is a `list` of two elements: a string column name,\n        and a string sort direction, which are:  "none", "asc", "desc",\n        "col asc", "col desc", "asc abs", "desc abs", "col asc abs", and\n        "col desc abs".\n\n        Returns:\n            `list`: the sort configurations of the view stored in a `list` of\n                `list`s\n        '
        return self._sort

    def get_expressions(self):
        if False:
            while True:
                i = 10
        'A list of string expressions that should be calculated.'
        return self._expressions

    def get_filter(self):
        if False:
            return 10
        'The columns that should be filtered.\n\n        A filter configuration is a `list` of three elements:\n            0: `str` column name.\n            1: a filter comparison string (i.e. "===", ">")\n            2: a value to compare (this will be casted to match the type of\n                the column)\n\n        Returns:\n            `list`: the filter configurations of the view stored in a `list` of\n                lists\n        '
        return self._filter

    def get_filter_op(self):
        if False:
            return 10
        'When multiple filters are applied, filter_op defines how data should\n        be returned.\n\n        Defaults to "and" if not set by the user, meaning that data returned\n        with multiple filters will satisfy all filters.  If "or" is provided,\n        returned data will satsify any one of the filters applied.\n\n        Returns:\n            `str`: the filter_op of the view\n        '
        return self._filter_op

    def get_config(self):
        if False:
            while True:
                i = 10
        'Returns the original dictionary config passed by the user.'
        return self._config
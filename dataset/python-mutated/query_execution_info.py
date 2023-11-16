"""Internal class for partitioned query execution info implementation in the Azure Cosmos database service.
"""
from azure.cosmos.documents import _DistinctType

class _PartitionedQueryExecutionInfo(object):
    """Represents a wrapper helper for partitioned query execution info
    dictionary returned by the backend.
    """
    QueryInfoPath = 'queryInfo'
    HasSelectValue = [QueryInfoPath, 'hasSelectValue']
    TopPath = [QueryInfoPath, 'top']
    OffsetPath = [QueryInfoPath, 'offset']
    LimitPath = [QueryInfoPath, 'limit']
    DistinctTypePath = [QueryInfoPath, 'distinctType']
    OrderByPath = [QueryInfoPath, 'orderBy']
    AggregatesPath = [QueryInfoPath, 'aggregates']
    QueryRangesPath = 'queryRanges'
    RewrittenQueryPath = [QueryInfoPath, 'rewrittenQuery']

    def __init__(self, query_execution_info):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param dict query_execution_info:\n        '
        self._query_execution_info = query_execution_info

    def get_top(self):
        if False:
            while True:
                i = 10
        'Returns the top count (if any) or None.\n        :returns: The top count.\n        :rtype: int\n        '
        return self._extract(_PartitionedQueryExecutionInfo.TopPath)

    def get_limit(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the limit count (if any) or None.\n        :returns: The limit count.\n        :rtype: int\n        '
        return self._extract(_PartitionedQueryExecutionInfo.LimitPath)

    def get_offset(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the offset count (if any) or None.\n        :returns: The offset count.\n        :rtype: int\n        '
        return self._extract(_PartitionedQueryExecutionInfo.OffsetPath)

    def get_distinct_type(self):
        if False:
            while True:
                i = 10
        'Returns the distinct type (if any) or None.\n        :returns: The distinct type.\n        :rtype: str\n        '
        return self._extract(_PartitionedQueryExecutionInfo.DistinctTypePath)

    def get_order_by(self):
        if False:
            i = 10
            return i + 15
        'Returns order by items (if any) or None.\n        :returns: The order by items.\n        :rtype: list\n        '
        return self._extract(_PartitionedQueryExecutionInfo.OrderByPath)

    def get_aggregates(self):
        if False:
            return 10
        'Returns aggregators (if any) or None.\n        :returns: The aggregate items.\n        :rtype: list\n        '
        return self._extract(_PartitionedQueryExecutionInfo.AggregatesPath)

    def get_query_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns query partition ranges (if any) or None.\n        :returns: The query ranges.\n        :rtype: list\n        '
        return self._extract(_PartitionedQueryExecutionInfo.QueryRangesPath)

    def get_rewritten_query(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns rewritten query or None (if any).\n        :returns: The rewritten query.\n        :rtype: str\n        '
        rewrittenQuery = self._extract(_PartitionedQueryExecutionInfo.RewrittenQueryPath)
        if rewrittenQuery is not None:
            rewrittenQuery = rewrittenQuery.replace('{documentdb-formattableorderbyquery-filter}', 'true')
        return rewrittenQuery

    def has_select_value(self):
        if False:
            print('Hello World!')
        return self._extract(self.HasSelectValue)

    def has_top(self):
        if False:
            while True:
                i = 10
        return self.get_top() is not None

    def has_limit(self):
        if False:
            print('Hello World!')
        return self.get_limit() is not None

    def has_offset(self):
        if False:
            while True:
                i = 10
        return self.get_offset() is not None

    def has_distinct_type(self):
        if False:
            return 10
        return self.get_distinct_type() != _DistinctType.NoneType

    def has_order_by(self):
        if False:
            for i in range(10):
                print('nop')
        order_by = self.get_order_by()
        return order_by is not None and len(order_by) > 0

    def has_aggregates(self):
        if False:
            for i in range(10):
                print('nop')
        aggregates = self.get_aggregates()
        return aggregates is not None and len(aggregates) > 0

    def has_rewritten_query(self):
        if False:
            while True:
                i = 10
        return self.get_rewritten_query() is not None

    def _extract(self, path):
        if False:
            while True:
                i = 10
        item = self._query_execution_info
        if isinstance(path, str):
            return item.get(path)
        for p in path:
            item = item.get(p)
            if item is None:
                return None
        return item
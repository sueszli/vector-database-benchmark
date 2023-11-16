"""Internal class for query execution endpoint component implementation in the
Azure Cosmos database service.
"""
import numbers
import copy
import hashlib
import json
from azure.cosmos._execution_context.aggregators import _AverageAggregator, _CountAggregator, _MaxAggregator, _MinAggregator, _SumAggregator

class _QueryExecutionEndpointComponent(object):

    def __init__(self, execution_context):
        if False:
            print('Hello World!')
        self._execution_context = execution_context

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        return next(self._execution_context)
    next = __next__

class _QueryExecutionOrderByEndpointComponent(_QueryExecutionEndpointComponent):
    """Represents an endpoint in handling an order by query.

    For each processed orderby result it returns 'payload' item of the result.
    """

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        return next(self._execution_context)['payload']
    next = __next__

class _QueryExecutionTopEndpointComponent(_QueryExecutionEndpointComponent):
    """Represents an endpoint in handling top query.

    It only returns as many results as top arg specified.
    """

    def __init__(self, execution_context, top_count):
        if False:
            while True:
                i = 10
        super(_QueryExecutionTopEndpointComponent, self).__init__(execution_context)
        self._top_count = top_count

    def __next__(self):
        if False:
            i = 10
            return i + 15
        if self._top_count > 0:
            res = next(self._execution_context)
            self._top_count -= 1
            return res
        raise StopIteration
    next = __next__

class _QueryExecutionDistinctOrderedEndpointComponent(_QueryExecutionEndpointComponent):
    """Represents an endpoint in handling distinct query.

    It returns only those values not already returned.
    """

    def __init__(self, execution_context):
        if False:
            return 10
        super(_QueryExecutionDistinctOrderedEndpointComponent, self).__init__(execution_context)
        self.last_result = None

    def __next__(self):
        if False:
            while True:
                i = 10
        res = next(self._execution_context)
        while self.last_result == res:
            res = next(self._execution_context)
        self.last_result = res
        return res
    next = __next__

class _QueryExecutionDistinctUnorderedEndpointComponent(_QueryExecutionEndpointComponent):
    """Represents an endpoint in handling distinct query.

    It returns only those values not already returned.
    """

    def __init__(self, execution_context):
        if False:
            print('Hello World!')
        super(_QueryExecutionDistinctUnorderedEndpointComponent, self).__init__(execution_context)
        self.last_result = set()

    def make_hash(self, value):
        if False:
            i = 10
            return i + 15
        if isinstance(value, (set, tuple, list)):
            return tuple([self.make_hash(v) for v in value])
        if not isinstance(value, dict):
            if isinstance(value, numbers.Number):
                return float(value)
            return value
        new_value = copy.deepcopy(value)
        for (k, v) in new_value.items():
            new_value[k] = self.make_hash(v)
        return tuple(frozenset(sorted(new_value.items())))

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        res = next(self._execution_context)
        json_repr = json.dumps(self.make_hash(res))
        json_repr = json_repr.encode('utf-8')
        hash_object = hashlib.sha1(json_repr)
        hashed_result = hash_object.hexdigest()
        while hashed_result in self.last_result:
            res = next(self._execution_context)
            json_repr = json.dumps(self.make_hash(res))
            json_repr = json_repr.encode('utf-8')
            hash_object = hashlib.sha1(json_repr)
            hashed_result = hash_object.hexdigest()
        self.last_result.add(hashed_result)
        return res
    next = __next__

class _QueryExecutionOffsetEndpointComponent(_QueryExecutionEndpointComponent):
    """Represents an endpoint in handling offset query.

    It returns results offset by as many results as offset arg specified.
    """

    def __init__(self, execution_context, offset_count):
        if False:
            for i in range(10):
                print('nop')
        super(_QueryExecutionOffsetEndpointComponent, self).__init__(execution_context)
        self._offset_count = offset_count

    def __next__(self):
        if False:
            while True:
                i = 10
        while self._offset_count > 0:
            res = next(self._execution_context)
            if res is not None:
                self._offset_count -= 1
            else:
                raise StopIteration
        return next(self._execution_context)
    next = __next__

class _QueryExecutionAggregateEndpointComponent(_QueryExecutionEndpointComponent):
    """Represents an endpoint in handling aggregate query.

    It returns only aggregated values.
    """

    def __init__(self, execution_context, aggregate_operators):
        if False:
            for i in range(10):
                print('nop')
        super(_QueryExecutionAggregateEndpointComponent, self).__init__(execution_context)
        self._local_aggregators = []
        self._results = None
        self._result_index = 0
        for operator in aggregate_operators:
            if operator == 'Average':
                self._local_aggregators.append(_AverageAggregator())
            elif operator == 'Count':
                self._local_aggregators.append(_CountAggregator())
            elif operator == 'Max':
                self._local_aggregators.append(_MaxAggregator())
            elif operator == 'Min':
                self._local_aggregators.append(_MinAggregator())
            elif operator == 'Sum':
                self._local_aggregators.append(_SumAggregator())

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        for res in self._execution_context:
            for item in res:
                for operator in self._local_aggregators:
                    if isinstance(item, dict) and item:
                        try:
                            operator.aggregate(item['item'])
                        except KeyError:
                            pass
                    elif isinstance(item, numbers.Number):
                        operator.aggregate(item)
        if self._results is None:
            self._results = []
            for operator in self._local_aggregators:
                self._results.append(operator.get_result())
        if self._result_index < len(self._results):
            res = self._results[self._result_index]
            self._result_index += 1
            return res
        raise StopIteration
    next = __next__
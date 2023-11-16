"""
Implements a Cloud Datastore query splitter.

For internal use only. No backwards compatibility guarantees.
"""
from apache_beam.io.gcp.datastore.v1new import types
from apache_beam.options.value_provider import ValueProvider
__all__ = ['QuerySplitterError', 'SplitNotPossibleError', 'get_splits']
SCATTER_PROPERTY_NAME = '__scatter__'
KEY_PROPERTY_NAME = '__key__'
KEYS_PER_SPLIT = 32

class QuerySplitterError(Exception):
    """Top-level error type."""

class SplitNotPossibleError(QuerySplitterError):
    """Raised when some parameter of the query does not allow splitting."""

def get_splits(client, query, num_splits):
    if False:
        while True:
            i = 10
    'Returns a list of sharded queries for the given Cloud Datastore query.\n\n  This will create up to the desired number of splits, however it may return\n  less splits if the desired number of splits is unavailable. This will happen\n  if the number of split points provided by the underlying Datastore is less\n  than the desired number, which will occur if the number of results for the\n  query is too small.\n\n  This implementation of the QuerySplitter uses the __scatter__ property to\n  gather random split points for a query.\n\n  Note: This implementation is derived from the java query splitter in\n  https://github.com/GoogleCloudPlatform/google-cloud-datastore/blob/master/java/datastore/src/main/java/com/google/datastore/v1/client/QuerySplitterImpl.java\n\n  Args:\n    client: the datastore client.\n    query: the query to split.\n    num_splits: the desired number of splits.\n\n  Returns:\n    A list of split queries, of a max length of `num_splits`\n\n  Raises:\n    QuerySplitterError: if split could not be performed owing to query or split\n      parameters.\n  '
    if num_splits <= 1:
        raise SplitNotPossibleError('num_splits must be > 1, got: %d' % num_splits)
    validate_split(query)
    splits = []
    client_scatter_keys = _get_scatter_keys(client, query, num_splits)
    last_client_key = None
    for next_client_key in _get_split_key(client_scatter_keys, num_splits):
        splits.append(_create_split(last_client_key, next_client_key, query))
        last_client_key = next_client_key
    splits.append(_create_split(last_client_key, None, query))
    return splits

def validate_split(query):
    if False:
        for i in range(10):
            print('nop')
    '\n  Verifies that the given query can be properly scattered.\n\n  Note that equality and ancestor filters are allowed, however they may result\n  in inefficient sharding.\n\n  Raises:\n    QuerySplitterError if split could not be performed owing to query\n      parameters.\n  '
    if query.order:
        raise SplitNotPossibleError('Query cannot have any sort orders.')
    if query.limit is not None:
        raise SplitNotPossibleError('Query cannot have a limit set.')
    for filter in query.filters:
        if isinstance(filter[1], ValueProvider):
            filter_operator = filter[1].get()
        else:
            filter_operator = filter[1]
        if filter_operator in ['<', '<=', '>', '>=']:
            raise SplitNotPossibleError('Query cannot have any inequality filters.')

def _create_scatter_query(query, num_splits):
    if False:
        while True:
            i = 10
    'Creates a scatter query from the given user query.'
    limit = (num_splits - 1) * KEYS_PER_SPLIT
    scatter_query = types.Query(kind=query.kind, project=query.project, namespace=query.namespace, order=[SCATTER_PROPERTY_NAME], projection=[KEY_PROPERTY_NAME], limit=limit)
    return scatter_query

class IdOrName(object):
    """Represents an ID or name of a Datastore key,

   Implements sort ordering: by ID, then by name, keys with IDs before those
   with names.
   """

    def __init__(self, id_or_name):
        if False:
            while True:
                i = 10
        self.id_or_name = id_or_name
        if isinstance(id_or_name, str):
            self.id = None
            self.name = id_or_name
        elif isinstance(id_or_name, int):
            self.id = id_or_name
            self.name = None
        else:
            raise TypeError('Unexpected type of id_or_name: %s' % id_or_name)

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, IdOrName):
            return super().__lt__(other)
        if self.id is not None:
            if other.id is None:
                return True
            else:
                return self.id < other.id
        if other.id is not None:
            return False
        return self.name < other.name

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, IdOrName):
            return super().__eq__(other)
        return self.id == other.id and self.name == other.name

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.id, self.other))

def client_key_sort_key(client_key):
    if False:
        print('Hello World!')
    'Key function for sorting lists of ``google.cloud.datastore.key.Key``.'
    sort_key = [client_key.project, client_key.namespace or '']
    flat_path = list(client_key.flat_path)
    while flat_path:
        sort_key.append(flat_path.pop(0))
        if flat_path:
            sort_key.append(IdOrName(flat_path.pop(0)))
    return sort_key

def _get_scatter_keys(client, query, num_splits):
    if False:
        return 10
    'Gets a list of split keys given a desired number of splits.\n\n  This list will contain multiple split keys for each split. Only a single split\n  key will be chosen as the split point, however providing multiple keys allows\n  for more uniform sharding.\n\n  Args:\n    client: the client to datastore containing the data.\n    query: the user query.\n    num_splits: the number of desired splits.\n\n  Returns:\n    A list of scatter keys returned by Datastore.\n  '
    scatter_point_query = _create_scatter_query(query, num_splits)
    client_query = scatter_point_query._to_client_query(client)
    client_key_splits = [client_entity.key for client_entity in client_query.fetch(client=client, limit=scatter_point_query.limit)]
    client_key_splits.sort(key=client_key_sort_key)
    return client_key_splits

def _get_split_key(client_keys, num_splits):
    if False:
        print('Hello World!')
    'Given a list of keys and a number of splits find the keys to split on.\n\n  Args:\n    client_keys: the list of keys.\n    num_splits: the number of splits.\n\n  Returns:\n    A list of keys to split on.\n\n  '
    if not client_keys or len(client_keys) < num_splits - 1:
        return client_keys
    num_keys_per_split = max(1.0, float(len(client_keys)) / (num_splits - 1))
    split_client_keys = []
    for i in range(1, num_splits):
        split_index = int(round(i * num_keys_per_split) - 1)
        split_client_keys.append(client_keys[split_index])
    return split_client_keys

def _create_split(last_client_key, next_client_key, query):
    if False:
        for i in range(10):
            print('nop')
    'Create a new {@link Query} given the query and range.\n\n  Args:\n    last_client_key: the previous key. If null then assumed to be the beginning.\n    next_client_key: the next key. If null then assumed to be the end.\n    query: query to base the split query on.\n\n  Returns:\n    A split query with fetches entities in the range [last_key, next_client_key)\n  '
    if not (last_client_key or next_client_key):
        return query
    split_query = query.clone()
    filters = list(split_query.filters)
    if last_client_key:
        filters.append((KEY_PROPERTY_NAME, '>=', last_client_key))
    if next_client_key:
        filters.append((KEY_PROPERTY_NAME, '<', next_client_key))
    split_query.filters = filters
    return split_query
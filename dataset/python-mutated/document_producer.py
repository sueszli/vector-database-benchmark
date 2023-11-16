"""Internal class for document producer implementation in the Azure Cosmos
database service.
"""
import numbers
from collections import deque
from azure.cosmos import _base
from azure.cosmos._execution_context.aio.base_execution_context import _DefaultQueryExecutionContext

class _DocumentProducer(object):
    """This class takes care of handling of the results for one single partition
    key range.

    When handling an orderby query, MultiExecutionContextAggregator instantiates
    one instance of this class per target partition key range and aggregates the
    result of each.
    """

    def __init__(self, partition_key_target_range, client, collection_link, query, document_producer_comp, options):
        if False:
            return 10
        '\n        Constructor\n        '
        self._options = options
        self._partition_key_target_range = partition_key_target_range
        self._doc_producer_comp = document_producer_comp
        self._client = client
        self._buffer = deque()
        self._is_finished = False
        self._has_started = False
        self._cur_item = None
        path = _base.GetPathFromLink(collection_link, 'docs')
        collection_id = _base.GetResourceIdOrFullNameFromLink(collection_link)

        async def fetch_fn(options):
            return await self._client.QueryFeed(path, collection_id, query, options, partition_key_target_range['id'])
        self._ex_context = _DefaultQueryExecutionContext(client, self._options, fetch_fn)

    def __aiter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    async def __anext__(self):
        """
        :return: The next result item.
        :rtype: dict
        :raises StopIteration: If there is no more result.

        """
        if self._cur_item is not None:
            res = self._cur_item
            self._cur_item = None
            return res
        return await self._ex_context.__anext__()

    def get_target_range(self):
        if False:
            print('Hello World!')
        'Returns the target partition key range.\n            :return:\n                Target partition key range.\n            :rtype: dict\n        '
        return self._partition_key_target_range

    async def peek(self):
        """
        TODO: use more_itertools.peekable instead
        :return: The current result item.
        :rtype: dict.
        :raises StopIteration: If there is no current item.

        """
        if self._cur_item is None:
            self._cur_item = await self._ex_context.__anext__()
        return self._cur_item

def _compare_helper(a, b):
    if False:
        return 10
    if a is None and b is None:
        return 0
    return (a > b) - (a < b)

class _PartitionKeyRangeDocumentProducerComparator(object):
    """
    Provides a Comparator for document producers using the min value of the
    corresponding target partition.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    async def compare(self, doc_producer1, doc_producer2):
        return _compare_helper(doc_producer1.get_target_range()['minInclusive'], doc_producer2.get_target_range()['minInclusive'])

class _OrderByHelper(object):

    @staticmethod
    def getTypeOrd(orderby_item):
        if False:
            print('Hello World!')
        "Returns the ordinal of the value of the item pair in the dictionary.\n\n        :param dict orderby_item:\n\n        :return:\n            0 if the item_pair doesn't have any 'item' key\n            1 if the value is undefined\n            2 if the value is a boolean\n            4 if the value is a number\n            5 if the value is a str or a unicode\n        :rtype: int\n        "
        if 'item' not in orderby_item:
            return 0
        val = orderby_item['item']
        if val is None:
            return 1
        if isinstance(val, bool):
            return 2
        if isinstance(val, numbers.Number):
            return 4
        if isinstance(val, str):
            return 5
        raise TypeError('unknown type' + str(val))

    @staticmethod
    def getTypeStr(orderby_item):
        if False:
            i = 10
            return i + 15
        'Returns the string representation of the type\n\n        :param dict orderby_item:\n        :return: String representation of the type\n        :rtype: str\n        '
        if 'item' not in orderby_item:
            return 'NoValue'
        val = orderby_item['item']
        if val is None:
            return 'Null'
        if isinstance(val, bool):
            return 'Boolean'
        if isinstance(val, numbers.Number):
            return 'Number'
        if isinstance(val, str):
            return 'String'
        raise TypeError('unknown type' + str(val))

    @staticmethod
    async def compare(orderby_item1, orderby_item2):
        """Compare two orderby item pairs.

        :param dict orderby_item1:
        :param dict orderby_item2:
        :return: Integer comparison result.
            The comparator acts such that
            - if the types are different we get:
                Undefined value < Null < booleans < Numbers < Strings
            - if both arguments are of the same type:
                it simply compares the values.
        :rtype: int
        """
        type1_ord = _OrderByHelper.getTypeOrd(orderby_item1)
        type2_ord = _OrderByHelper.getTypeOrd(orderby_item2)
        type_ord_diff = type1_ord - type2_ord
        if type_ord_diff:
            return type_ord_diff
        if type1_ord == 0:
            return 0
        return _compare_helper(orderby_item1['item'], orderby_item2['item'])

def _peek_order_by_items(peek_result):
    if False:
        i = 10
        return i + 15
    return peek_result['orderByItems']

class _OrderByDocumentProducerComparator(_PartitionKeyRangeDocumentProducerComparator):
    """Provide a Comparator for document producers which respects orderby sort order.
    """

    def __init__(self, sort_order):
        if False:
            print('Hello World!')
        'Instantiates this class\n\n        :param list sort_order:\n            List of sort orders (i.e., Ascending, Descending)\n\n        :ivar list sort_order:\n            List of sort orders (i.e., Ascending, Descending)\n\n        '
        self._sort_order = sort_order

    async def compare(self, doc_producer1, doc_producer2):
        """Compares the given two instances of DocumentProducers.

        Based on the orderby query items and whether the sort order is Ascending
        or Descending compares the peek result of the two DocumentProducers.

        If the peek results are equal based on the sort order, this comparator
        compares the target partition key range of the two DocumentProducers.

        :param _DocumentProducer doc_producer1: first instance to be compared
        :param _DocumentProducer doc_producer2: second instance to be compared
        :return:
            Integer value of compare result.
                positive integer if doc_producer1 > doc_producer2
                negative integer if doc_producer1 < doc_producer2
        :rtype: int
        """
        res1 = _peek_order_by_items(await doc_producer1.peek())
        res2 = _peek_order_by_items(await doc_producer2.peek())
        self._validate_orderby_items(res1, res2)
        for (i, (elt1, elt2)) in enumerate(zip(res1, res2)):
            res = await _OrderByHelper.compare(elt1, elt2)
            if res != 0:
                if self._sort_order[i] == 'Ascending':
                    return res
                if self._sort_order[i] == 'Descending':
                    return -res
        return await _PartitionKeyRangeDocumentProducerComparator.compare(self, doc_producer1, doc_producer2)

    def _validate_orderby_items(self, res1, res2):
        if False:
            i = 10
            return i + 15
        if len(res1) != len(res2):
            raise ValueError('orderByItems cannot have different size')
        if len(res1) != len(self._sort_order):
            raise ValueError('orderByItems cannot have a different size than sort orders.')
        for (elt1, elt2) in zip(res1, res2):
            type1 = _OrderByHelper.getTypeStr(elt1)
            type2 = _OrderByHelper.getTypeStr(elt2)
            if type1 != type2:
                raise ValueError('Expected {}, but got {}.'.format(type1, type2))
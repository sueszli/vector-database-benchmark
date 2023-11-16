"""Internal class for partition key range cache implementation in the Azure
Cosmos database service.
"""
from .. import _base
from .collection_routing_map import CollectionRoutingMap
from . import routing_range
from .routing_range import PartitionKeyRange

class PartitionKeyRangeCache(object):
    """
    PartitionKeyRangeCache provides list of effective partition key ranges for a
    collection.

    This implementation loads and caches the collection routing map per
    collection on demand.
    """

    def __init__(self, client):
        if False:
            i = 10
            return i + 15
        '\n        Constructor\n        '
        self._documentClient = client
        self._collection_routing_map_by_item = {}

    def get_overlapping_ranges(self, collection_link, partition_key_ranges):
        if False:
            while True:
                i = 10
        'Given a partition key range and a collection, return the list of\n        overlapping partition key ranges.\n\n        :param str collection_link: The name of the collection.\n        :param list partition_key_ranges: List of partition key range.\n        :return: List of overlapping partition key ranges.\n        :rtype: list\n        '
        cl = self._documentClient
        collection_id = _base.GetResourceIdOrFullNameFromLink(collection_link)
        collection_routing_map = self._collection_routing_map_by_item.get(collection_id)
        if collection_routing_map is None:
            collection_pk_ranges = list(cl._ReadPartitionKeyRanges(collection_link))
            collection_pk_ranges = PartitionKeyRangeCache._discard_parent_ranges(collection_pk_ranges)
            collection_routing_map = CollectionRoutingMap.CompleteRoutingMap([(r, True) for r in collection_pk_ranges], collection_id)
            self._collection_routing_map_by_item[collection_id] = collection_routing_map
        return collection_routing_map.get_overlapping_ranges(partition_key_ranges)

    @staticmethod
    def _discard_parent_ranges(partitionKeyRanges):
        if False:
            print('Hello World!')
        parentIds = set()
        for r in partitionKeyRanges:
            if isinstance(r, dict) and PartitionKeyRange.Parents in r:
                for parentId in r[PartitionKeyRange.Parents]:
                    parentIds.add(parentId)
        return (r for r in partitionKeyRanges if r[PartitionKeyRange.Id] not in parentIds)

def _second_range_is_after_first_range(range1, range2):
    if False:
        for i in range(10):
            print('nop')
    if range1.max > range2.min:
        return False
    if range2.min == range1.max and range1.isMaxInclusive and range2.isMinInclusive:
        return False
    return True

def _is_sorted_and_non_overlapping(ranges):
    if False:
        for i in range(10):
            print('nop')
    for (idx, r) in list(enumerate(ranges))[1:]:
        previous_r = ranges[idx - 1]
        if not _second_range_is_after_first_range(previous_r, r):
            return False
    return True

def _subtract_range(r, partition_key_range):
    if False:
        i = 10
        return i + 15
    'Evaluates and returns r - partition_key_range\n\n    :param dict partition_key_range: Partition key range.\n    :param routing_range.Range r: query range.\n    :return: The subtract r - partition_key_range.\n    :rtype: routing_range.Range\n    '
    left = max(partition_key_range[routing_range.PartitionKeyRange.MaxExclusive], r.min)
    if left == r.min:
        leftInclusive = r.isMinInclusive
    else:
        leftInclusive = False
    queryRange = routing_range.Range(left, r.max, leftInclusive, r.isMaxInclusive)
    return queryRange

class SmartRoutingMapProvider(PartitionKeyRangeCache):
    """
    Efficiently uses PartitionKeyRangeCache and minimizes the unnecessary
    invocation of CollectionRoutingMap.get_overlapping_ranges()
    """

    def get_overlapping_ranges(self, collection_link, partition_key_ranges):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given the sorted ranges and a collection,\n        Returns the list of overlapping partition key ranges\n\n        :param str collection_link: The collection link.\n        :param (list of routing_range.Range) partition_key_ranges:\n            The sorted list of non-overlapping ranges.\n        :return: List of partition key ranges.\n        :rtype: list of dict\n        :raises ValueError:\n            If two ranges in partition_key_ranges overlap or if the list is not sorted\n        '
        if not _is_sorted_and_non_overlapping(partition_key_ranges):
            raise ValueError('the list of ranges is not a non-overlapping sorted ranges')
        target_partition_key_ranges = []
        it = iter(partition_key_ranges)
        try:
            currentProvidedRange = next(it)
            while True:
                if currentProvidedRange.isEmpty():
                    currentProvidedRange = next(it)
                    continue
                if target_partition_key_ranges:
                    queryRange = _subtract_range(currentProvidedRange, target_partition_key_ranges[-1])
                else:
                    queryRange = currentProvidedRange
                overlappingRanges = PartitionKeyRangeCache.get_overlapping_ranges(self, collection_link, queryRange)
                assert overlappingRanges, 'code bug: returned overlapping ranges for queryRange {} is empty'.format(queryRange)
                target_partition_key_ranges.extend(overlappingRanges)
                lastKnownTargetRange = routing_range.Range.PartitionKeyRangeToRange(target_partition_key_ranges[-1])
                assert currentProvidedRange.max <= lastKnownTargetRange.max, 'code bug: returned overlapping ranges {} does not contain the requested range {}'.format(overlappingRanges, queryRange)
                currentProvidedRange = next(it)
                while currentProvidedRange.max <= lastKnownTargetRange.max:
                    currentProvidedRange = next(it)
        except StopIteration:
            pass
        return target_partition_key_ranges
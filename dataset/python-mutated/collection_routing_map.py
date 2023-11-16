"""Internal class for collection routing map implementation in the Azure Cosmos
database service.
"""
import bisect
from azure.cosmos._routing import routing_range
from azure.cosmos._routing.routing_range import PartitionKeyRange

class CollectionRoutingMap(object):
    """Stores partition key ranges in an efficient way with some additional
    information and provides convenience methods for working with set of ranges.
    """
    MinimumInclusiveEffectivePartitionKey = ''
    MaximumExclusiveEffectivePartitionKey = 'FF'

    def __init__(self, range_by_id, range_by_info, ordered_partition_key_ranges, ordered_partition_info, collection_unique_id):
        if False:
            print('Hello World!')
        self._rangeById = range_by_id
        self._rangeByInfo = range_by_info
        self._orderedPartitionKeyRanges = ordered_partition_key_ranges
        self._orderedRanges = [routing_range.Range(pkr[PartitionKeyRange.MinInclusive], pkr[PartitionKeyRange.MaxExclusive], True, False) for pkr in ordered_partition_key_ranges]
        self._orderedPartitionInfo = ordered_partition_info
        self._collectionUniqueId = collection_unique_id

    @classmethod
    def CompleteRoutingMap(cls, partition_key_range_info_tuple_list, collection_unique_id):
        if False:
            while True:
                i = 10
        rangeById = {}
        rangeByInfo = {}
        sortedRanges = []
        for r in partition_key_range_info_tuple_list:
            rangeById[r[0][PartitionKeyRange.Id]] = r
            rangeByInfo[r[1]] = r[0]
            sortedRanges.append(r)
        sortedRanges.sort(key=lambda r: r[0][PartitionKeyRange.MinInclusive])
        partitionKeyOrderedRange = [r[0] for r in sortedRanges]
        orderedPartitionInfo = [r[1] for r in sortedRanges]
        if not CollectionRoutingMap.is_complete_set_of_range(partitionKeyOrderedRange):
            return None
        return cls(rangeById, rangeByInfo, partitionKeyOrderedRange, orderedPartitionInfo, collection_unique_id)

    def get_ordered_partition_key_ranges(self):
        if False:
            i = 10
            return i + 15
        'Gets the ordered partition key ranges\n\n        :return: Ordered list of partition key ranges.\n        :rtype: list\n        '
        return self._orderedPartitionKeyRanges

    def get_range_by_effective_partition_key(self, effective_partition_key_value):
        if False:
            for i in range(10):
                print('nop')
        'Gets the range containing the given partition key\n\n        :param str effective_partition_key_value: The partition key value.\n        :return: The partition key range.\n        :rtype: dict\n        '
        if CollectionRoutingMap.MinimumInclusiveEffectivePartitionKey == effective_partition_key_value:
            return self._orderedPartitionKeyRanges[0]
        if CollectionRoutingMap.MaximumExclusiveEffectivePartitionKey == effective_partition_key_value:
            return None
        sortedLow = [(r.min, not r.isMinInclusive) for r in self._orderedRanges]
        index = bisect.bisect_right(sortedLow, (effective_partition_key_value, True))
        if index > 0:
            index = index - 1
        return self._orderedPartitionKeyRanges[index]

    def get_range_by_partition_key_range_id(self, partition_key_range_id):
        if False:
            for i in range(10):
                print('nop')
        'Gets the partition key range given the partition key range id\n\n        :param str partition_key_range_id: The partition key range id.\n        :return: The partition key range.\n        :rtype: dict\n        '
        t = self._rangeById.get(partition_key_range_id)
        if t is None:
            return None
        return t[0]

    def get_overlapping_ranges(self, provided_partition_key_ranges):
        if False:
            for i in range(10):
                print('nop')
        'Gets the partition key ranges overlapping the provided ranges\n\n        :param list provided_partition_key_ranges: List of partition key ranges.\n        :return: List of partition key ranges, where each is a dict.\n        :rtype: list\n        '
        if isinstance(provided_partition_key_ranges, routing_range.Range):
            return self.get_overlapping_ranges([provided_partition_key_ranges])
        minToPartitionRange = {}
        sortedLow = [(r.min, not r.isMinInclusive) for r in self._orderedRanges]
        sortedHigh = [(r.max, r.isMaxInclusive) for r in self._orderedRanges]
        for providedRange in provided_partition_key_ranges:
            minIndex = bisect.bisect_right(sortedLow, (providedRange.min, not providedRange.isMinInclusive))
            if minIndex > 0:
                minIndex = minIndex - 1
            maxIndex = bisect.bisect_left(sortedHigh, (providedRange.max, providedRange.isMaxInclusive))
            if maxIndex >= len(sortedHigh):
                maxIndex = maxIndex - 1
            for i in range(minIndex, maxIndex + 1):
                if routing_range.Range.overlaps(self._orderedRanges[i], providedRange):
                    minToPartitionRange[self._orderedPartitionKeyRanges[i][PartitionKeyRange.MinInclusive]] = self._orderedPartitionKeyRanges[i]
        overlapping_partition_key_ranges = list(minToPartitionRange.values())

        def getKey(r):
            if False:
                for i in range(10):
                    print('nop')
            return r[PartitionKeyRange.MinInclusive]
        overlapping_partition_key_ranges.sort(key=getKey)
        return overlapping_partition_key_ranges

    @staticmethod
    def is_complete_set_of_range(ordered_partition_key_range_list):
        if False:
            print('Hello World!')
        isComplete = False
        if ordered_partition_key_range_list:
            firstRange = ordered_partition_key_range_list[0]
            lastRange = ordered_partition_key_range_list[-1]
            isComplete = firstRange[PartitionKeyRange.MinInclusive] == CollectionRoutingMap.MinimumInclusiveEffectivePartitionKey
            isComplete &= lastRange[PartitionKeyRange.MaxExclusive] == CollectionRoutingMap.MaximumExclusiveEffectivePartitionKey
            for i in range(1, len(ordered_partition_key_range_list)):
                previousRange = ordered_partition_key_range_list[i - 1]
                currentRange = ordered_partition_key_range_list[i]
                isComplete &= previousRange[PartitionKeyRange.MaxExclusive] == currentRange[PartitionKeyRange.MinInclusive]
                if not isComplete:
                    if previousRange[PartitionKeyRange.MaxExclusive] > currentRange[PartitionKeyRange.MinInclusive]:
                        raise ValueError('Ranges overlap')
                    break
        return isComplete
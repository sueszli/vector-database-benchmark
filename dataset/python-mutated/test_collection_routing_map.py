import unittest
import pytest
from azure.cosmos._routing.collection_routing_map import CollectionRoutingMap
import azure.cosmos._routing.routing_range as routing_range
from azure.cosmos._routing.routing_map_provider import PartitionKeyRangeCache
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class CollectionRoutingMapTests(unittest.TestCase):

    def test_advanced(self):
        if False:
            for i in range(10):
                print('nop')
        partition_key_ranges = [{u'id': u'0', u'minInclusive': u'', u'maxExclusive': u'05C1C9CD673398'}, {u'id': u'1', u'minInclusive': u'05C1C9CD673398', u'maxExclusive': u'05C1D9CD673398'}, {u'id': u'2', u'minInclusive': u'05C1D9CD673398', u'maxExclusive': u'05C1E399CD6732'}, {u'id': u'3', u'minInclusive': u'05C1E399CD6732', u'maxExclusive': u'05C1E9CD673398'}, {u'id': u'4', u'minInclusive': u'05C1E9CD673398', u'maxExclusive': u'FF'}]
        partitionRangeWithInfo = [(r, True) for r in partition_key_ranges]
        pkRange = routing_range.Range('', 'FF', True, False)
        collection_routing_map = CollectionRoutingMap.CompleteRoutingMap(partitionRangeWithInfo, 'sample collection id')
        overlapping_partition_key_ranges = collection_routing_map.get_overlapping_ranges(pkRange)
        self.assertEqual(len(overlapping_partition_key_ranges), len(partition_key_ranges))
        self.assertEqual(overlapping_partition_key_ranges, partition_key_ranges)

    def test_partition_key_ranges_parent_filter(self):
        if False:
            print('Hello World!')
        Id = 'id'
        MinInclusive = 'minInclusive'
        MaxExclusive = 'maxExclusive'
        Parents = 'parents'
        partitionKeyRanges = [{Id: '2', MinInclusive: '0000000050', MaxExclusive: '0000000070', Parents: []}, {Id: '0', MinInclusive: '', MaxExclusive: '0000000030'}, {Id: '1', MinInclusive: '0000000030', MaxExclusive: '0000000050'}, {Id: '3', MinInclusive: '0000000070', MaxExclusive: 'FF', Parents: []}]

        def get_range_id(r):
            if False:
                i = 10
                return i + 15
            return r[Id]
        filteredRanges = PartitionKeyRangeCache._discard_parent_ranges(partitionKeyRanges)
        self.assertEqual(['2', '0', '1', '3'], list(map(get_range_id, filteredRanges)))
        partitionKeyRanges.append({Id: '6', MinInclusive: '', MaxExclusive: '0000000010', Parents: ['0', '4']})
        partitionKeyRanges.append({Id: '7', MinInclusive: '0000000010', MaxExclusive: '0000000020', Parents: ['0', '4']})
        partitionKeyRanges.append({Id: '5', MinInclusive: '0000000020', MaxExclusive: '0000000030', Parents: ['0']})
        filteredRanges = PartitionKeyRangeCache._discard_parent_ranges(partitionKeyRanges)
        expectedRanges = ['2', '1', '3', '6', '7', '5']
        self.assertEqual(expectedRanges, list(map(get_range_id, filteredRanges)))

    def test_collection_routing_map(self):
        if False:
            while True:
                i = 10
        Id = 'id'
        MinInclusive = 'minInclusive'
        MaxExclusive = 'maxExclusive'
        partitionKeyRanges = [({Id: '2', MinInclusive: '0000000050', MaxExclusive: '0000000070'}, 2), ({Id: '0', MinInclusive: '', MaxExclusive: '0000000030'}, 0), ({Id: '1', MinInclusive: '0000000030', MaxExclusive: '0000000050'}, 1), ({Id: '3', MinInclusive: '0000000070', MaxExclusive: 'FF'}, 3)]
        crm = CollectionRoutingMap.CompleteRoutingMap(partitionKeyRanges, '')
        self.assertEqual('0', crm._orderedPartitionKeyRanges[0][Id])
        self.assertEqual('1', crm._orderedPartitionKeyRanges[1][Id])
        self.assertEqual('2', crm._orderedPartitionKeyRanges[2][Id])
        self.assertEqual('3', crm._orderedPartitionKeyRanges[3][Id])
        self.assertEqual(0, crm._orderedPartitionInfo[0])
        self.assertEqual(1, crm._orderedPartitionInfo[1])
        self.assertEqual(2, crm._orderedPartitionInfo[2])
        self.assertEqual(3, crm._orderedPartitionInfo[3])
        self.assertEqual('0', crm.get_range_by_effective_partition_key('')[Id])
        self.assertEqual('0', crm.get_range_by_effective_partition_key('0000000000')[Id])
        self.assertEqual('1', crm.get_range_by_effective_partition_key('0000000030')[Id])
        self.assertEqual('1', crm.get_range_by_effective_partition_key('0000000031')[Id])
        self.assertEqual('3', crm.get_range_by_effective_partition_key('0000000071')[Id])
        self.assertEqual('0', crm.get_range_by_partition_key_range_id('0')[Id])
        self.assertEqual('1', crm.get_range_by_partition_key_range_id('1')[Id])
        fullRangeMinToMaxRange = routing_range.Range(CollectionRoutingMap.MinimumInclusiveEffectivePartitionKey, CollectionRoutingMap.MaximumExclusiveEffectivePartitionKey, True, False)
        overlappingRanges = crm.get_overlapping_ranges([fullRangeMinToMaxRange])
        self.assertEqual(4, len(overlappingRanges))
        onlyPartitionRanges = [item[0] for item in partitionKeyRanges]

        def getKey(r):
            if False:
                i = 10
                return i + 15
            return r['id']
        onlyPartitionRanges.sort(key=getKey)
        self.assertEqual(overlappingRanges, onlyPartitionRanges)
        noPoint = routing_range.Range(CollectionRoutingMap.MinimumInclusiveEffectivePartitionKey, CollectionRoutingMap.MinimumInclusiveEffectivePartitionKey, False, False)
        self.assertEqual(0, len(crm.get_overlapping_ranges([noPoint])))
        onePoint = routing_range.Range('0000000040', '0000000040', True, True)
        overlappingPartitionKeyRanges = crm.get_overlapping_ranges([onePoint])
        self.assertEqual(1, len(overlappingPartitionKeyRanges))
        self.assertEqual('1', overlappingPartitionKeyRanges[0][Id])
        ranges = [routing_range.Range('0000000040', '0000000045', True, True), routing_range.Range('0000000045', '0000000046', True, True), routing_range.Range('0000000046', '0000000050', True, True)]
        overlappingPartitionKeyRanges = crm.get_overlapping_ranges(ranges)
        self.assertEqual(2, len(overlappingPartitionKeyRanges))
        self.assertEqual('1', overlappingPartitionKeyRanges[0][Id])
        self.assertEqual('2', overlappingPartitionKeyRanges[1][Id])

    def test_invalid_routing_map(self):
        if False:
            return 10
        partitionKeyRanges = [({'id': '1', 'minInclusive': '0000000020', 'maxExclusive': '0000000030'}, 2), ({'id': '2', 'minInclusive': '0000000025', 'maxExclusive': '0000000035'}, 2)]
        collectionUniqueId = ''

        def createRoutingMap():
            if False:
                while True:
                    i = 10
            CollectionRoutingMap.CompleteRoutingMap(partitionKeyRanges, collectionUniqueId)
        self.assertRaises(ValueError, createRoutingMap)

    def test_incomplete_routing_map(self):
        if False:
            return 10
        crm = CollectionRoutingMap.CompleteRoutingMap([({'id': '2', 'minInclusive': '', 'maxExclusive': '0000000030'}, 2), ({'id': '3', 'minInclusive': '0000000031', 'maxExclusive': 'FF'}, 2)], '')
        self.assertIsNone(crm)
        crm = CollectionRoutingMap.CompleteRoutingMap([({'id': '2', 'minInclusive': '', 'maxExclusive': '0000000030'}, 2), ({'id': '2', 'minInclusive': '0000000030', 'maxExclusive': 'FF'}, 2)], '')
        self.assertIsNotNone(crm)
if __name__ == '__main__':
    unittest.main()
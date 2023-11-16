import unittest
import pytest
from azure.cosmos._routing.routing_map_provider import SmartRoutingMapProvider
from azure.cosmos._routing.routing_map_provider import CollectionRoutingMap
from azure.cosmos._routing import routing_range as routing_range
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class RoutingMapProviderTests(unittest.TestCase):

    class MockedCosmosClientConnection(object):

        def __init__(self, partition_key_ranges):
            if False:
                for i in range(10):
                    print('nop')
            self.partition_key_ranges = partition_key_ranges

        def _ReadPartitionKeyRanges(self, collection_link):
            if False:
                print('Hello World!')
            return self.partition_key_ranges

    def setUp(self):
        if False:
            while True:
                i = 10
        self.partition_key_ranges = [{u'id': u'0', u'minInclusive': u'', u'maxExclusive': u'05C1C9CD673398'}, {u'id': u'1', u'minInclusive': u'05C1C9CD673398', u'maxExclusive': u'05C1D9CD673398'}, {u'id': u'2', u'minInclusive': u'05C1D9CD673398', u'maxExclusive': u'05C1E399CD6732'}, {u'id': u'3', u'minInclusive': u'05C1E399CD6732', u'maxExclusive': u'05C1E9CD673398'}, {u'id': u'4', u'minInclusive': u'05C1E9CD673398', u'maxExclusive': u'FF'}]
        self.smart_routing_map_provider = self.instantiate_smart_routing_map_provider(self.partition_key_ranges)
        partitionRangeWithInfo = map(lambda r: (r, True), self.partition_key_ranges)
        self.cached_collection_routing_map = CollectionRoutingMap.CompleteRoutingMap(partitionRangeWithInfo, 'sample collection id')

    def instantiate_smart_routing_map_provider(self, partition_key_ranges):
        if False:
            return 10
        client = RoutingMapProviderTests.MockedCosmosClientConnection(partition_key_ranges)
        return SmartRoutingMapProvider(client)

    def test_full_range(self):
        if False:
            return 10
        pkRange = routing_range.Range('', 'FF', True, False)
        overlapping_partition_key_ranges = self.get_overlapping_ranges([pkRange])
        self.assertEqual(len(overlapping_partition_key_ranges), len(self.partition_key_ranges))
        self.assertEqual(overlapping_partition_key_ranges, self.partition_key_ranges)
        pkRange = routing_range.Range('', 'FF', False, False)
        overlapping_partition_key_ranges = self.get_overlapping_ranges([pkRange])
        self.assertEqual(overlapping_partition_key_ranges, self.partition_key_ranges)
        self.assertEqual(self.cached_collection_routing_map.get_overlapping_ranges([pkRange]), self.partition_key_ranges)

    def test_empty_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        pkRange = routing_range.Range('', 'FF', True, False)
        overlapping_partition_key_ranges = self.get_overlapping_ranges([pkRange])
        self.assertEqual(len(overlapping_partition_key_ranges), len(self.partition_key_ranges))
        self.assertEqual(overlapping_partition_key_ranges, self.partition_key_ranges)
        overlapping_partition_key_ranges = self.get_overlapping_ranges([])
        self.assertEqual(len(overlapping_partition_key_ranges), 0)
        empty_start_range = routing_range.Range('', '', False, True)
        empty_end_range = routing_range.Range('FF', 'FF', False, True)
        empty_range = routing_range.Range('AA', 'AA', False, True)
        self.validate_empty_query_ranges([empty_range], [empty_start_range], [empty_end_range], [empty_start_range, empty_range], [empty_start_range, empty_end_range], [empty_range, empty_end_range], [empty_range, empty_range, empty_end_range])

    def test_bad_overlapping_query_ranges(self):
        if False:
            return 10
        r1 = routing_range.Range('', 'AA', True, True)
        r2 = routing_range.Range('AA', 'FF', True, False)

        def func_one_point_overlap():
            if False:
                return 10
            self.smart_routing_map_provider.get_overlapping_ranges('sample collection id', [r1, r2])
        self.assertRaises(ValueError, func_one_point_overlap)
        r1 = routing_range.Range('', 'AB', True, False)
        r2 = routing_range.Range('AA', 'FA', True, False)

        def func_overlap():
            if False:
                print('Hello World!')
            self.smart_routing_map_provider.get_overlapping_ranges('sample collection id', [r1, r2])
        self.assertRaises(ValueError, func_overlap)
        r1 = routing_range.Range('AB', 'AC', True, False)
        r1 = routing_range.Range('AA', 'AB', True, False)

        def func_non_sorted():
            if False:
                while True:
                    i = 10
            self.smart_routing_map_provider.get_overlapping_ranges('sample collection id', [r1, r2])
        self.assertRaises(ValueError, func_overlap)

    def test_empty_ranges_are_thrown_away(self):
        if False:
            print('Hello World!')
        e1 = routing_range.Range('', '', True, False)
        r1 = routing_range.Range('', 'AB', True, False)
        e2 = routing_range.Range('AB', 'AB', True, False)
        r2 = routing_range.Range('AB', 'AC', True, False)
        e3 = routing_range.Range('AC', 'AC', True, False)
        e4 = routing_range.Range('AD', 'AD', True, False)
        self.validate_overlapping_ranges_results([e1, r1, e2, r2, e3, e4], self.get_overlapping_ranges([r1, r2]))
        self.validate_against_cached_collection_results([e1, r1, e2, r2, e3, e4])

    def test_simple(self):
        if False:
            while True:
                i = 10
        r = routing_range.Range('AB', 'AC', True, False)
        self.validate_against_cached_collection_results([r])
        ranges = [routing_range.Range('0000000040', '0000000045', True, False), routing_range.Range('0000000045', '0000000046', True, False), routing_range.Range('0000000046', '0000000050', True, False)]
        self.validate_against_cached_collection_results(ranges)

    def test_simple_boundary(self):
        if False:
            print('Hello World!')
        ranges = [routing_range.Range('05C1C9CD673398', '05C1D9CD673398', True, False)]
        self.validate_against_cached_collection_results(ranges)
        self.validate_overlapping_ranges_results(ranges, self.partition_key_ranges[1:2])

    def test_two_adjacent_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        ranges = [routing_range.Range('05C1C9CD673398', '05C1D9CD673398', True, False), routing_range.Range('05C1D9CD673398', '05C1D9CD673399', True, False)]
        self.validate_against_cached_collection_results(ranges)
        self.validate_overlapping_ranges_results(ranges, self.partition_key_ranges[1:3])

    def test_two_ranges_in_one_partition_key_range(self):
        if False:
            print('Hello World!')
        ranges = [routing_range.Range('05C1C9CD673400', '05C1C9CD673401', True, False), routing_range.Range('05C1C9CD673402', '05C1C9CD673403', True, False)]
        self.validate_against_cached_collection_results(ranges)
        self.validate_overlapping_ranges_results(ranges, self.partition_key_ranges[1:2])

    def test_complex(self):
        if False:
            while True:
                i = 10
        ranges = [routing_range.Range('05C1C9CD673398', '05C1D9CD673391', True, False), routing_range.Range('05C1D9CD673391', '05C1D9CD673392', True, False), routing_range.Range('05C1D9CD673393', '05C1D9CD673395', True, False), routing_range.Range('05C1D9CD673395', '05C1D9CD673395', True, False), routing_range.Range('05C1E9CD673398', '05C1E9CD673401', True, False), routing_range.Range('05C1E9CD673402', '05C1E9CD673403', True, False), routing_range.Range('FF', 'FF', True, False)]
        self.validate_against_cached_collection_results(ranges)
        self.validate_overlapping_ranges_results(ranges, [self.partition_key_ranges[1], self.partition_key_ranges[4]])

    def validate_against_cached_collection_results(self, queryRanges):
        if False:
            for i in range(10):
                print('nop')
        overlapping_partition_key_ranges = self.get_overlapping_ranges(queryRanges)
        self.assertEqual(overlapping_partition_key_ranges, self.cached_collection_routing_map.get_overlapping_ranges(queryRanges))

    def validate_overlapping_ranges_results(self, queryRanges, expected_overlapping_partition_key_ranges):
        if False:
            return 10
        overlapping_partition_key_ranges = self.get_overlapping_ranges(queryRanges)
        self.assertEqual(overlapping_partition_key_ranges, expected_overlapping_partition_key_ranges)

    def validate_empty_query_ranges(self, smart_routing_map_provider, *queryRangesList):
        if False:
            print('Hello World!')
        for queryRanges in queryRangesList:
            self.validate_overlapping_ranges_results(queryRanges, [])

    def get_overlapping_ranges(self, queryRanges):
        if False:
            while True:
                i = 10
        return self.smart_routing_map_provider.get_overlapping_ranges('sample collection id', queryRanges)
if __name__ == '__main__':
    unittest.main()
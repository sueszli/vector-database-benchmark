import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
from kafka.vendor import six
log = logging.getLogger(__name__)
ConsumerPair = namedtuple('ConsumerPair', ['src_member_id', 'dst_member_id'])
'\nRepresents a pair of Kafka consumer ids involved in a partition reassignment.\nEach ConsumerPair corresponds to a particular partition or topic, indicates that the particular partition or some\npartition of the particular topic was moved from the source consumer to the destination consumer\nduring the rebalance. This class helps in determining whether a partition reassignment results in cycles among\nthe generated graph of consumer pairs.\n'

def is_sublist(source, target):
    if False:
        while True:
            i = 10
    'Checks if one list is a sublist of another.\n\n    Arguments:\n      source: the list in which to search for the occurrence of target.\n      target: the list to search for as a sublist of source\n\n    Returns:\n      true if target is in source; false otherwise\n    '
    for index in (i for (i, e) in enumerate(source) if e == target[0]):
        if tuple(source[index:index + len(target)]) == target:
            return True
    return False

class PartitionMovements:
    """
    This class maintains some data structures to simplify lookup of partition movements among consumers.
    At each point of time during a partition rebalance it keeps track of partition movements
    corresponding to each topic, and also possible movement (in form a ConsumerPair object) for each partition.
    """

    def __init__(self):
        if False:
            return 10
        self.partition_movements_by_topic = defaultdict(lambda : defaultdict(set))
        self.partition_movements = {}

    def move_partition(self, partition, old_consumer, new_consumer):
        if False:
            while True:
                i = 10
        pair = ConsumerPair(src_member_id=old_consumer, dst_member_id=new_consumer)
        if partition in self.partition_movements:
            existing_pair = self._remove_movement_record_of_partition(partition)
            assert existing_pair.dst_member_id == old_consumer
            if existing_pair.src_member_id != new_consumer:
                self._add_partition_movement_record(partition, ConsumerPair(src_member_id=existing_pair.src_member_id, dst_member_id=new_consumer))
        else:
            self._add_partition_movement_record(partition, pair)

    def get_partition_to_be_moved(self, partition, old_consumer, new_consumer):
        if False:
            for i in range(10):
                print('nop')
        if partition.topic not in self.partition_movements_by_topic:
            return partition
        if partition in self.partition_movements:
            assert old_consumer == self.partition_movements[partition].dst_member_id
            old_consumer = self.partition_movements[partition].src_member_id
        reverse_pair = ConsumerPair(src_member_id=new_consumer, dst_member_id=old_consumer)
        if reverse_pair not in self.partition_movements_by_topic[partition.topic]:
            return partition
        return next(iter(self.partition_movements_by_topic[partition.topic][reverse_pair]))

    def are_sticky(self):
        if False:
            print('Hello World!')
        for (topic, movements) in six.iteritems(self.partition_movements_by_topic):
            movement_pairs = set(movements.keys())
            if self._has_cycles(movement_pairs):
                log.error('Stickiness is violated for topic {}\nPartition movements for this topic occurred among the following consumer pairs:\n{}'.format(topic, movement_pairs))
                return False
        return True

    def _remove_movement_record_of_partition(self, partition):
        if False:
            i = 10
            return i + 15
        pair = self.partition_movements[partition]
        del self.partition_movements[partition]
        self.partition_movements_by_topic[partition.topic][pair].remove(partition)
        if not self.partition_movements_by_topic[partition.topic][pair]:
            del self.partition_movements_by_topic[partition.topic][pair]
        if not self.partition_movements_by_topic[partition.topic]:
            del self.partition_movements_by_topic[partition.topic]
        return pair

    def _add_partition_movement_record(self, partition, pair):
        if False:
            for i in range(10):
                print('nop')
        self.partition_movements[partition] = pair
        self.partition_movements_by_topic[partition.topic][pair].add(partition)

    def _has_cycles(self, consumer_pairs):
        if False:
            i = 10
            return i + 15
        cycles = set()
        for pair in consumer_pairs:
            reduced_pairs = deepcopy(consumer_pairs)
            reduced_pairs.remove(pair)
            path = [pair.src_member_id]
            if self._is_linked(pair.dst_member_id, pair.src_member_id, reduced_pairs, path) and (not self._is_subcycle(path, cycles)):
                cycles.add(tuple(path))
                log.error('A cycle of length {} was found: {}'.format(len(path) - 1, path))
        for cycle in cycles:
            if len(cycle) == 3:
                return True
        return False

    @staticmethod
    def _is_subcycle(cycle, cycles):
        if False:
            print('Hello World!')
        super_cycle = deepcopy(cycle)
        super_cycle = super_cycle[:-1]
        super_cycle.extend(cycle)
        for found_cycle in cycles:
            if len(found_cycle) == len(cycle) and is_sublist(super_cycle, found_cycle):
                return True
        return False

    def _is_linked(self, src, dst, pairs, current_path):
        if False:
            print('Hello World!')
        if src == dst:
            return False
        if not pairs:
            return False
        if ConsumerPair(src, dst) in pairs:
            current_path.append(src)
            current_path.append(dst)
            return True
        for pair in pairs:
            if pair.src_member_id == src:
                reduced_set = deepcopy(pairs)
                reduced_set.remove(pair)
                current_path.append(pair.src_member_id)
                return self._is_linked(pair.dst_member_id, dst, reduced_set, current_path)
        return False
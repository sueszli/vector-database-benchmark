import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
from kafka.cluster import ClusterMetadata
from kafka.coordinator.assignors.abstract import AbstractPartitionAssignor
from kafka.coordinator.assignors.sticky.partition_movements import PartitionMovements
from kafka.coordinator.assignors.sticky.sorted_set import SortedSet
from kafka.coordinator.protocol import ConsumerProtocolMemberMetadata, ConsumerProtocolMemberAssignment
from kafka.coordinator.protocol import Schema
from kafka.protocol.struct import Struct
from kafka.protocol.types import String, Array, Int32
from kafka.structs import TopicPartition
from kafka.vendor import six
log = logging.getLogger(__name__)
ConsumerGenerationPair = namedtuple('ConsumerGenerationPair', ['consumer', 'generation'])

def has_identical_list_elements(list_):
    if False:
        print('Hello World!')
    'Checks if all lists in the collection have the same members\n\n    Arguments:\n      list_: collection of lists\n\n    Returns:\n      true if all lists in the collection have the same members; false otherwise\n    '
    if not list_:
        return True
    for i in range(1, len(list_)):
        if list_[i] != list_[i - 1]:
            return False
    return True

def subscriptions_comparator_key(element):
    if False:
        print('Hello World!')
    return (len(element[1]), element[0])

def partitions_comparator_key(element):
    if False:
        i = 10
        return i + 15
    return (len(element[1]), element[0].topic, element[0].partition)

def remove_if_present(collection, element):
    if False:
        for i in range(10):
            print('nop')
    try:
        collection.remove(element)
    except (ValueError, KeyError):
        pass
StickyAssignorMemberMetadataV1 = namedtuple('StickyAssignorMemberMetadataV1', ['subscription', 'partitions', 'generation'])

class StickyAssignorUserDataV1(Struct):
    """
    Used for preserving consumer's previously assigned partitions
    list and sending it as user data to the leader during a rebalance
    """
    SCHEMA = Schema(('previous_assignment', Array(('topic', String('utf-8')), ('partitions', Array(Int32)))), ('generation', Int32))

class StickyAssignmentExecutor:

    def __init__(self, cluster, members):
        if False:
            while True:
                i = 10
        self.members = members
        self.current_assignment = defaultdict(list)
        self.previous_assignment = {}
        self.current_partition_consumer = {}
        self.is_fresh_assignment = False
        self.partition_to_all_potential_consumers = {}
        self.consumer_to_all_potential_partitions = {}
        self.sorted_current_subscriptions = SortedSet()
        self.sorted_partitions = []
        self.unassigned_partitions = []
        self.revocation_required = False
        self.partition_movements = PartitionMovements()
        self._initialize(cluster)

    def perform_initial_assignment(self):
        if False:
            i = 10
            return i + 15
        self._populate_sorted_partitions()
        self._populate_partitions_to_reassign()

    def balance(self):
        if False:
            for i in range(10):
                print('nop')
        self._initialize_current_subscriptions()
        initializing = len(self.current_assignment[self._get_consumer_with_most_subscriptions()]) == 0
        for partition in self.unassigned_partitions:
            if not self.partition_to_all_potential_consumers[partition]:
                continue
            self._assign_partition(partition)
        fixed_partitions = set()
        for partition in six.iterkeys(self.partition_to_all_potential_consumers):
            if not self._can_partition_participate_in_reassignment(partition):
                fixed_partitions.add(partition)
        for fixed_partition in fixed_partitions:
            remove_if_present(self.sorted_partitions, fixed_partition)
            remove_if_present(self.unassigned_partitions, fixed_partition)
        fixed_assignments = {}
        for consumer in six.iterkeys(self.consumer_to_all_potential_partitions):
            if not self._can_consumer_participate_in_reassignment(consumer):
                self._remove_consumer_from_current_subscriptions_and_maintain_order(consumer)
                fixed_assignments[consumer] = self.current_assignment[consumer]
                del self.current_assignment[consumer]
        prebalance_assignment = deepcopy(self.current_assignment)
        prebalance_partition_consumers = deepcopy(self.current_partition_consumer)
        if not self.revocation_required:
            self._perform_reassignments(self.unassigned_partitions)
        reassignment_performed = self._perform_reassignments(self.sorted_partitions)
        if not initializing and reassignment_performed and (self._get_balance_score(self.current_assignment) >= self._get_balance_score(prebalance_assignment)):
            self.current_assignment = prebalance_assignment
            self.current_partition_consumer.clear()
            self.current_partition_consumer.update(prebalance_partition_consumers)
        for (consumer, partitions) in six.iteritems(fixed_assignments):
            self.current_assignment[consumer] = partitions
            self._add_consumer_to_current_subscriptions_and_maintain_order(consumer)

    def get_final_assignment(self, member_id):
        if False:
            print('Hello World!')
        assignment = defaultdict(list)
        for topic_partition in self.current_assignment[member_id]:
            assignment[topic_partition.topic].append(topic_partition.partition)
        assignment = {k: sorted(v) for (k, v) in six.iteritems(assignment)}
        return six.viewitems(assignment)

    def _initialize(self, cluster):
        if False:
            print('Hello World!')
        self._init_current_assignments(self.members)
        for topic in cluster.topics():
            partitions = cluster.partitions_for_topic(topic)
            if partitions is None:
                log.warning('No partition metadata for topic %s', topic)
                continue
            for p in partitions:
                partition = TopicPartition(topic=topic, partition=p)
                self.partition_to_all_potential_consumers[partition] = []
        for (consumer_id, member_metadata) in six.iteritems(self.members):
            self.consumer_to_all_potential_partitions[consumer_id] = []
            for topic in member_metadata.subscription:
                if cluster.partitions_for_topic(topic) is None:
                    log.warning('No partition metadata for topic {}'.format(topic))
                    continue
                for p in cluster.partitions_for_topic(topic):
                    partition = TopicPartition(topic=topic, partition=p)
                    self.consumer_to_all_potential_partitions[consumer_id].append(partition)
                    self.partition_to_all_potential_consumers[partition].append(consumer_id)
            if consumer_id not in self.current_assignment:
                self.current_assignment[consumer_id] = []

    def _init_current_assignments(self, members):
        if False:
            i = 10
            return i + 15
        sorted_partition_consumers_by_generation = {}
        for (consumer, member_metadata) in six.iteritems(members):
            for partitions in member_metadata.partitions:
                if partitions in sorted_partition_consumers_by_generation:
                    consumers = sorted_partition_consumers_by_generation[partitions]
                    if member_metadata.generation and member_metadata.generation in consumers:
                        log.warning('Partition {} is assigned to multiple consumers following sticky assignment generation {}.'.format(partitions, member_metadata.generation))
                    else:
                        consumers[member_metadata.generation] = consumer
                else:
                    sorted_consumers = {member_metadata.generation: consumer}
                    sorted_partition_consumers_by_generation[partitions] = sorted_consumers
        for (partitions, consumers) in six.iteritems(sorted_partition_consumers_by_generation):
            generations = sorted(consumers.keys(), reverse=True)
            self.current_assignment[consumers[generations[0]]].append(partitions)
            if len(generations) > 1:
                self.previous_assignment[partitions] = ConsumerGenerationPair(consumer=consumers[generations[1]], generation=generations[1])
        self.is_fresh_assignment = len(self.current_assignment) == 0
        for (consumer_id, partitions) in six.iteritems(self.current_assignment):
            for partition in partitions:
                self.current_partition_consumer[partition] = consumer_id

    def _are_subscriptions_identical(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            true, if both potential consumers of partitions and potential partitions that consumers can\n            consume are the same\n        '
        if not has_identical_list_elements(list(six.itervalues(self.partition_to_all_potential_consumers))):
            return False
        return has_identical_list_elements(list(six.itervalues(self.consumer_to_all_potential_partitions)))

    def _populate_sorted_partitions(self):
        if False:
            return 10
        all_partitions = set(((tp, tuple(consumers)) for (tp, consumers) in six.iteritems(self.partition_to_all_potential_consumers)))
        partitions_sorted_by_num_of_potential_consumers = sorted(all_partitions, key=partitions_comparator_key)
        self.sorted_partitions = []
        if not self.is_fresh_assignment and self._are_subscriptions_identical():
            assignments = deepcopy(self.current_assignment)
            for (consumer_id, partitions) in six.iteritems(assignments):
                to_remove = []
                for partition in partitions:
                    if partition not in self.partition_to_all_potential_consumers:
                        to_remove.append(partition)
                for partition in to_remove:
                    partitions.remove(partition)
            sorted_consumers = SortedSet(iterable=[(consumer, tuple(partitions)) for (consumer, partitions) in six.iteritems(assignments)], key=subscriptions_comparator_key)
            while sorted_consumers:
                (consumer, _) = sorted_consumers.pop_last()
                remaining_partitions = assignments[consumer]
                previous_partitions = set(six.iterkeys(self.previous_assignment)).intersection(set(remaining_partitions))
                if previous_partitions:
                    partition = previous_partitions.pop()
                    remaining_partitions.remove(partition)
                    self.sorted_partitions.append(partition)
                    sorted_consumers.add((consumer, tuple(assignments[consumer])))
                elif remaining_partitions:
                    self.sorted_partitions.append(remaining_partitions.pop())
                    sorted_consumers.add((consumer, tuple(assignments[consumer])))
            while partitions_sorted_by_num_of_potential_consumers:
                partition = partitions_sorted_by_num_of_potential_consumers.pop(0)[0]
                if partition not in self.sorted_partitions:
                    self.sorted_partitions.append(partition)
        else:
            while partitions_sorted_by_num_of_potential_consumers:
                self.sorted_partitions.append(partitions_sorted_by_num_of_potential_consumers.pop(0)[0])

    def _populate_partitions_to_reassign(self):
        if False:
            print('Hello World!')
        self.unassigned_partitions = deepcopy(self.sorted_partitions)
        assignments_to_remove = []
        for (consumer_id, partitions) in six.iteritems(self.current_assignment):
            if consumer_id not in self.members:
                for partition in partitions:
                    del self.current_partition_consumer[partition]
                assignments_to_remove.append(consumer_id)
            else:
                partitions_to_remove = []
                for partition in partitions:
                    if partition not in self.partition_to_all_potential_consumers:
                        partitions_to_remove.append(partition)
                    elif partition.topic not in self.members[consumer_id].subscription:
                        partitions_to_remove.append(partition)
                        self.revocation_required = True
                    else:
                        self.unassigned_partitions.remove(partition)
                for partition in partitions_to_remove:
                    self.current_assignment[consumer_id].remove(partition)
                    del self.current_partition_consumer[partition]
        for consumer_id in assignments_to_remove:
            del self.current_assignment[consumer_id]

    def _initialize_current_subscriptions(self):
        if False:
            while True:
                i = 10
        self.sorted_current_subscriptions = SortedSet(iterable=[(consumer, tuple(partitions)) for (consumer, partitions) in six.iteritems(self.current_assignment)], key=subscriptions_comparator_key)

    def _get_consumer_with_least_subscriptions(self):
        if False:
            i = 10
            return i + 15
        return self.sorted_current_subscriptions.first()[0]

    def _get_consumer_with_most_subscriptions(self):
        if False:
            i = 10
            return i + 15
        return self.sorted_current_subscriptions.last()[0]

    def _remove_consumer_from_current_subscriptions_and_maintain_order(self, consumer):
        if False:
            for i in range(10):
                print('nop')
        self.sorted_current_subscriptions.remove((consumer, tuple(self.current_assignment[consumer])))

    def _add_consumer_to_current_subscriptions_and_maintain_order(self, consumer):
        if False:
            print('Hello World!')
        self.sorted_current_subscriptions.add((consumer, tuple(self.current_assignment[consumer])))

    def _is_balanced(self):
        if False:
            print('Hello World!')
        'Determines if the current assignment is a balanced one'
        if len(self.current_assignment[self._get_consumer_with_least_subscriptions()]) >= len(self.current_assignment[self._get_consumer_with_most_subscriptions()]) - 1:
            return True
        all_assigned_partitions = {}
        for (consumer_id, consumer_partitions) in six.iteritems(self.current_assignment):
            for partition in consumer_partitions:
                if partition in all_assigned_partitions:
                    log.error('{} is assigned to more than one consumer.'.format(partition))
                all_assigned_partitions[partition] = consumer_id
        for (consumer, _) in self.sorted_current_subscriptions:
            consumer_partition_count = len(self.current_assignment[consumer])
            if consumer_partition_count == len(self.consumer_to_all_potential_partitions[consumer]):
                continue
            for partition in self.consumer_to_all_potential_partitions[consumer]:
                if partition not in self.current_assignment[consumer]:
                    other_consumer = all_assigned_partitions[partition]
                    other_consumer_partition_count = len(self.current_assignment[other_consumer])
                    if consumer_partition_count < other_consumer_partition_count:
                        return False
        return True

    def _assign_partition(self, partition):
        if False:
            while True:
                i = 10
        for (consumer, _) in self.sorted_current_subscriptions:
            if partition in self.consumer_to_all_potential_partitions[consumer]:
                self._remove_consumer_from_current_subscriptions_and_maintain_order(consumer)
                self.current_assignment[consumer].append(partition)
                self.current_partition_consumer[partition] = consumer
                self._add_consumer_to_current_subscriptions_and_maintain_order(consumer)
                break

    def _can_partition_participate_in_reassignment(self, partition):
        if False:
            while True:
                i = 10
        return len(self.partition_to_all_potential_consumers[partition]) >= 2

    def _can_consumer_participate_in_reassignment(self, consumer):
        if False:
            i = 10
            return i + 15
        current_partitions = self.current_assignment[consumer]
        current_assignment_size = len(current_partitions)
        max_assignment_size = len(self.consumer_to_all_potential_partitions[consumer])
        if current_assignment_size > max_assignment_size:
            log.error('The consumer {} is assigned more partitions than the maximum possible.'.format(consumer))
        if current_assignment_size < max_assignment_size:
            return True
        for partition in current_partitions:
            if self._can_partition_participate_in_reassignment(partition):
                return True
        return False

    def _perform_reassignments(self, reassignable_partitions):
        if False:
            i = 10
            return i + 15
        reassignment_performed = False
        while True:
            modified = False
            for partition in reassignable_partitions:
                if self._is_balanced():
                    break
                if len(self.partition_to_all_potential_consumers[partition]) <= 1:
                    log.error('Expected more than one potential consumer for partition {}'.format(partition))
                consumer = self.current_partition_consumer.get(partition)
                if consumer is None:
                    log.error('Expected partition {} to be assigned to a consumer'.format(partition))
                if partition in self.previous_assignment and len(self.current_assignment[consumer]) > len(self.current_assignment[self.previous_assignment[partition].consumer]) + 1:
                    self._reassign_partition_to_consumer(partition, self.previous_assignment[partition].consumer)
                    reassignment_performed = True
                    modified = True
                    continue
                for other_consumer in self.partition_to_all_potential_consumers[partition]:
                    if len(self.current_assignment[consumer]) > len(self.current_assignment[other_consumer]) + 1:
                        self._reassign_partition(partition)
                        reassignment_performed = True
                        modified = True
                        break
            if not modified:
                break
        return reassignment_performed

    def _reassign_partition(self, partition):
        if False:
            i = 10
            return i + 15
        new_consumer = None
        for (another_consumer, _) in self.sorted_current_subscriptions:
            if partition in self.consumer_to_all_potential_partitions[another_consumer]:
                new_consumer = another_consumer
                break
        assert new_consumer is not None
        self._reassign_partition_to_consumer(partition, new_consumer)

    def _reassign_partition_to_consumer(self, partition, new_consumer):
        if False:
            print('Hello World!')
        consumer = self.current_partition_consumer[partition]
        partition_to_be_moved = self.partition_movements.get_partition_to_be_moved(partition, consumer, new_consumer)
        self._move_partition(partition_to_be_moved, new_consumer)

    def _move_partition(self, partition, new_consumer):
        if False:
            print('Hello World!')
        old_consumer = self.current_partition_consumer[partition]
        self._remove_consumer_from_current_subscriptions_and_maintain_order(old_consumer)
        self._remove_consumer_from_current_subscriptions_and_maintain_order(new_consumer)
        self.partition_movements.move_partition(partition, old_consumer, new_consumer)
        self.current_assignment[old_consumer].remove(partition)
        self.current_assignment[new_consumer].append(partition)
        self.current_partition_consumer[partition] = new_consumer
        self._add_consumer_to_current_subscriptions_and_maintain_order(new_consumer)
        self._add_consumer_to_current_subscriptions_and_maintain_order(old_consumer)

    @staticmethod
    def _get_balance_score(assignment):
        if False:
            return 10
        'Calculates a balance score of a give assignment\n        as the sum of assigned partitions size difference of all consumer pairs.\n        A perfectly balanced assignment (with all consumers getting the same number of partitions)\n        has a balance score of 0. Lower balance score indicates a more balanced assignment.\n\n        Arguments:\n          assignment (dict): {consumer: list of assigned topic partitions}\n\n        Returns:\n          the balance score of the assignment\n        '
        score = 0
        consumer_to_assignment = {}
        for (consumer_id, partitions) in six.iteritems(assignment):
            consumer_to_assignment[consumer_id] = len(partitions)
        consumers_to_explore = set(consumer_to_assignment.keys())
        for consumer_id in consumer_to_assignment.keys():
            if consumer_id in consumers_to_explore:
                consumers_to_explore.remove(consumer_id)
                for other_consumer_id in consumers_to_explore:
                    score += abs(consumer_to_assignment[consumer_id] - consumer_to_assignment[other_consumer_id])
        return score

class StickyPartitionAssignor(AbstractPartitionAssignor):
    """
    https://cwiki.apache.org/confluence/display/KAFKA/KIP-54+-+Sticky+Partition+Assignment+Strategy
    
    The sticky assignor serves two purposes. First, it guarantees an assignment that is as balanced as possible, meaning either:
    - the numbers of topic partitions assigned to consumers differ by at most one; or
    - each consumer that has 2+ fewer topic partitions than some other consumer cannot get any of those topic partitions transferred to it.
    
    Second, it preserved as many existing assignment as possible when a reassignment occurs.
    This helps in saving some of the overhead processing when topic partitions move from one consumer to another.
    
    Starting fresh it would work by distributing the partitions over consumers as evenly as possible.
    Even though this may sound similar to how round robin assignor works, the second example below shows that it is not.
    During a reassignment it would perform the reassignment in such a way that in the new assignment
    - topic partitions are still distributed as evenly as possible, and
    - topic partitions stay with their previously assigned consumers as much as possible.
    
    The first goal above takes precedence over the second one.
    
    Example 1.
    Suppose there are three consumers C0, C1, C2,
    four topics t0, t1, t2, t3, and each topic has 2 partitions,
    resulting in partitions t0p0, t0p1, t1p0, t1p1, t2p0, t2p1, t3p0, t3p1.
    Each consumer is subscribed to all three topics.
    
    The assignment with both sticky and round robin assignors will be:
    - C0: [t0p0, t1p1, t3p0]
    - C1: [t0p1, t2p0, t3p1]
    - C2: [t1p0, t2p1]
    
    Now, let's assume C1 is removed and a reassignment is about to happen. The round robin assignor would produce:
    - C0: [t0p0, t1p0, t2p0, t3p0]
    - C2: [t0p1, t1p1, t2p1, t3p1]
    
    while the sticky assignor would result in:
    - C0 [t0p0, t1p1, t3p0, t2p0]
    - C2 [t1p0, t2p1, t0p1, t3p1]
    preserving all the previous assignments (unlike the round robin assignor).
    
    
    Example 2.
    There are three consumers C0, C1, C2,
    and three topics t0, t1, t2, with 1, 2, and 3 partitions respectively.
    Therefore, the partitions are t0p0, t1p0, t1p1, t2p0, t2p1, t2p2.
    C0 is subscribed to t0;
    C1 is subscribed to t0, t1;
    and C2 is subscribed to t0, t1, t2.
    
    The round robin assignor would come up with the following assignment:
    - C0 [t0p0]
    - C1 [t1p0]
    - C2 [t1p1, t2p0, t2p1, t2p2]
    
    which is not as balanced as the assignment suggested by sticky assignor:
    - C0 [t0p0]
    - C1 [t1p0, t1p1]
    - C2 [t2p0, t2p1, t2p2]
    
    Now, if consumer C0 is removed, these two assignors would produce the following assignments.
    Round Robin (preserves 3 partition assignments):
    - C1 [t0p0, t1p1]
    - C2 [t1p0, t2p0, t2p1, t2p2]
    
    Sticky (preserves 5 partition assignments):
    - C1 [t1p0, t1p1, t0p0]
    - C2 [t2p0, t2p1, t2p2]
    """
    DEFAULT_GENERATION_ID = -1
    name = 'sticky'
    version = 0
    member_assignment = None
    generation = DEFAULT_GENERATION_ID
    _latest_partition_movements = None

    @classmethod
    def assign(cls, cluster, members):
        if False:
            for i in range(10):
                print('nop')
        'Performs group assignment given cluster metadata and member subscriptions\n\n        Arguments:\n            cluster (ClusterMetadata): cluster metadata\n            members (dict of {member_id: MemberMetadata}): decoded metadata for each member in the group.\n\n        Returns:\n          dict: {member_id: MemberAssignment}\n        '
        members_metadata = {}
        for (consumer, member_metadata) in six.iteritems(members):
            members_metadata[consumer] = cls.parse_member_metadata(member_metadata)
        executor = StickyAssignmentExecutor(cluster, members_metadata)
        executor.perform_initial_assignment()
        executor.balance()
        cls._latest_partition_movements = executor.partition_movements
        assignment = {}
        for member_id in members:
            assignment[member_id] = ConsumerProtocolMemberAssignment(cls.version, sorted(executor.get_final_assignment(member_id)), b'')
        return assignment

    @classmethod
    def parse_member_metadata(cls, metadata):
        if False:
            i = 10
            return i + 15
        '\n        Parses member metadata into a python object.\n        This implementation only serializes and deserializes the StickyAssignorMemberMetadataV1 user data,\n        since no StickyAssignor written in Python was deployed ever in the wild with version V0, meaning that\n        there is no need to support backward compatibility with V0.\n\n        Arguments:\n          metadata (MemberMetadata): decoded metadata for a member of the group.\n\n        Returns:\n          parsed metadata (StickyAssignorMemberMetadataV1)\n        '
        user_data = metadata.user_data
        if not user_data:
            return StickyAssignorMemberMetadataV1(partitions=[], generation=cls.DEFAULT_GENERATION_ID, subscription=metadata.subscription)
        try:
            decoded_user_data = StickyAssignorUserDataV1.decode(user_data)
        except Exception as e:
            log.error('Could not parse member data', e)
            return StickyAssignorMemberMetadataV1(partitions=[], generation=cls.DEFAULT_GENERATION_ID, subscription=metadata.subscription)
        member_partitions = []
        for (topic, partitions) in decoded_user_data.previous_assignment:
            member_partitions.extend([TopicPartition(topic, partition) for partition in partitions])
        return StickyAssignorMemberMetadataV1(partitions=member_partitions, generation=decoded_user_data.generation, subscription=metadata.subscription)

    @classmethod
    def metadata(cls, topics):
        if False:
            return 10
        return cls._metadata(topics, cls.member_assignment, cls.generation)

    @classmethod
    def _metadata(cls, topics, member_assignment_partitions, generation=-1):
        if False:
            i = 10
            return i + 15
        if member_assignment_partitions is None:
            log.debug('No member assignment available')
            user_data = b''
        else:
            log.debug('Member assignment is available, generating the metadata: generation {}'.format(cls.generation))
            partitions_by_topic = defaultdict(list)
            for topic_partition in member_assignment_partitions:
                partitions_by_topic[topic_partition.topic].append(topic_partition.partition)
            data = StickyAssignorUserDataV1(six.viewitems(partitions_by_topic), generation)
            user_data = data.encode()
        return ConsumerProtocolMemberMetadata(cls.version, list(topics), user_data)

    @classmethod
    def on_assignment(cls, assignment):
        if False:
            return 10
        "Callback that runs on each assignment. Updates assignor's state.\n\n        Arguments:\n          assignment: MemberAssignment\n        "
        log.debug('On assignment: assignment={}'.format(assignment))
        cls.member_assignment = assignment.partitions()

    @classmethod
    def on_generation_assignment(cls, generation):
        if False:
            while True:
                i = 10
        "Callback that runs on each assignment. Updates assignor's generation id.\n\n        Arguments:\n          generation: generation id\n        "
        log.debug('On generation assignment: generation={}'.format(generation))
        cls.generation = generation
from __future__ import absolute_import
import abc
import logging
import re
from kafka.vendor import six
from kafka.errors import IllegalStateError
from kafka.protocol.offset import OffsetResetStrategy
from kafka.structs import OffsetAndMetadata
log = logging.getLogger(__name__)

class SubscriptionState(object):
    """
    A class for tracking the topics, partitions, and offsets for the consumer.
    A partition is "assigned" either directly with assign_from_user() (manual
    assignment) or with assign_from_subscribed() (automatic assignment from
    subscription).

    Once assigned, the partition is not considered "fetchable" until its initial
    position has been set with seek(). Fetchable partitions track a fetch
    position which is used to set the offset of the next fetch, and a consumed
    position which is the last offset that has been returned to the user. You
    can suspend fetching from a partition through pause() without affecting the
    fetched/consumed offsets. The partition will remain unfetchable until the
    resume() is used. You can also query the pause state independently with
    is_paused().

    Note that pause state as well as fetch/consumed positions are not preserved
    when partition assignment is changed whether directly by the user or
    through a group rebalance.

    This class also maintains a cache of the latest commit position for each of
    the assigned partitions. This is updated through committed() and can be used
    to set the initial fetch position (e.g. Fetcher._reset_offset() ).
    """
    _SUBSCRIPTION_EXCEPTION_MESSAGE = 'You must choose only one way to configure your consumer: (1) subscribe to specific topics by name, (2) subscribe to topics matching a regex pattern, (3) assign itself specific topic-partitions.'
    _MAX_NAME_LENGTH = 249
    _TOPIC_LEGAL_CHARS = re.compile('^[a-zA-Z0-9._-]+$')

    def __init__(self, offset_reset_strategy='earliest'):
        if False:
            i = 10
            return i + 15
        "Initialize a SubscriptionState instance\n\n        Keyword Arguments:\n            offset_reset_strategy: 'earliest' or 'latest', otherwise\n                exception will be raised when fetching an offset that is no\n                longer available. Default: 'earliest'\n        "
        try:
            offset_reset_strategy = getattr(OffsetResetStrategy, offset_reset_strategy.upper())
        except AttributeError:
            log.warning('Unrecognized offset_reset_strategy, using NONE')
            offset_reset_strategy = OffsetResetStrategy.NONE
        self._default_offset_reset_strategy = offset_reset_strategy
        self.subscription = None
        self.subscribed_pattern = None
        self._group_subscription = set()
        self._user_assignment = set()
        self.assignment = dict()
        self.listener = None
        self.needs_fetch_committed_offsets = True

    def subscribe(self, topics=(), pattern=None, listener=None):
        if False:
            i = 10
            return i + 15
        "Subscribe to a list of topics, or a topic regex pattern.\n\n        Partitions will be dynamically assigned via a group coordinator.\n        Topic subscriptions are not incremental: this list will replace the\n        current assignment (if there is one).\n\n        This method is incompatible with assign_from_user()\n\n        Arguments:\n            topics (list): List of topics for subscription.\n            pattern (str): Pattern to match available topics. You must provide\n                either topics or pattern, but not both.\n            listener (ConsumerRebalanceListener): Optionally include listener\n                callback, which will be called before and after each rebalance\n                operation.\n\n                As part of group management, the consumer will keep track of the\n                list of consumers that belong to a particular group and will\n                trigger a rebalance operation if one of the following events\n                trigger:\n\n                * Number of partitions change for any of the subscribed topics\n                * Topic is created or deleted\n                * An existing member of the consumer group dies\n                * A new member is added to the consumer group\n\n                When any of these events are triggered, the provided listener\n                will be invoked first to indicate that the consumer's assignment\n                has been revoked, and then again when the new assignment has\n                been received. Note that this listener will immediately override\n                any listener set in a previous call to subscribe. It is\n                guaranteed, however, that the partitions revoked/assigned\n                through this interface are from topics subscribed in this call.\n        "
        if self._user_assignment or (topics and pattern):
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)
        assert topics or pattern, 'Must provide topics or pattern'
        if pattern:
            log.info('Subscribing to pattern: /%s/', pattern)
            self.subscription = set()
            self.subscribed_pattern = re.compile(pattern)
        else:
            self.change_subscription(topics)
        if listener and (not isinstance(listener, ConsumerRebalanceListener)):
            raise TypeError('listener must be a ConsumerRebalanceListener')
        self.listener = listener

    def _ensure_valid_topic_name(self, topic):
        if False:
            i = 10
            return i + 15
        ' Ensures that the topic name is valid according to the kafka source. '
        if topic is None:
            raise TypeError('All topics must not be None')
        if not isinstance(topic, six.string_types):
            raise TypeError('All topics must be strings')
        if len(topic) == 0:
            raise ValueError('All topics must be non-empty strings')
        if topic == '.' or topic == '..':
            raise ValueError('Topic name cannot be "." or ".."')
        if len(topic) > self._MAX_NAME_LENGTH:
            raise ValueError('Topic name is illegal, it can\'t be longer than {0} characters, topic: "{1}"'.format(self._MAX_NAME_LENGTH, topic))
        if not self._TOPIC_LEGAL_CHARS.match(topic):
            raise ValueError('Topic name "{0}" is illegal, it contains a character other than ASCII alphanumerics, ".", "_" and "-"'.format(topic))

    def change_subscription(self, topics):
        if False:
            i = 10
            return i + 15
        "Change the topic subscription.\n\n        Arguments:\n            topics (list of str): topics for subscription\n\n        Raises:\n            IllegalStateError: if assign_from_user has been used already\n            TypeError: if a topic is None or a non-str\n            ValueError: if a topic is an empty string or\n                        - a topic name is '.' or '..' or\n                        - a topic name does not consist of ASCII-characters/'-'/'_'/'.'\n        "
        if self._user_assignment:
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)
        if isinstance(topics, six.string_types):
            topics = [topics]
        if self.subscription == set(topics):
            log.warning('subscription unchanged by change_subscription(%s)', topics)
            return
        for t in topics:
            self._ensure_valid_topic_name(t)
        log.info('Updating subscribed topics to: %s', topics)
        self.subscription = set(topics)
        self._group_subscription.update(topics)
        for tp in set(self.assignment.keys()):
            if tp.topic not in self.subscription:
                del self.assignment[tp]

    def group_subscribe(self, topics):
        if False:
            print('Hello World!')
        'Add topics to the current group subscription.\n\n        This is used by the group leader to ensure that it receives metadata\n        updates for all topics that any member of the group is subscribed to.\n\n        Arguments:\n            topics (list of str): topics to add to the group subscription\n        '
        if self._user_assignment:
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)
        self._group_subscription.update(topics)

    def reset_group_subscription(self):
        if False:
            while True:
                i = 10
        "Reset the group's subscription to only contain topics subscribed by this consumer."
        if self._user_assignment:
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)
        assert self.subscription is not None, 'Subscription required'
        self._group_subscription.intersection_update(self.subscription)

    def assign_from_user(self, partitions):
        if False:
            return 10
        "Manually assign a list of TopicPartitions to this consumer.\n\n        This interface does not allow for incremental assignment and will\n        replace the previous assignment (if there was one).\n\n        Manual topic assignment through this method does not use the consumer's\n        group management functionality. As such, there will be no rebalance\n        operation triggered when group membership or cluster and topic metadata\n        change. Note that it is not possible to use both manual partition\n        assignment with assign() and group assignment with subscribe().\n\n        Arguments:\n            partitions (list of TopicPartition): assignment for this instance.\n\n        Raises:\n            IllegalStateError: if consumer has already called subscribe()\n        "
        if self.subscription is not None:
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)
        if self._user_assignment != set(partitions):
            self._user_assignment = set(partitions)
            for partition in partitions:
                if partition not in self.assignment:
                    self._add_assigned_partition(partition)
            for tp in set(self.assignment.keys()) - self._user_assignment:
                del self.assignment[tp]
            self.needs_fetch_committed_offsets = True

    def assign_from_subscribed(self, assignments):
        if False:
            i = 10
            return i + 15
        "Update the assignment to the specified partitions\n\n        This method is called by the coordinator to dynamically assign\n        partitions based on the consumer's topic subscription. This is different\n        from assign_from_user() which directly sets the assignment from a\n        user-supplied TopicPartition list.\n\n        Arguments:\n            assignments (list of TopicPartition): partitions to assign to this\n                consumer instance.\n        "
        if not self.partitions_auto_assigned():
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)
        for tp in assignments:
            if tp.topic not in self.subscription:
                raise ValueError('Assigned partition %s for non-subscribed topic.' % (tp,))
        self.assignment.clear()
        for tp in assignments:
            self._add_assigned_partition(tp)
        self.needs_fetch_committed_offsets = True
        log.info('Updated partition assignment: %s', assignments)

    def unsubscribe(self):
        if False:
            i = 10
            return i + 15
        'Clear all topic subscriptions and partition assignments'
        self.subscription = None
        self._user_assignment.clear()
        self.assignment.clear()
        self.subscribed_pattern = None

    def group_subscription(self):
        if False:
            for i in range(10):
                print('nop')
        "Get the topic subscription for the group.\n\n        For the leader, this will include the union of all member subscriptions.\n        For followers, it is the member's subscription only.\n\n        This is used when querying topic metadata to detect metadata changes\n        that would require rebalancing (the leader fetches metadata for all\n        topics in the group so that it can do partition assignment).\n\n        Returns:\n            set: topics\n        "
        return self._group_subscription

    def seek(self, partition, offset):
        if False:
            print('Hello World!')
        'Manually specify the fetch offset for a TopicPartition.\n\n        Overrides the fetch offsets that the consumer will use on the next\n        poll(). If this API is invoked for the same partition more than once,\n        the latest offset will be used on the next poll(). Note that you may\n        lose data if this API is arbitrarily used in the middle of consumption,\n        to reset the fetch offsets.\n\n        Arguments:\n            partition (TopicPartition): partition for seek operation\n            offset (int): message offset in partition\n        '
        self.assignment[partition].seek(offset)

    def assigned_partitions(self):
        if False:
            print('Hello World!')
        'Return set of TopicPartitions in current assignment.'
        return set(self.assignment.keys())

    def paused_partitions(self):
        if False:
            print('Hello World!')
        'Return current set of paused TopicPartitions.'
        return set((partition for partition in self.assignment if self.is_paused(partition)))

    def fetchable_partitions(self):
        if False:
            i = 10
            return i + 15
        'Return set of TopicPartitions that should be Fetched.'
        fetchable = set()
        for (partition, state) in six.iteritems(self.assignment):
            if state.is_fetchable():
                fetchable.add(partition)
        return fetchable

    def partitions_auto_assigned(self):
        if False:
            return 10
        'Return True unless user supplied partitions manually.'
        return self.subscription is not None

    def all_consumed_offsets(self):
        if False:
            while True:
                i = 10
        'Returns consumed offsets as {TopicPartition: OffsetAndMetadata}'
        all_consumed = {}
        for (partition, state) in six.iteritems(self.assignment):
            if state.has_valid_position:
                all_consumed[partition] = OffsetAndMetadata(state.position, '')
        return all_consumed

    def need_offset_reset(self, partition, offset_reset_strategy=None):
        if False:
            return 10
        'Mark partition for offset reset using specified or default strategy.\n\n        Arguments:\n            partition (TopicPartition): partition to mark\n            offset_reset_strategy (OffsetResetStrategy, optional)\n        '
        if offset_reset_strategy is None:
            offset_reset_strategy = self._default_offset_reset_strategy
        self.assignment[partition].await_reset(offset_reset_strategy)

    def has_default_offset_reset_policy(self):
        if False:
            print('Hello World!')
        'Return True if default offset reset policy is Earliest or Latest'
        return self._default_offset_reset_strategy != OffsetResetStrategy.NONE

    def is_offset_reset_needed(self, partition):
        if False:
            i = 10
            return i + 15
        return self.assignment[partition].awaiting_reset

    def has_all_fetch_positions(self):
        if False:
            return 10
        for state in self.assignment.values():
            if not state.has_valid_position:
                return False
        return True

    def missing_fetch_positions(self):
        if False:
            for i in range(10):
                print('nop')
        missing = set()
        for (partition, state) in six.iteritems(self.assignment):
            if not state.has_valid_position:
                missing.add(partition)
        return missing

    def is_assigned(self, partition):
        if False:
            print('Hello World!')
        return partition in self.assignment

    def is_paused(self, partition):
        if False:
            print('Hello World!')
        return partition in self.assignment and self.assignment[partition].paused

    def is_fetchable(self, partition):
        if False:
            print('Hello World!')
        return partition in self.assignment and self.assignment[partition].is_fetchable()

    def pause(self, partition):
        if False:
            i = 10
            return i + 15
        self.assignment[partition].pause()

    def resume(self, partition):
        if False:
            for i in range(10):
                print('nop')
        self.assignment[partition].resume()

    def _add_assigned_partition(self, partition):
        if False:
            return 10
        self.assignment[partition] = TopicPartitionState()

class TopicPartitionState(object):

    def __init__(self):
        if False:
            return 10
        self.committed = None
        self.has_valid_position = False
        self.paused = False
        self.awaiting_reset = False
        self.reset_strategy = None
        self._position = None
        self.highwater = None
        self.drop_pending_message_set = False
        self.last_offset_from_message_batch = None

    def _set_position(self, offset):
        if False:
            return 10
        assert self.has_valid_position, 'Valid position required'
        self._position = offset

    def _get_position(self):
        if False:
            i = 10
            return i + 15
        return self._position
    position = property(_get_position, _set_position, None, 'last position')

    def await_reset(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        self.awaiting_reset = True
        self.reset_strategy = strategy
        self._position = None
        self.last_offset_from_message_batch = None
        self.has_valid_position = False

    def seek(self, offset):
        if False:
            return 10
        self._position = offset
        self.awaiting_reset = False
        self.reset_strategy = None
        self.has_valid_position = True
        self.drop_pending_message_set = True
        self.last_offset_from_message_batch = None

    def pause(self):
        if False:
            while True:
                i = 10
        self.paused = True

    def resume(self):
        if False:
            while True:
                i = 10
        self.paused = False

    def is_fetchable(self):
        if False:
            i = 10
            return i + 15
        return not self.paused and self.has_valid_position

class ConsumerRebalanceListener(object):
    """
    A callback interface that the user can implement to trigger custom actions
    when the set of partitions assigned to the consumer changes.

    This is applicable when the consumer is having Kafka auto-manage group
    membership. If the consumer's directly assign partitions, those
    partitions will never be reassigned and this callback is not applicable.

    When Kafka is managing the group membership, a partition re-assignment will
    be triggered any time the members of the group changes or the subscription
    of the members changes. This can occur when processes die, new process
    instances are added or old instances come back to life after failure.
    Rebalances can also be triggered by changes affecting the subscribed
    topics (e.g. when then number of partitions is administratively adjusted).

    There are many uses for this functionality. One common use is saving offsets
    in a custom store. By saving offsets in the on_partitions_revoked(), call we
    can ensure that any time partition assignment changes the offset gets saved.

    Another use is flushing out any kind of cache of intermediate results the
    consumer may be keeping. For example, consider a case where the consumer is
    subscribed to a topic containing user page views, and the goal is to count
    the number of page views per users for each five minute window.  Let's say
    the topic is partitioned by the user id so that all events for a particular
    user will go to a single consumer instance. The consumer can keep in memory
    a running tally of actions per user and only flush these out to a remote
    data store when its cache gets too big. However if a partition is reassigned
    it may want to automatically trigger a flush of this cache, before the new
    owner takes over consumption.

    This callback will execute in the user thread as part of the Consumer.poll()
    whenever partition assignment changes.

    It is guaranteed that all consumer processes will invoke
    on_partitions_revoked() prior to any process invoking
    on_partitions_assigned(). So if offsets or other state is saved in the
    on_partitions_revoked() call, it should be saved by the time the process
    taking over that partition has their on_partitions_assigned() callback
    called to load the state.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def on_partitions_revoked(self, revoked):
        if False:
            i = 10
            return i + 15
        '\n        A callback method the user can implement to provide handling of offset\n        commits to a customized store on the start of a rebalance operation.\n        This method will be called before a rebalance operation starts and\n        after the consumer stops fetching data. It is recommended that offsets\n        should be committed in this callback to either Kafka or a custom offset\n        store to prevent duplicate data.\n\n        NOTE: This method is only called before rebalances. It is not called\n        prior to KafkaConsumer.close()\n\n        Arguments:\n            revoked (list of TopicPartition): the partitions that were assigned\n                to the consumer on the last rebalance\n        '
        pass

    @abc.abstractmethod
    def on_partitions_assigned(self, assigned):
        if False:
            for i in range(10):
                print('nop')
        '\n        A callback method the user can implement to provide handling of\n        customized offsets on completion of a successful partition\n        re-assignment. This method will be called after an offset re-assignment\n        completes and before the consumer starts fetching data.\n\n        It is guaranteed that all the processes in a consumer group will execute\n        their on_partitions_revoked() callback before any instance executes its\n        on_partitions_assigned() callback.\n\n        Arguments:\n            assigned (list of TopicPartition): the partitions assigned to the\n                consumer (may include partitions that were previously assigned)\n        '
        pass
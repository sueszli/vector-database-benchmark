from queue import Queue
from airbyte_cdk.sources.streams.concurrent.partitions.partition_generator import PartitionGenerator
from airbyte_cdk.sources.streams.concurrent.partitions.types import PARTITIONS_GENERATED_SENTINEL, QueueItem

class PartitionEnqueuer:
    """
    Generates partitions from a partition generator and puts them in a queue.
    """

    def __init__(self, queue: Queue[QueueItem], sentinel: PARTITIONS_GENERATED_SENTINEL) -> None:
        if False:
            print('Hello World!')
        '\n        :param queue:  The queue to put the partitions in.\n        :param sentinel: The sentinel to put in the queue when all the partitions have been generated.\n        '
        self._queue = queue
        self._sentinel = sentinel

    def generate_partitions(self, partition_generator: PartitionGenerator) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Generate partitions from a partition generator and put them in a queue.\n        When all the partitions are added to the queue, a sentinel is added to the queue to indicate that all the partitions have been generated.\n\n        If an exception is encountered, the exception will be caught and put in the queue.\n\n        This method is meant to be called in a separate thread.\n        :param partition_generator: The partition Generator\n        :return:\n        '
        try:
            for partition in partition_generator.generate():
                self._queue.put(partition)
            self._queue.put(self._sentinel)
        except Exception as e:
            self._queue.put(e)
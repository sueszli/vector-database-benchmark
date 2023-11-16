from queue import Queue
from airbyte_cdk.sources.streams.concurrent.partitions.partition import Partition
from airbyte_cdk.sources.streams.concurrent.partitions.types import PartitionCompleteSentinel, QueueItem

class PartitionReader:
    """
    Generates records from a partition and puts them in a queue.
    """

    def __init__(self, queue: Queue[QueueItem]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param queue: The queue to put the records in.\n        '
        self._queue = queue

    def process_partition(self, partition: Partition) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Process a partition and put the records in the output queue.\n        When all the partitions are added to the queue, a sentinel is added to the queue to indicate that all the partitions have been generated.\n\n        If an exception is encountered, the exception will be caught and put in the queue.\n\n        This method is meant to be called from a thread.\n        :param partition: The partition to read data from\n        :return: None\n        '
        try:
            for record in partition.read():
                self._queue.put(record)
            self._queue.put(PartitionCompleteSentinel(partition))
        except Exception as e:
            self._queue.put(e)
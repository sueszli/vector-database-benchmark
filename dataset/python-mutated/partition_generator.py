from abc import ABC, abstractmethod
from typing import Iterable
from airbyte_cdk.sources.streams.concurrent.partitions.partition import Partition

class PartitionGenerator(ABC):

    @abstractmethod
    def generate(self) -> Iterable[Partition]:
        if False:
            i = 10
            return i + 15
        '\n        Generates partitions for a given sync mode.\n        :return: An iterable of partitions\n        '
        pass
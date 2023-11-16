from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Optional
from airbyte_cdk.sources.streams.concurrent.partitions.record import Record

class Partition(ABC):
    """
    A partition is responsible for reading a specific set of data from a source.
    """

    @abstractmethod
    def read(self) -> Iterable[Record]:
        if False:
            return 10
        '\n        Reads the data from the partition.\n        :return: An iterable of records.\n        '
        pass

    @abstractmethod
    def to_slice(self) -> Optional[Mapping[str, Any]]:
        if False:
            print('Hello World!')
        '\n        Converts the partition to a slice that can be serialized and deserialized.\n\n        Note: it would have been interesting to have a type of `Mapping[str, Comparable]` to simplify typing but some slices can have nested\n         values ([example](https://github.com/airbytehq/airbyte/blob/1ce84d6396e446e1ac2377362446e3fb94509461/airbyte-integrations/connectors/source-stripe/source_stripe/streams.py#L584-L596))\n        :return: A mapping representing a slice\n        '
        pass

    @abstractmethod
    def __hash__(self) -> int:
        if False:
            return 10
        '\n        Returns a hash of the partition.\n        Partitions must be hashable so that they can be used as keys in a dictionary.\n        '
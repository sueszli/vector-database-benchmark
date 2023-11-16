from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from airbyte_cdk.sources.declarative.types import Config, Record, StreamSlice, StreamState

@dataclass
class RecordTransformation:
    """
    Implementations of this class define transformations that can be applied to records of a stream.
    """

    @abstractmethod
    def transform(self, record: Record, config: Optional[Config]=None, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Transform a record by adding, deleting, or mutating fields.\n\n        :param record: The input record to be transformed\n        :param config: The user-provided configuration as specified by the source's spec\n        :param stream_state: The stream state\n        :param stream_slice: The stream slice\n        :return: The transformed record\n        "

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return other.__dict__ == self.__dict__
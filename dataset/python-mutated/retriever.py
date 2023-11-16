from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional
from airbyte_cdk.sources.declarative.types import StreamSlice, StreamState
from airbyte_cdk.sources.streams.core import StreamData

@dataclass
class Retriever:
    """
    Responsible for fetching a stream's records from an HTTP API source.
    """

    @abstractmethod
    def read_records(self, stream_slice: Optional[StreamSlice]=None) -> Iterable[StreamData]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Fetch a stream's records from an HTTP API source\n\n        :param sync_mode: Unused but currently necessary for integrating with HttpStream\n        :param cursor_field: Unused but currently necessary for integrating with HttpStream\n        :param stream_slice: The stream slice to read data for\n        :param stream_state: The initial stream state\n        :return: The records read from the API source\n        "

    @abstractmethod
    def stream_slices(self) -> Iterable[Optional[StreamSlice]]:
        if False:
            return 10
        'Returns the stream slices'

    @property
    @abstractmethod
    def state(self) -> StreamState:
        if False:
            return 10
        'State getter, should return state in form that can serialized to a string and send to the output\n        as a STATE AirbyteMessage.\n\n        A good example of a state is a cursor_value:\n            {\n                self.cursor_field: "cursor_value"\n            }\n\n         State should try to be as small as possible but at the same time descriptive enough to restore\n         syncing process from the point where it stopped.\n        '

    @state.setter
    @abstractmethod
    def state(self, value: StreamState) -> None:
        if False:
            for i in range(10):
                print('nop')
        'State setter, accept state serialized by state getter.'
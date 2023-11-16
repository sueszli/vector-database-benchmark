from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable
from airbyte_cdk.sources.declarative.requesters.request_options.request_options_provider import RequestOptionsProvider
from airbyte_cdk.sources.declarative.types import StreamSlice

@dataclass
class StreamSlicer(RequestOptionsProvider):
    """
    Slices the stream into a subset of records.
    Slices enable state checkpointing and data retrieval parallelization.

    The stream slicer keeps track of the cursor state as a dict of cursor_field -> cursor_value

    See the stream slicing section of the docs for more information.
    """

    @abstractmethod
    def stream_slices(self) -> Iterable[StreamSlice]:
        if False:
            print('Hello World!')
        '\n        Defines stream slices\n\n        :return: List of stream slices\n        '
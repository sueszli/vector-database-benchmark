from abc import ABC, abstractmethod
from typing import Optional
from airbyte_cdk.sources.declarative.stream_slicers.stream_slicer import StreamSlicer
from airbyte_cdk.sources.declarative.types import Record, StreamSlice, StreamState

class Cursor(ABC, StreamSlicer):
    """
    Cursors are components that allow for incremental syncs. They keep track of what data has been consumed and slices the requests based on
    that information.
    """

    @abstractmethod
    def set_initial_state(self, stream_state: StreamState) -> None:
        if False:
            while True:
                i = 10
        '\n        Cursors are not initialized with their state. As state is needed in order to function properly, this method should be called\n        before calling anything else\n\n        :param stream_state: The state of the stream as returned by get_stream_state\n        '

    @abstractmethod
    def close_slice(self, stream_slice: StreamSlice, most_recent_record: Optional[Record]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update state based on the stream slice and the latest record. Note that `stream_slice.cursor_slice` and\n        `last_record.associated_slice` are expected to be the same but we make it explicit here that `stream_slice` should be leveraged to\n        update the state.\n\n        :param stream_slice: slice to close\n        :param last_record: the latest record we have received for the slice. This is important to consider because even if the cursor emits\n          a slice, some APIs are not able to enforce the upper boundary. The outcome is that the last_record might have a higher cursor\n          value than the slice upper boundary and if we want to reduce the duplication as much as possible, we need to consider the highest\n          value between the internal cursor, the stream slice upper boundary and the record cursor value.\n        '

    @abstractmethod
    def get_stream_state(self) -> StreamState:
        if False:
            i = 10
            return i + 15
        "\n        Returns the current stream state. We would like to restrict it's usage since it does expose internal of state. As of 2023-06-14, it\n        is used for two things:\n        * Interpolation of the requests\n        * Transformation of records\n        * Saving the state\n\n        For the first case, we are probably stuck with exposing the stream state. For the second, we can probably expose a method that\n        allows for emitting the state to the platform.\n        "

    @abstractmethod
    def should_be_synced(self, record: Record) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Evaluating if a record should be synced allows for filtering and stop condition on pagination\n        '

    @abstractmethod
    def is_greater_than_or_equal(self, first: Record, second: Record) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Evaluating which record is greater in terms of cursor. This is used to avoid having to capture all the records to close a slice\n        '
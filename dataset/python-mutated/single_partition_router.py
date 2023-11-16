from dataclasses import InitVar, dataclass
from typing import Any, Iterable, Mapping, Optional
from airbyte_cdk.sources.declarative.stream_slicers.stream_slicer import StreamSlicer
from airbyte_cdk.sources.declarative.types import StreamSlice, StreamState

@dataclass
class SinglePartitionRouter(StreamSlicer):
    """Partition router returning only a stream slice"""
    parameters: InitVar[Mapping[str, Any]]

    def get_request_params(self, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        return {}

    def get_request_headers(self, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            return 10
        return {}

    def get_request_body_data(self, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            return 10
        return {}

    def get_request_body_json(self, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            return 10
        return {}

    def stream_slices(self) -> Iterable[StreamSlice]:
        if False:
            for i in range(10):
                print('nop')
        yield dict()
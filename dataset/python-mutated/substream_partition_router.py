from dataclasses import InitVar, dataclass
from typing import Any, Iterable, List, Mapping, Optional, Union
import dpath.util
from airbyte_cdk.models import AirbyteMessage, SyncMode, Type
from airbyte_cdk.sources.declarative.interpolation.interpolated_string import InterpolatedString
from airbyte_cdk.sources.declarative.requesters.request_option import RequestOption, RequestOptionType
from airbyte_cdk.sources.declarative.stream_slicers.stream_slicer import StreamSlicer
from airbyte_cdk.sources.declarative.types import Config, Record, StreamSlice, StreamState
from airbyte_cdk.sources.streams.core import Stream

@dataclass
class ParentStreamConfig:
    """
    Describes how to create a stream slice from a parent stream

    stream: The stream to read records from
    parent_key: The key of the parent stream's records that will be the stream slice key
    partition_field: The partition key
    request_option: How to inject the slice value on an outgoing HTTP request
    """
    stream: Stream
    parent_key: Union[InterpolatedString, str]
    partition_field: Union[InterpolatedString, str]
    config: Config
    parameters: InitVar[Mapping[str, Any]]
    request_option: Optional[RequestOption] = None

    def __post_init__(self, parameters: Mapping[str, Any]):
        if False:
            for i in range(10):
                print('nop')
        self.parent_key = InterpolatedString.create(self.parent_key, parameters=parameters)
        self.partition_field = InterpolatedString.create(self.partition_field, parameters=parameters)

@dataclass
class SubstreamPartitionRouter(StreamSlicer):
    """
    Partition router that iterates over the parent's stream records and emits slices
    Will populate the state with `partition_field` and `parent_slice` so they can be accessed by other components

    Attributes:
        parent_stream_configs (List[ParentStreamConfig]): parent streams to iterate over and their config
    """
    parent_stream_configs: List[ParentStreamConfig]
    config: Config
    parameters: InitVar[Mapping[str, Any]]

    def __post_init__(self, parameters: Mapping[str, Any]):
        if False:
            while True:
                i = 10
        if not self.parent_stream_configs:
            raise ValueError('SubstreamPartitionRouter needs at least 1 parent stream')
        self._parameters = parameters

    def get_request_params(self, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_request_option(RequestOptionType.request_parameter, stream_slice)

    def get_request_headers(self, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        return self._get_request_option(RequestOptionType.header, stream_slice)

    def get_request_body_data(self, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_request_option(RequestOptionType.body_data, stream_slice)

    def get_request_body_json(self, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Optional[Mapping]:
        if False:
            return 10
        return self._get_request_option(RequestOptionType.body_json, stream_slice)

    def _get_request_option(self, option_type: RequestOptionType, stream_slice: StreamSlice):
        if False:
            i = 10
            return i + 15
        params = {}
        if stream_slice:
            for parent_config in self.parent_stream_configs:
                if parent_config.request_option and parent_config.request_option.inject_into == option_type:
                    key = parent_config.partition_field.eval(self.config)
                    value = stream_slice.get(key)
                    if value:
                        params.update({parent_config.request_option.field_name: value})
        return params

    def stream_slices(self) -> Iterable[StreamSlice]:
        if False:
            i = 10
            return i + 15
        "\n        Iterate over each parent stream's record and create a StreamSlice for each record.\n\n        For each stream, iterate over its stream_slices.\n        For each stream slice, iterate over each record.\n        yield a stream slice for each such records.\n\n        If a parent slice contains no record, emit a slice with parent_record=None.\n\n        The template string can interpolate the following values:\n        - parent_stream_slice: mapping representing the parent's stream slice\n        - parent_record: mapping representing the parent record\n        - parent_stream_name: string representing the parent stream name\n        "
        if not self.parent_stream_configs:
            yield from []
        else:
            for parent_stream_config in self.parent_stream_configs:
                parent_stream = parent_stream_config.stream
                parent_field = parent_stream_config.parent_key.eval(self.config)
                stream_state_field = parent_stream_config.partition_field.eval(self.config)
                for parent_stream_slice in parent_stream.stream_slices(sync_mode=SyncMode.full_refresh, cursor_field=None, stream_state=None):
                    empty_parent_slice = True
                    parent_slice = parent_stream_slice
                    for parent_record in parent_stream.read_records(sync_mode=SyncMode.full_refresh, cursor_field=None, stream_slice=parent_stream_slice, stream_state=None):
                        if isinstance(parent_record, AirbyteMessage):
                            if parent_record.type == Type.RECORD:
                                parent_record = parent_record.record.data
                            else:
                                continue
                        elif isinstance(parent_record, Record):
                            parent_record = parent_record.data
                        try:
                            stream_state_value = dpath.util.get(parent_record, parent_field)
                        except KeyError:
                            pass
                        else:
                            empty_parent_slice = False
                            yield {stream_state_field: stream_state_value, 'parent_slice': parent_slice}
                    if empty_parent_slice:
                        yield from []
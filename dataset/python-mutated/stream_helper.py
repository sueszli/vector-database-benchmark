from typing import Any, Mapping, Optional
from airbyte_cdk.models import SyncMode
from airbyte_cdk.sources.streams.core import Stream, StreamData

def get_first_stream_slice(stream) -> Optional[Mapping[str, Any]]:
    if False:
        i = 10
        return i + 15
    "\n    Gets the first stream_slice from a given stream's stream_slices.\n    :param stream: stream\n    :raises StopIteration: if there is no first slice to return (the stream_slices generator is empty)\n    :return: first stream slice from 'stream_slices' generator (`None` is a valid stream slice)\n    "
    slices = iter(stream.stream_slices(cursor_field=stream.cursor_field, sync_mode=SyncMode.full_refresh))
    return next(slices)

def get_first_record_for_slice(stream: Stream, stream_slice: Optional[Mapping[str, Any]]) -> StreamData:
    if False:
        while True:
            i = 10
    '\n    Gets the first record for a stream_slice of a stream.\n    :param stream: stream\n    :param stream_slice: stream_slice\n    :raises StopIteration: if there is no first record to return (the read_records generator is empty)\n    :return: StreamData containing the first record in the slice\n    '
    records_for_slice = iter(stream.read_records(sync_mode=SyncMode.full_refresh, stream_slice=stream_slice))
    return next(records_for_slice)
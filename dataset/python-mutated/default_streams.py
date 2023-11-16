from typing import List, Set
from zerver.lib.types import DefaultStreamDict
from zerver.models import DefaultStream, Stream

def get_slim_realm_default_streams(realm_id: int) -> List[Stream]:
    if False:
        i = 10
        return i + 15
    return list(Stream.objects.filter(defaultstream__realm_id=realm_id))

def get_default_stream_ids_for_realm(realm_id: int) -> Set[int]:
    if False:
        while True:
            i = 10
    return set(DefaultStream.objects.filter(realm_id=realm_id).values_list('stream_id', flat=True))

def get_default_streams_for_realm_as_dicts(realm_id: int) -> List[DefaultStreamDict]:
    if False:
        while True:
            i = 10
    '\n    Return all the default streams for a realm using a list of dictionaries sorted\n    by stream name.\n    '
    streams = get_slim_realm_default_streams(realm_id)
    stream_dicts = [stream.to_dict() for stream in streams]
    return sorted(stream_dicts, key=lambda stream: stream['name'])
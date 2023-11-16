from typing import Any, Callable, Dict, Iterable, Optional
import dpath
from airbyte_cdk.models import AirbyteStream

def get_first(iterable: Iterable[Any], predicate: Callable[[Any], bool]=lambda m: True) -> Optional[Any]:
    if False:
        i = 10
        return i + 15
    return next(filter(predicate, iterable), None)

def get_defined_id(stream: AirbyteStream, data: Dict[str, Any]) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    if not stream.source_defined_primary_key:
        return None
    primary_key = []
    for key in stream.source_defined_primary_key:
        try:
            primary_key.append(str(dpath.util.get(data, key)))
        except KeyError:
            primary_key.append('__not_found__')
    return '_'.join(primary_key)
"""Simple wrapper around JSON to guarantee consistent use of load/dump. """
import json
from typing import Any, Dict, Optional
import spack.error
__all__ = ['load', 'dump', 'SpackJSONError']
_json_dump_args = {'indent': None, 'separators': (',', ':')}

def load(stream: Any) -> Dict:
    if False:
        print('Hello World!')
    'Spack JSON needs to be ordered to support specs.'
    if isinstance(stream, str):
        return json.loads(stream)
    return json.load(stream)

def dump(data: Dict, stream: Optional[Any]=None) -> Optional[str]:
    if False:
        while True:
            i = 10
    'Dump JSON with a reasonable amount of indentation and separation.'
    if stream is None:
        return json.dumps(data, **_json_dump_args)
    json.dump(data, stream, **_json_dump_args)
    return None

class SpackJSONError(spack.error.SpackError):
    """Raised when there are issues with JSON parsing."""

    def __init__(self, msg: str, json_error: BaseException):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(msg, str(json_error))
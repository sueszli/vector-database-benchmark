"""JSON serialize to/from utf8 bytes

.. versionchanged:: 22.2
    Remove optional imports of different JSON implementations.
    Now that we require recent Python, unconditionally use the standard library.
    Custom JSON libraries can be used via custom serialization functions.
"""
import json
from typing import Any, Dict, List, Union
jsonmod = json

def dumps(o: Any, **kwargs) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    'Serialize object to JSON bytes (utf-8).\n\n    Keyword arguments are passed along to :py:func:`json.dumps`.\n    '
    return json.dumps(o, **kwargs).encode('utf8')

def loads(s: Union[bytes, str], **kwargs) -> Union[Dict, List, str, int, float]:
    if False:
        while True:
            i = 10
    'Load object from JSON bytes (utf-8).\n\n    Keyword arguments are passed along to :py:func:`json.loads`.\n    '
    if isinstance(s, bytes):
        s = s.decode('utf8')
    return json.loads(s, **kwargs)
__all__ = ['dumps', 'loads']
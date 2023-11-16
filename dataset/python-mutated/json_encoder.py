""" Provide a functions and classes to implement a custom JSON encoder for
serializing objects for BokehJS.

In general, functions in this module convert values in the following way:

* Datetime values (Python, Pandas, NumPy) are converted to floating point
  milliseconds since epoch.

* TimeDelta values are converted to absolute floating point milliseconds.

* RelativeDelta values are converted to dictionaries.

* Decimal values are converted to floating point.

* Sequences (Pandas Series, NumPy arrays, python sequences) that are passed
  though this interface are converted to lists. Note, however, that arrays in
  data sources inside Bokeh Documents are converted elsewhere, and by default
  use a binary encoded format.

* Bokeh ``Model`` instances are usually serialized elsewhere in the context
  of an entire Bokeh Document. Models passed trough this interface are
  converted to references.

* ``HasProps`` (that are not Bokeh models) are converted to key/value dicts or
  all their properties and values.

* ``Color`` instances are converted to CSS color values.

.. |serialize_json| replace:: :class:`~bokeh.core.json_encoder.serialize_json`

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from json import JSONEncoder
from typing import Any
from ..settings import settings
from .serialization import Buffer, Serialized
__all__ = ('serialize_json',)

def serialize_json(obj: Any | Serialized[Any], *, pretty: bool | None=None, indent: int | None=None) -> str:
    if False:
        return 10
    '\n    Convert an object or a serialized representation to a JSON string.\n\n    This function accepts Python-serializable objects and converts them to\n    a JSON string. This function does not perform any advaced serialization,\n    in particular it won\'t serialize Bokeh models or numpy arrays. For that,\n    use :class:`bokeh.core.serialization.Serializer` class, which handles\n    serialization of all types of objects that may be encountered in Bokeh.\n\n    Args:\n        obj (obj) : the object to serialize to JSON format\n\n        pretty (bool, optional) :\n\n            Whether to generate prettified output. If ``True``, spaces are\n            added after added after separators, and indentation and newlines\n            are applied. (default: False)\n\n            Pretty output can also be enabled with the environment variable\n            ``BOKEH_PRETTY``, which overrides this argument, if set.\n\n        indent (int or None, optional) :\n\n            Amount of indentation to use in generated JSON output. If ``None``\n            then no indentation is used, unless pretty output is enabled,\n            in which case two spaces are used. (default: None)\n\n    Returns:\n\n        str: RFC-8259 JSON string\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> import numpy as np\n\n            >>> from bokeh.core.serialization import Serializer\n            >>> from bokeh.core.json_encoder import serialize_json\n\n            >>> s = Serializer()\n\n            >>> obj = dict(b=np.datetime64("2023-02-25"), a=np.arange(3))\n            >>> rep = s.encode(obj)\n            >>> rep\n            {\n                \'type\': \'map\',\n                \'entries\': [\n                    (\'b\', 1677283200000.0),\n                    (\'a\', {\n                        \'type\': \'ndarray\',\n                        \'array\': {\'type\': \'bytes\', \'data\': Buffer(id=\'p1000\', data=<memory at 0x7fe5300e2d40>)},\n                        \'shape\': [3],\n                        \'dtype\': \'int32\',\n                        \'order\': \'little\',\n                    }),\n                ],\n            }\n\n            >>> serialize_json(rep)\n            \'{"type":"map","entries":[["b",1677283200000.0],["a",{"type":"ndarray","array":\'\n            "{"type":"bytes","data":"AAAAAAEAAAACAAAA"},"shape":[3],"dtype":"int32","order":"little"}]]}\'\n\n    .. note::\n\n        Using this function isn\'t strictly necessary. The serializer can be\n        configured to produce output that\'s fully compatible with ``dumps()``\n        from the standard library module ``json``. The main difference between\n        this function and ``dumps()`` is handling of memory buffers. Use the\n        following setup:\n\n        .. code-block:: python\n\n            >>> s = Serializer(deferred=False)\n\n            >>> import json\n            >>> json.dumps(s.encode(obj))\n\n    '
    pretty = settings.pretty(pretty)
    if pretty:
        separators = (',', ': ')
    else:
        separators = (',', ':')
    if pretty and indent is None:
        indent = 2
    content: Any
    buffers: list[Buffer]
    if isinstance(obj, Serialized):
        content = obj.content
        buffers = obj.buffers or []
    else:
        content = obj
        buffers = []
    encoder = PayloadEncoder(buffers=buffers, indent=indent, separators=separators)
    return encoder.encode(content)

class PayloadEncoder(JSONEncoder):

    def __init__(self, *, buffers: list[Buffer]=[], threshold: int=100, indent: int | None=None, separators: tuple[str, str] | None=None):
        if False:
            while True:
                i = 10
        super().__init__(sort_keys=False, allow_nan=False, indent=indent, separators=separators)
        self._buffers = {buf.id: buf for buf in buffers}
        self._threshold = threshold

    def default(self, obj: Any) -> Any:
        if False:
            while True:
                i = 10
        if isinstance(obj, Buffer):
            if obj.id in self._buffers:
                return obj.ref
            else:
                return obj.to_base64()
        else:
            return super().default(obj)
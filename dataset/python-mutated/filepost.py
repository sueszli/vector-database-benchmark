from __future__ import annotations
import binascii
import codecs
import os
import typing
from io import BytesIO
from .fields import _TYPE_FIELD_VALUE_TUPLE, RequestField
writer = codecs.lookup('utf-8')[3]
_TYPE_FIELDS_SEQUENCE = typing.Sequence[typing.Union[typing.Tuple[str, _TYPE_FIELD_VALUE_TUPLE], RequestField]]
_TYPE_FIELDS = typing.Union[_TYPE_FIELDS_SEQUENCE, typing.Mapping[str, _TYPE_FIELD_VALUE_TUPLE]]

def choose_boundary() -> str:
    if False:
        return 10
    '\n    Our embarrassingly-simple replacement for mimetools.choose_boundary.\n    '
    return binascii.hexlify(os.urandom(16)).decode()

def iter_field_objects(fields: _TYPE_FIELDS) -> typing.Iterable[RequestField]:
    if False:
        i = 10
        return i + 15
    '\n    Iterate over fields.\n\n    Supports list of (k, v) tuples and dicts, and lists of\n    :class:`~urllib3.fields.RequestField`.\n\n    '
    iterable: typing.Iterable[RequestField | tuple[str, _TYPE_FIELD_VALUE_TUPLE]]
    if isinstance(fields, typing.Mapping):
        iterable = fields.items()
    else:
        iterable = fields
    for field in iterable:
        if isinstance(field, RequestField):
            yield field
        else:
            yield RequestField.from_tuples(*field)

def encode_multipart_formdata(fields: _TYPE_FIELDS, boundary: str | None=None) -> tuple[bytes, str]:
    if False:
        return 10
    '\n    Encode a dictionary of ``fields`` using the multipart/form-data MIME format.\n\n    :param fields:\n        Dictionary of fields or list of (key, :class:`~urllib3.fields.RequestField`).\n        Values are processed by :func:`urllib3.fields.RequestField.from_tuples`.\n\n    :param boundary:\n        If not specified, then a random boundary will be generated using\n        :func:`urllib3.filepost.choose_boundary`.\n    '
    body = BytesIO()
    if boundary is None:
        boundary = choose_boundary()
    for field in iter_field_objects(fields):
        body.write(f'--{boundary}\r\n'.encode('latin-1'))
        writer(body).write(field.render_headers())
        data = field.data
        if isinstance(data, int):
            data = str(data)
        if isinstance(data, str):
            writer(body).write(data)
        else:
            body.write(data)
        body.write(b'\r\n')
    body.write(f'--{boundary}--\r\n'.encode('latin-1'))
    content_type = f'multipart/form-data; boundary={boundary}'
    return (body.getvalue(), content_type)
"""
Utilities for creating permalinks.
"""
import base64
import binascii
import logging
from typing import Optional, NamedTuple
from allennlp.common.util import JsonDict
logger = logging.getLogger(__name__)
Permadata = NamedTuple('Permadata', [('model_name', str), ('request_data', JsonDict), ('response_data', JsonDict)])

def int_to_slug(i: int) -> str:
    if False:
        while True:
            i = 10
    '\n    Turn an integer id into a semi-opaque string slug\n    to use as the permalink.\n    '
    byt = str(i).encode('utf-8')
    slug_bytes = base64.urlsafe_b64encode(byt)
    return slug_bytes.decode('utf-8')

def slug_to_int(slug: str) -> Optional[int]:
    if False:
        i = 10
        return i + 15
    '\n    Convert the permalink slug back to the integer id.\n    Returns ``None`` if slug is not well-formed.\n    '
    byt = slug.encode('utf-8')
    try:
        int_bytes = base64.urlsafe_b64decode(byt)
        return int(int_bytes)
    except (binascii.Error, ValueError):
        logger.error('Unable to interpret slug: %s', slug)
        return None
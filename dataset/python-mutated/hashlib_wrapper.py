from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import ReadableBuffer
from airflow import PY39

def md5(__string: ReadableBuffer=b'') -> hashlib._Hash:
    if False:
        return 10
    '\n    Safely allows calling the ``hashlib.md5`` function when ``usedforsecurity`` is disabled in configuration.\n\n    :param __string: The data to hash. Default to empty str byte.\n    :return: The hashed value.\n    '
    if PY39:
        return hashlib.md5(__string, usedforsecurity=False)
    return hashlib.md5(__string)
"""A BSON wrapper for long (int in python3)"""
from __future__ import annotations
from typing import Any

class Int64(int):
    """Representation of the BSON int64 type.

    This is necessary because every integral number is an :class:`int` in
    Python 3. Small integral numbers are encoded to BSON int32 by default,
    but Int64 numbers will always be encoded to BSON int64.

    :Parameters:
      - `value`: the numeric value to represent
    """
    __slots__ = ()
    _type_marker = 18

    def __getstate__(self) -> Any:
        if False:
            print('Hello World!')
        return {}

    def __setstate__(self, state: Any) -> None:
        if False:
            while True:
                i = 10
        pass
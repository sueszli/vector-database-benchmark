"""Advanced options for MongoDB drivers implemented on top of PyMongo."""
from __future__ import annotations
from collections import namedtuple
from typing import Optional

class DriverInfo(namedtuple('DriverInfo', ['name', 'version', 'platform'])):
    """Info about a driver wrapping PyMongo.

    The MongoDB server logs PyMongo's name, version, and platform whenever
    PyMongo establishes a connection. A driver implemented on top of PyMongo
    can add its own info to this log message. Initialize with three strings
    like 'MyDriver', '1.2.3', 'some platform info'. Any of these strings may be
    None to accept PyMongo's default.
    """

    def __new__(cls, name: str, version: Optional[str]=None, platform: Optional[str]=None) -> DriverInfo:
        if False:
            print('Hello World!')
        self = super().__new__(cls, name, version, platform)
        for (key, value) in self._asdict().items():
            if value is not None and (not isinstance(value, str)):
                raise TypeError(f'Wrong type for DriverInfo {key} option, value must be an instance of str')
        return self
"""DBM-like dummy module"""
import collections
from typing import Any, DefaultDict

class DummyDB(dict):
    """Provide dummy DBM-like interface."""

    def close(self):
        if False:
            i = 10
            return i + 15
        pass
error = KeyError
_DATABASES: DefaultDict[Any, DummyDB] = collections.defaultdict(DummyDB)

def open(file, flag='r', mode=438):
    if False:
        return 10
    'Open or create a dummy database compatible.\n\n    Arguments ``flag`` and ``mode`` are ignored.\n    '
    return _DATABASES[file]
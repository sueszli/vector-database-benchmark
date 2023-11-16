from builtins import _test_sink, _test_source
from typing import Dict, Optional

def indexes_are_strings() -> None:
    if False:
        while True:
            i = 10
    d = {}
    d[1] = _test_source()
    _test_sink(d['1'])

class SpecialDict:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.attribute: Optional[str] = None
        self.dict: Dict[str, str] = {}

    def __getitem__(self, key: str) -> str:
        if False:
            while True:
                i = 10
        return self.dict.get(key, '')

def indexes_and_attributes():
    if False:
        print('Hello World!')
    o = SpecialDict()
    o.attribute = _test_source()
    _test_sink(o['attribute'])

def indexes_are_attributes_for___dict__():
    if False:
        for i in range(10):
            print('nop')
    o = object()
    o.attribute = _test_source()
    _test_sink(o.__dict__['attribute'])
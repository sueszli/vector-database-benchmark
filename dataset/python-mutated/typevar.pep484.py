"""TypeVar test."""
from typing import Dict, TypeVar
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class UserDict:

    def __init__(self, initialdata: Dict[_KT, _VT]=None):
        if False:
            i = 10
            return i + 15
        pass
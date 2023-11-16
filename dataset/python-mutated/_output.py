from typing import Any, MutableMapping
from ._config import _SECRET_SENTINEL

class OutputValue:
    value: Any
    secret: bool

    def __init__(self, value: Any, secret: bool):
        if False:
            while True:
                i = 10
        self.value = value
        self.secret = secret

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return _SECRET_SENTINEL if self.secret else repr(self.value)
OutputMap = MutableMapping[str, OutputValue]
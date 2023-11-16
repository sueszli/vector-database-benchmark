from __future__ import annotations
from typing import Any

class ConfigSource:

    def add_property(self, key: str, value: Any) -> None:
        if False:
            return 10
        raise NotImplementedError()

    def remove_property(self, key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()
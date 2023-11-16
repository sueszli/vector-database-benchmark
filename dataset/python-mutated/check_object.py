from __future__ import annotations
from typing import Any

class Diagnostic:

    def __reduce__(self) -> str | tuple[Any, ...]:
        if False:
            for i in range(10):
                print('nop')
        res = super().__reduce__()
        if isinstance(res, tuple) and len(res) >= 3:
            res[2]['_info'] = 42
        return res
from __future__ import annotations
from typing_extensions import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame

def example() -> DataFrame:
    if False:
        for i in range(10):
            print('nop')
    pass
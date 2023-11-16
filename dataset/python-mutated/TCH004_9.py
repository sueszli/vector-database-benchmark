from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple, List, Dict
x: Tuple

class C:
    x: List

def f():
    if False:
        for i in range(10):
            print('nop')
    x: Dict
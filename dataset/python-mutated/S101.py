assert True

def fn():
    if False:
        i = 10
        return i + 15
    x = 1
    assert x == 1
    assert x == 2
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    assert True
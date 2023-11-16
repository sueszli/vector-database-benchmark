from typing import TYPE_CHECKING, Type
if TYPE_CHECKING:
    from typing import Any

def example(*args: Any, **kwargs: Any):
    if False:
        for i in range(10):
            print('nop')
    return
my_type: Type[Any] | Any
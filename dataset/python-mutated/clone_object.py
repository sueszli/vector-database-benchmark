from typing import Callable, Any, List, Union
from ding.data.buffer import BufferedData
from ding.utils import fastcopy

def clone_object():
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        This middleware freezes the objects saved in memory buffer and return copies during sampling,\n        try this middleware when you need to keep the object unchanged in buffer, and modify        the object after sampling it (usually in multiple threads)\n    '

    def push(chain: Callable, data: Any, *args, **kwargs) -> BufferedData:
        if False:
            return 10
        data = fastcopy.copy(data)
        return chain(data, *args, **kwargs)

    def sample(chain: Callable, *args, **kwargs) -> Union[List[BufferedData], List[List[BufferedData]]]:
        if False:
            return 10
        data = chain(*args, **kwargs)
        return fastcopy.copy(data)

    def _clone_object(action: str, chain: Callable, *args, **kwargs):
        if False:
            return 10
        if action == 'push':
            return push(chain, *args, **kwargs)
        elif action == 'sample':
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)
    return _clone_object
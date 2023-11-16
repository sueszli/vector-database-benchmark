from typing import Optional, Callable, List, Any, Iterable
import torch

def example_get_data_fn() -> Any:
    if False:
        return 10
    '\n    Note: staticmethod or static function, all the operation is on CPU\n    '
    pass

class IDataLoader:

    def __next__(self, batch_size: Optional[int]=None) -> torch.Tensor:
        if False:
            return 10
        '\n        Arguments:\n            batch_size: sometimes, batch_size is specified by each iteration, if batch_size is None,\n                use default batch_size value\n        '
        if batch_size is None:
            batch_size = self._batch_size
        data = self._get_data(batch_size)
        return self._collate_fn(data)

    def __iter__(self) -> Iterable:
        if False:
            print('Hello World!')
        return self

    def _get_data(self, batch_size: Optional[int]=None) -> List[torch.Tensor]:
        if False:
            return 10
        raise NotImplementedError

    def close(self) -> None:
        if False:
            while True:
                i = 10
        pass
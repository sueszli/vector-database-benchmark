from typing import Any, Dict, List
from allennlp.data.fields.field import Field

class FlagField(Field[Any]):
    """
    A class representing a flag, which must be constant across all instances in a batch.
    This will be passed to a `forward` method as a single value of whatever type you pass in.
    """
    __slots__ = ['flag_value']

    def __init__(self, flag_value: Any) -> None:
        if False:
            while True:
                i = 10
        self.flag_value = flag_value

    def get_padding_lengths(self) -> Dict[str, int]:
        if False:
            print('Hello World!')
        return {}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> Any:
        if False:
            print('Hello World!')
        return self.flag_value

    def empty_field(self):
        if False:
            i = 10
            return i + 15
        return FlagField(self.flag_value)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'FlagField({self.flag_value})'

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return 1

    def batch_tensors(self, tensor_list: List[Any]) -> Any:
        if False:
            print('Hello World!')
        if len(set(tensor_list)) != 1:
            raise ValueError(f'Got different values in a FlagField when trying to batch them: {tensor_list}')
        return tensor_list[0]

    def human_readable_repr(self) -> Any:
        if False:
            print('Hello World!')
        if hasattr(self.flag_value, 'human_readable_repr'):
            return self.flag_value.human_readable_repr()
        return self.flag_value
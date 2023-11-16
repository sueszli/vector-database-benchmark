from typing import Dict, List, Iterator, Sequence, Any
from allennlp.data.fields.field import DataArray, Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.util import pad_sequence_to_length

class ListField(SequenceField[DataArray]):
    """
    A `ListField` is a list of other fields.  You would use this to represent, e.g., a list of
    answer options that are themselves `TextFields`.

    This field will get converted into a tensor that has one more mode than the items in the list.
    If this is a list of `TextFields` that have shape (num_words, num_characters), this
    `ListField` will output a tensor of shape (num_sentences, num_words, num_characters).

    # Parameters

    field_list : `List[Field]`
        A list of `Field` objects to be concatenated into a single input tensor.  All of the
        contained `Field` objects must be of the same type.
    """
    __slots__ = ['field_list']

    def __init__(self, field_list: Sequence[Field]) -> None:
        if False:
            print('Hello World!')
        field_class_set = {field.__class__ for field in field_list}
        assert len(field_class_set) == 1, 'ListFields must contain a single field type, found ' + str(field_class_set)
        self.field_list = field_list

    def __iter__(self) -> Iterator[Field]:
        if False:
            return 10
        return iter(self.field_list)

    def __getitem__(self, idx: int) -> Field:
        if False:
            i = 10
            return i + 15
        return self.field_list[idx]

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self.field_list)

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if False:
            for i in range(10):
                print('nop')
        for field in self.field_list:
            field.count_vocab_items(counter)

    def index(self, vocab: Vocabulary):
        if False:
            i = 10
            return i + 15
        for field in self.field_list:
            field.index(vocab)

    def get_padding_lengths(self) -> Dict[str, int]:
        if False:
            i = 10
            return i + 15
        field_lengths = [field.get_padding_lengths() for field in self.field_list]
        padding_lengths = {'num_fields': len(self.field_list)}
        possible_padding_keys = [key for field_length in field_lengths for key in list(field_length.keys())]
        for key in set(possible_padding_keys):
            padding_lengths['list_' + key] = max((x[key] if key in x else 0 for x in field_lengths))
        for padding_key in padding_lengths:
            padding_lengths[padding_key] = max(padding_lengths[padding_key], 1)
        return padding_lengths

    def sequence_length(self) -> int:
        if False:
            print('Hello World!')
        return len(self.field_list)

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        if False:
            print('Hello World!')
        padded_field_list = pad_sequence_to_length(self.field_list, padding_lengths['num_fields'], self.field_list[0].empty_field)
        child_padding_lengths = {key.replace('list_', '', 1): value for (key, value) in padding_lengths.items() if key.startswith('list_')}
        padded_fields = [field.as_tensor(child_padding_lengths) for field in padded_field_list]
        return self.field_list[0].batch_tensors(padded_fields)

    def empty_field(self):
        if False:
            i = 10
            return i + 15
        return ListField([self.field_list[0].empty_field()])

    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:
        if False:
            print('Hello World!')
        return self.field_list[0].batch_tensors(tensor_list)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        field_class = self.field_list[0].__class__.__name__
        base_string = f'ListField of {len(self.field_list)} {field_class}s : \n'
        return ' '.join([base_string] + [f'\t {field} \n' for field in self.field_list])

    def human_readable_repr(self) -> List[Any]:
        if False:
            return 10
        return [f.human_readable_repr() for f in self.field_list]
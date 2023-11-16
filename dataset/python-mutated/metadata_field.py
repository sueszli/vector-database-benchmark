from typing import Any, Dict, List, Mapping
from allennlp.data.fields.field import DataArray, Field

class MetadataField(Field[DataArray], Mapping[str, Any]):
    """
    A `MetadataField` is a `Field` that does not get converted into tensors.  It just carries
    side information that might be needed later on, for computing some third-party metric, or
    outputting debugging information, or whatever else you need.  We use this in the BiDAF model,
    for instance, to keep track of question IDs and passage token offsets, so we can more easily
    use the official evaluation script to compute metrics.

    We don't try to do any kind of smart combination of this field for batched input - when you use
    this `Field` in a model, you'll get a list of metadata objects, one for each instance in the
    batch.

    # Parameters

    metadata : `Any`
        Some object containing the metadata that you want to store.  It's likely that you'll want
        this to be a dictionary, but it could be anything you want.
    """
    __slots__ = ['metadata']

    def __init__(self, metadata: Any) -> None:
        if False:
            print('Hello World!')
        self.metadata = metadata

    def __getitem__(self, key: str) -> Any:
        if False:
            i = 10
            return i + 15
        try:
            return self.metadata[key]
        except TypeError:
            raise TypeError('your metadata is not a dict')

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return iter(self.metadata)
        except TypeError:
            raise TypeError('your metadata is not iterable')

    def __len__(self):
        if False:
            i = 10
            return i + 15
        try:
            return len(self.metadata)
        except TypeError:
            raise TypeError('your metadata has no length')

    def get_padding_lengths(self) -> Dict[str, int]:
        if False:
            while True:
                i = 10
        return {}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        if False:
            print('Hello World!')
        return self.metadata

    def empty_field(self) -> 'MetadataField':
        if False:
            for i in range(10):
                print('nop')
        return MetadataField(None)

    def batch_tensors(self, tensor_list: List[DataArray]) -> List[DataArray]:
        if False:
            return 10
        return tensor_list

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'MetadataField (print field.metadata to see specific information).'

    def human_readable_repr(self):
        if False:
            while True:
                i = 10
        if hasattr(self.metadata, 'human_readable_repr'):
            return self.metadata.human_readable_repr()
        return self.metadata
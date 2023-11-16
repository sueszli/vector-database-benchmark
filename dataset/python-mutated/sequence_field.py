from allennlp.data.fields.field import DataArray, Field

class SequenceField(Field[DataArray]):
    """
    A `SequenceField` represents a sequence of things.  This class just adds a method onto
    `Field`: :func:`sequence_length`.  It exists so that `SequenceLabelField`, `IndexField` and other
    similar `Fields` can have a single type to require, with a consistent API, whether they are
    pointing to words in a `TextField`, items in a `ListField`, or something else.
    """
    __slots__ = []

    def sequence_length(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        How many elements are there in this sequence?\n        '
        raise NotImplementedError

    def empty_field(self) -> 'SequenceField':
        if False:
            print('Hello World!')
        raise NotImplementedError
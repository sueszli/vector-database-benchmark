"""The implementation of `tf.data.Dataset.zip`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.types import data as data_types

def _zip(datasets, name):
    if False:
        for i in range(10):
            print('nop')
    return _ZipDataset(datasets, name)

class _ZipDataset(dataset_ops.DatasetV2):
    """A `Dataset` that zips its inputs together."""

    def __init__(self, datasets, name=None):
        if False:
            return 10
        'See `Dataset.zip()` for details.'
        for ds in nest.flatten(datasets):
            if not isinstance(ds, data_types.DatasetV2):
                if isinstance(ds, list):
                    raise TypeError('Invalid input to `zip`. Inputs are expected to be (nested) structures of `tf.data.Dataset` objects. Python `list` is not supported and you should use `tuple` instead.')
                else:
                    raise TypeError(f'Invalid input to `zip`. Inputs are expected to be (nested) structures of `tf.data.Dataset` objects but encountered object of type {type(ds)}.')
        self._datasets = datasets
        self._structure = nest.pack_sequence_as(self._datasets, [ds.element_spec for ds in nest.flatten(self._datasets)])
        self._name = name
        variant_tensor = gen_dataset_ops.zip_dataset([ds._variant_tensor for ds in nest.flatten(self._datasets)], **self._common_args)
        super().__init__(variant_tensor)

    def _inputs(self):
        if False:
            return 10
        return nest.flatten(self._datasets)

    @property
    def element_spec(self):
        if False:
            return 10
        return self._structure
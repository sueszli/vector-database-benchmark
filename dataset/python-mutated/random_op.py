"""The implementation of `tf.data.Dataset.random`."""
import warnings
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops

def _random(seed=None, rerandomize_each_iteration=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'See `Dataset.random()` for details.'
    return _RandomDataset(seed=seed, rerandomize_each_iteration=rerandomize_each_iteration, name=name)

class _RandomDataset(dataset_ops.DatasetSource):
    """A `Dataset` of pseudorandom values."""

    def __init__(self, seed=None, rerandomize_each_iteration=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        'A `Dataset` of pseudorandom values.'
        (self._seed, self._seed2) = random_seed.get_seed(seed)
        self._rerandomize = rerandomize_each_iteration
        self._name = name
        if rerandomize_each_iteration:
            if not tf2.enabled():
                warnings.warn('In TF 1, the `rerandomize_each_iteration=True` option is only supported for repeat-based epochs.')
            variant_tensor = ged_ops.random_dataset_v2(seed=self._seed, seed2=self._seed2, seed_generator=gen_dataset_ops.dummy_seed_generator(), rerandomize_each_iteration=self._rerandomize, **self._common_args)
        else:
            variant_tensor = ged_ops.random_dataset(seed=self._seed, seed2=self._seed2, **self._common_args)
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        if False:
            while True:
                i = 10
        return tensor_spec.TensorSpec([], dtypes.int64)
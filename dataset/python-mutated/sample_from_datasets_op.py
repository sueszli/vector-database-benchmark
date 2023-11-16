"""The implementation of `tf.data.Dataset.sample_from_datasets`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import directed_interleave_op
from tensorflow.python.data.ops import map_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.types import data as data_types

def _sample_from_datasets(datasets, weights=None, seed=None, stop_on_empty_dataset=False, rerandomize_each_iteration=None):
    if False:
        return 10
    'See `Dataset.sample_from_datasets()` for details.'

    def _skip_datasets_with_zero_weight(datasets, weights):
        if False:
            while True:
                i = 10
        datasets_and_weights = [(dataset, weight) for (dataset, weight) in zip(datasets, weights) if weight > 0]
        return zip(*datasets_and_weights) if datasets_and_weights else ([datasets[0].take(0)], [1.0])
    if not datasets:
        raise ValueError('Invalid `datasets`. `datasets` should not be empty.')
    if not isinstance(weights, data_types.DatasetV2):
        if weights is None:
            logits = [[1.0] * len(datasets)]
        else:
            if isinstance(weights, tensor.Tensor):
                if not weights.shape.is_compatible_with([len(datasets)]):
                    raise ValueError(f'Invalid `weights`. The shape of `weights` should be compatible with `[len(datasets)]` but is {weights.shape}.')
            elif len(datasets) != len(weights):
                raise ValueError(f'Invalid `weights`. `weights` should have the same length as `datasets` but got `len(weights)={len(weights)}` vs. `len(datasets)={len(datasets)}`.')
            if not isinstance(weights, tensor.Tensor):
                (datasets, weights) = _skip_datasets_with_zero_weight(datasets, weights)
            weights = ops.convert_to_tensor(weights, name='weights')
            if weights.dtype not in (dtypes.float32, dtypes.float64):
                raise TypeError(f'Invalid `weights`. `weights` type must be either `tf.float32` or `tf.float64` but is {weights.dtype}.')
            logits = array_ops.expand_dims(math_ops.log(weights, name='logits'), 0)
        if len(datasets) == 1:
            return datasets[0]

        def select_dataset_constant_logits(seed):
            if False:
                print('Hello World!')
            return array_ops.squeeze(gen_stateless_random_ops.stateless_multinomial(logits, 1, seed=seed), axis=[0, 1])
        selector_input = map_op._MapDataset(dataset_ops.Dataset.random(seed=seed, rerandomize_each_iteration=rerandomize_each_iteration).batch(2), select_dataset_constant_logits, use_inter_op_parallelism=False)
    else:
        logits_ds = weights.map(lambda *p: math_ops.log(p, name='logits'))

        def select_dataset_varying_logits(logits, seed):
            if False:
                for i in range(10):
                    print('nop')
            return array_ops.squeeze(gen_stateless_random_ops.stateless_multinomial(logits, 1, seed=seed), axis=[0, 1])
        logits_and_seeds = dataset_ops.Dataset.zip((logits_ds, dataset_ops.Dataset.random(seed=seed, rerandomize_each_iteration=rerandomize_each_iteration).batch(2)))
        selector_input = map_op._MapDataset(logits_and_seeds, select_dataset_varying_logits, use_inter_op_parallelism=False)
    return directed_interleave_op._directed_interleave(selector_input, datasets, stop_on_empty_dataset)
import tree
from keras.trainers.data_adapters import data_adapter_utils
from keras.trainers.data_adapters.data_adapter import DataAdapter

class TFDatasetAdapter(DataAdapter):
    """Adapter that handles `tf.data.Dataset`."""

    def __init__(self, dataset, class_weight=None, distribution=None):
        if False:
            while True:
                i = 10
        'Iniitialize the TFDatasetAdapter.\n\n        Args:\n            dataset: The input `tf.data.Dataset` instance.\n            class_weight: A map where the keys are integer class ids and values\n                are the class weights, e.g. `{0: 0.2, 1: 0.6, 2: 0.3}`.\n            distribution: A `keras.distribution.Distribution` instance. Used to\n                shard the input dataset into per worker/process dataset\n                instance.\n        '
        from keras.utils.module_utils import tensorflow as tf
        if not isinstance(dataset, (tf.data.Dataset, tf.distribute.DistributedDataset)):
            raise ValueError(f'Expected argument `dataset` to be a tf.data.Dataset. Received: {dataset}')
        if class_weight is not None:
            dataset = dataset.map(make_class_weight_map_fn(class_weight)).prefetch(tf.data.AUTOTUNE)
        if distribution is not None:
            dataset = distribution.distribute_dataset(dataset)
        self._dataset = dataset

    def get_numpy_iterator(self):
        if False:
            i = 10
            return i + 15
        from keras.utils.module_utils import tensorflow as tf

        def convert_to_numpy(x):
            if False:
                print('Hello World!')
            if isinstance(x, tf.SparseTensor):
                x = tf.sparse.to_dense(x)
            return x.numpy()
        for batch in self._dataset:
            yield tree.map_structure(convert_to_numpy, batch)

    def get_tf_dataset(self):
        if False:
            while True:
                i = 10
        return self._dataset

    @property
    def num_batches(self):
        if False:
            return 10
        cardinality = self._dataset.cardinality
        if callable(cardinality):
            cardinality = int(self._dataset.cardinality())
        else:
            cardinality = int(cardinality)
        if cardinality < 0:
            return None
        return cardinality

    @property
    def batch_size(self):
        if False:
            return 10
        first_element_spec = tree.flatten(self._dataset.element_spec)[0]
        return first_element_spec.shape[0]

    @property
    def has_partial_batch(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    @property
    def partial_batch_size(self):
        if False:
            for i in range(10):
                print('nop')
        return None

def make_class_weight_map_fn(class_weight):
    if False:
        return 10
    'Applies class weighting to a `Dataset`.\n\n    The `Dataset` is assumed to be in format `(x, y)` or `(x, y, sw)`, where\n    `y` must be a single `Tensor`.\n\n    Args:\n        class_weight: A map where the keys are integer class ids and values are\n            the class weights, e.g. `{0: 0.2, 1: 0.6, 2: 0.3}`\n\n    Returns:\n        A function that can be used with `tf.data.Dataset.map` to apply class\n        weighting.\n    '
    from keras.utils.module_utils import tensorflow as tf
    class_weight_tensor = tf.convert_to_tensor([class_weight.get(int(c), 1.0) for c in range(max(class_weight.keys()) + 1)])

    def class_weights_map_fn(*data):
        if False:
            print('Hello World!')
        'Convert `class_weight` to `sample_weight`.'
        (x, y, sw) = data_adapter_utils.unpack_x_y_sample_weight(data)
        if sw is not None:
            raise ValueError('You cannot `class_weight` and `sample_weight` at the same time.')
        if tree.is_nested(y):
            raise ValueError('`class_weight` is only supported for Models with a single output.')
        if y.shape.rank >= 2:
            y_classes = tf.__internal__.smart_cond.smart_cond(tf.shape(y)[-1] > 1, lambda : tf.argmax(y, axis=-1), lambda : tf.cast(tf.round(tf.squeeze(y, axis=-1)), tf.int32))
        else:
            y_classes = tf.cast(tf.round(y), tf.int32)
        cw = tf.gather(class_weight_tensor, y_classes)
        return (x, y, cw)
    return class_weights_map_fn
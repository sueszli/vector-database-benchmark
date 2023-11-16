import tensorflow as tf
from ray.util.annotations import PublicAPI

@PublicAPI(stability='beta')
def prepare_dataset_shard(tf_dataset_shard: tf.data.Dataset):
    if False:
        while True:
            i = 10
    'A utility function that overrides default config for Tensorflow Dataset.\n\n    This should be used on a TensorFlow ``Dataset`` created by calling\n    ``iter_tf_batches()`` on a ``ray.data.Dataset`` returned by\n    ``ray.train.get_dataset_shard()`` since the dataset has already\n    been sharded across the workers.\n\n    Args:\n        tf_dataset_shard (tf.data.Dataset): A TensorFlow Dataset.\n\n    Returns:\n        A TensorFlow Dataset with:\n            - autosharding turned off\n            - prefetching turned on with autotune enabled\n    '
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    return tf_dataset_shard.with_options(options).prefetch(tf.data.AUTOTUNE)
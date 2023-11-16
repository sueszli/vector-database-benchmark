import random
import numpy as np
from keras import backend
from keras.api_export import keras_export
from keras.utils.module_utils import tensorflow as tf

@keras_export('keras.utils.set_random_seed')
def set_random_seed(seed):
    if False:
        i = 10
        return i + 15
    "Sets all random seeds (Python, NumPy, and backend framework, e.g. TF).\n\n    You can use this utility to make almost any Keras program fully\n    deterministic. Some limitations apply in cases where network communications\n    are involved (e.g. parameter server distribution), which creates additional\n    sources of randomness, or when certain non-deterministic cuDNN ops are\n    involved.\n\n    Calling this utility is equivalent to the following:\n\n    ```python\n    import random\n    import numpy as np\n    from keras.utils.module_utils import tensorflow as tf\n    random.seed(seed)\n    np.random.seed(seed)\n    tf.random.set_seed(seed)\n    ```\n\n    Note that the TensorFlow seed is set even if you're not using TensorFlow\n    as your backend framework, since many workflows leverage `tf.data`\n    pipelines (which feature random shuffling). Likewise many workflows\n    might leverage NumPy APIs.\n\n    Arguments:\n        seed: Integer, the random seed to use.\n    "
    if not isinstance(seed, int):
        raise ValueError(f'Expected `seed` argument to be an integer. Received: seed={seed} (of type {type(seed)})')
    random.seed(seed)
    np.random.seed(seed)
    if tf.available:
        tf.random.set_seed(seed)
    if backend.backend() == 'torch':
        import torch
        torch.manual_seed(seed)
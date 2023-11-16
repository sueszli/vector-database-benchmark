import random as python_random
import numpy as np
from keras import backend
from keras.api_export import keras_export
from keras.backend.common import global_state
from keras.utils import jax_utils

@keras_export('keras.random.SeedGenerator')
class SeedGenerator:
    """Generates variable seeds upon each call to a RNG-using function.

    In Keras, all RNG-using methods (such as `keras.random.normal()`)
    are stateless, meaning that if you pass an integer seed to them
    (such as `seed=42`), they will return the same values at each call.
    In order to get different values at each call, you must use a
    `SeedGenerator` instead as the seed argument. The `SeedGenerator`
    object is stateful.

    Example:

    ```python
    seed_gen = keras.random.SeedGenerator(seed=42)
    values = keras.random.normal(shape=(2, 3), seed=seed_gen)
    new_values = keras.random.normal(shape=(2, 3), seed=seed_gen)
    ```

    Usage in a layer:

    ```python
    class Dropout(keras.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, x, training=False):
            if training:
                return keras.random.dropout(
                    x, rate=0.5, seed=self.seed_generator
                )
            return x
    ```
    """

    def __init__(self, seed=None, **kwargs):
        if False:
            return 10
        custom_backend = kwargs.pop('backend', None)
        if kwargs:
            raise ValueError(f'Unrecognized keyword arguments: {kwargs}')
        if custom_backend is not None:
            self.backend = custom_backend
        else:
            self.backend = backend
        self._initial_seed = seed
        if seed is None:
            seed = make_default_seed()
        if not isinstance(seed, int):
            raise ValueError(f'Argument `seed` must be an integer. Received: seed={seed}')

        def seed_initializer(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            dtype = kwargs.get('dtype', None)
            return self.backend.convert_to_tensor([seed, 0], dtype=dtype)
        self.state = self.backend.Variable(seed_initializer, shape=(2,), dtype='uint32', trainable=False, name='seed_generator_state')

    def next(self, ordered=True):
        if False:
            i = 10
            return i + 15
        seed_state = self.state
        new_seed_value = seed_state.value * 1
        if ordered:
            increment = self.backend.convert_to_tensor(np.array([0, 1]), dtype='uint32')
            self.state.assign(seed_state + increment)
        else:
            self.state.assign((seed_state + 1) * 5387 % 933199)
        return new_seed_value

def global_seed_generator():
    if False:
        while True:
            i = 10
    if jax_utils.is_in_jax_tracing_scope():
        raise ValueError('When tracing a JAX function, you should only use seeded random ops, e.g. you should create a `SeedGenerator` instance, attach it to your layer/model, and pass the instance as the `seed` argument when calling random ops. Unseeded random ops would get incorrectly traced by JAX and would become constant after tracing.')
    gen = global_state.get_global_attribute('global_seed_generator')
    if gen is None:
        gen = SeedGenerator()
        global_state.set_global_attribute('global_seed_generator', gen)
    return gen

def make_default_seed():
    if False:
        while True:
            i = 10
    return python_random.randint(1, int(1000000000.0))

def draw_seed(seed):
    if False:
        i = 10
        return i + 15
    from keras.backend import convert_to_tensor
    if isinstance(seed, SeedGenerator):
        return seed.next()
    elif isinstance(seed, int):
        return convert_to_tensor([seed, 0], dtype='uint32')
    elif seed is None:
        return global_seed_generator().next(ordered=False)
    raise ValueError(f'Argument `seed` must be either an integer or an instance of `SeedGenerator`. Received: seed={seed} (of type {type(seed)})')
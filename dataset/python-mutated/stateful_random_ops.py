"""Operations for generating random numbers."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
UINT64_HALF_SPAN = 2 ** 63
MAX_INT64 = UINT64_HALF_SPAN - 1
MIN_INT64 = -UINT64_HALF_SPAN
UINT64_SPAN = UINT64_HALF_SPAN * 2
SEED_TYPE = 'int64'
SEED_MIN = MIN_INT64
SEED_MAX = MAX_INT64
SEED_UINT_SPAN = UINT64_SPAN
SEED_TYPE_BITS = 64
SEED_BIT_MASK = 18446744073709551615
SEED_SIZE = 16
STATE_TYPE = SEED_TYPE
ALGORITHM_TYPE = STATE_TYPE
PHILOX_KEY_SIZE = 1
THREEFRY_KEY_SIZE = 1
PHILOX_COUNTER_SIZE = 2
THREEFRY_COUNTER_SIZE = 1
PHILOX_STATE_SIZE = PHILOX_COUNTER_SIZE + PHILOX_KEY_SIZE
THREEFRY_STATE_SIZE = THREEFRY_COUNTER_SIZE + THREEFRY_KEY_SIZE
RNG_ALG_PHILOX = random_ops_util.Algorithm.PHILOX.value
RNG_ALG_THREEFRY = random_ops_util.Algorithm.THREEFRY.value
DEFAULT_ALGORITHM = RNG_ALG_PHILOX

def non_deterministic_ints(shape, dtype=dtypes.int64):
    if False:
        i = 10
        return i + 15
    'Non-deterministically generates some integers.\n\n  This op may use some OS-provided source of non-determinism (e.g. an RNG), so\n  each execution will give different results.\n\n  Args:\n    shape: the shape of the result.\n    dtype: (optional) the dtype of the result.\n\n  Returns:\n    a tensor whose element values are non-deterministically chosen.\n  '
    return gen_stateful_random_ops.non_deterministic_ints(shape=shape, dtype=dtype)

def _uint_to_int(n):
    if False:
        while True:
            i = 10
    if isinstance(n, int) and n > SEED_MAX:
        n = n - SEED_UINT_SPAN
    return n

def _make_1d_state(state_size, seed):
    if False:
        return 10
    'Makes a 1-D RNG state.\n\n  Args:\n    state_size: an integer.\n    seed: an integer or 1-D tensor.\n\n  Returns:\n    a 1-D tensor of shape [state_size] and dtype STATE_TYPE.\n  '
    if isinstance(seed, int):
        ls = []
        for _ in range(state_size):
            ls.append(seed & SEED_BIT_MASK)
            seed >>= SEED_TYPE_BITS
        seed = ls
    seed = nest.map_structure(_uint_to_int, seed)
    seed = math_ops.cast(seed, STATE_TYPE)
    seed = array_ops.reshape(seed, [-1])
    seed = seed[0:state_size]
    seed_size = seed.shape[0]
    if seed_size is None:
        seed_size = array_ops.shape(seed)[0]
    padding_size = math_ops.maximum(state_size - seed_size, 0)
    padding = array_ops.zeros([padding_size], seed.dtype)
    seed = array_ops.concat([padding, seed], axis=0)
    seed.set_shape([state_size])
    return seed

def _get_counter_size(alg):
    if False:
        while True:
            i = 10
    if alg == random_ops_util.Algorithm.PHILOX.value:
        return PHILOX_COUNTER_SIZE
    elif alg == random_ops_util.Algorithm.THREEFRY.value:
        return THREEFRY_COUNTER_SIZE
    elif alg == random_ops_util.Algorithm.AUTO_SELECT.value:
        return PHILOX_COUNTER_SIZE
    else:
        raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))

def _get_state_size(alg):
    if False:
        print('Hello World!')
    if alg == random_ops_util.Algorithm.PHILOX.value:
        return PHILOX_STATE_SIZE
    elif alg == random_ops_util.Algorithm.THREEFRY.value:
        return THREEFRY_STATE_SIZE
    elif alg == random_ops_util.Algorithm.AUTO_SELECT.value:
        return PHILOX_STATE_SIZE
    else:
        raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))

def _check_state_shape(shape, alg):
    if False:
        return 10
    if isinstance(alg, tensor.Tensor) and (not context.executing_eagerly()):
        return
    shape.assert_is_compatible_with([_get_state_size(int(alg))])

def _make_state_from_seed(seed, alg):
    if False:
        while True:
            i = 10
    return _make_1d_state(_get_state_size(alg), seed)

@tf_export('random.create_rng_state', 'random.experimental.create_rng_state')
def create_rng_state(seed, alg):
    if False:
        while True:
            i = 10
    'Creates a RNG state from an integer or a vector.\n\n  Example:\n\n  >>> tf.random.create_rng_state(\n  ...     1234, "philox")\n  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([1234,    0,    0])>\n  >>> tf.random.create_rng_state(\n  ...     [12, 34], "threefry")\n  <tf.Tensor: shape=(2,), dtype=int64, numpy=array([12, 34])>\n\n  Args:\n    seed: an integer or 1-D numpy array.\n    alg: the RNG algorithm. Can be a string, an `Algorithm` or an integer.\n\n  Returns:\n    a 1-D numpy array whose size depends on the algorithm.\n  '
    alg = random_ops_util.convert_alg_to_int(alg)
    return _make_state_from_seed(seed, alg)

def _shape_tensor(shape):
    if False:
        i = 10
        return i + 15
    'Convert to an int32 or int64 tensor, defaulting to int64 if empty.'
    if isinstance(shape, (tuple, list)) and (not shape):
        dtype = dtypes.int64
    else:
        dtype = None
    return ops.convert_to_tensor(shape, dtype=dtype, name='shape')

def _convert_to_state_tensor(t):
    if False:
        return 10
    t = nest.map_structure(_uint_to_int, t)
    return math_ops.cast(t, STATE_TYPE)

def get_replica_id():
    if False:
        for i in range(10):
            print('nop')
    rctx = distribute_lib.get_replica_context()
    if rctx is None:
        return None
    return rctx.replica_id_in_sync_group

@tf_export('random.Generator', 'random.experimental.Generator')
class Generator(autotrackable.AutoTrackable):
    """Random-number generator.

  Example:

  Creating a generator from a seed:

  >>> g = tf.random.Generator.from_seed(1234)
  >>> g.normal(shape=(2, 3))
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
  array([[ 0.9356609 ,  1.0854305 , -0.93788373],
         [-0.5061547 ,  1.3169702 ,  0.7137579 ]], dtype=float32)>

  Creating a generator from a non-deterministic state:

  >>> g = tf.random.Generator.from_non_deterministic_state()
  >>> g.normal(shape=(2, 3))
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=...>

  All the constructors allow explicitly choosing an Random-Number-Generation
  (RNG) algorithm. Supported algorithms are `"philox"` and `"threefry"`. For
  example:

  >>> g = tf.random.Generator.from_seed(123, alg="philox")
  >>> g.normal(shape=(2, 3))
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
  array([[ 0.8673864 , -0.29899067, -0.9310337 ],
         [-1.5828488 ,  1.2481191 , -0.6770643 ]], dtype=float32)>

  CPU, GPU and TPU with the same algorithm and seed will generate the same
  integer random numbers. Float-point results (such as the output of `normal`)
  may have small numerical discrepancies between different devices.

  This class uses a `tf.Variable` to manage its internal state. Every time
  random numbers are generated, the state of the generator will change. For
  example:

  >>> g = tf.random.Generator.from_seed(1234)
  >>> g.state
  <tf.Variable ... numpy=array([1234,    0,    0])>
  >>> g.normal(shape=(2, 3))
  <...>
  >>> g.state
  <tf.Variable ... numpy=array([2770,    0,    0])>

  The shape of the state is algorithm-specific.

  There is also a global generator:

  >>> g = tf.random.get_global_generator()
  >>> g.normal(shape=(2, 3))
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=...>

  When creating a generator inside a `tf.distribute.Strategy` scope, each
  replica will get a different stream of random numbers.

  For example, in this code:

  ```
  strat = tf.distribute.MirroredStrategy(devices=["cpu:0", "cpu:1"])
  with strat.scope():
    g = tf.random.Generator.from_seed(1)
    def f():
      return g.normal([])
    results = strat.run(f).values
  ```

  `results[0]` and `results[1]` will have different values.

  If the generator is seeded (e.g. created via `Generator.from_seed`), the
  random numbers will be determined by the seed, even though different replicas
  get different numbers.  One can think of a random number generated on a
  replica as a hash of the replica ID and a "master" random number that may be
  common to all replicas. Hence, the whole system is still deterministic.

  (Note that the random numbers on different replicas are not correlated, even
  if they are deterministically determined by the same seed. They are not
  correlated in the sense that no matter what statistics one calculates on them,
  there won't be any discernable correlation.)

  Generators can be freely saved and restored using `tf.train.Checkpoint`. The
  checkpoint can be restored in a distribution strategy with a different number
  of replicas than the original strategy. If a replica ID is present in both the
  original and the new distribution strategy, its state will be properly
  restored (i.e. the random-number stream from the restored point will be the
  same as that from the saving point) unless the replicas have already diverged
  in their RNG call traces before saving (e.g. one replica has made one RNG call
  while another has made two RNG calls). We don't have such guarantee if the
  generator is saved in a strategy scope and restored outside of any strategy
  scope, or vice versa.

  When a generator is created within the scope of
  `tf.distribute.experimental.ParameterServerStrategy`, the workers
  will share the generator's state (placed on one of the parameter
  servers). In this way the workers will still get different
  random-number streams, as stated above. (This is similar to replicas
  in a `tf.distribute.MirroredStrategy` sequentially accessing a
  generator created outside the strategy.) Each RNG call on a worker
  will incur a round-trip to a parameter server, which may have
  performance impacts. When creating a
  `tf.distribute.experimental.ParameterServerStrategy`, please make
  sure that the `variable_partitioner` argument won't shard small
  variables of shape `[2]` or `[3]` (because generator states must not
  be sharded). Ways to avoid sharding small variables include setting
  `variable_partitioner` to `None` or to
  `tf.distribute.experimental.partitioners.MinSizePartitioner` with a
  large enough `min_shard_bytes` (see
  `tf.distribute.experimental.ParameterServerStrategy`'s documentation
  for more details).
  """

    @classmethod
    def from_state(cls, state, alg):
        if False:
            while True:
                i = 10
        'Creates a generator from a state.\n\n    See `__init__` for description of `state` and `alg`.\n\n    Args:\n      state: the new state.\n      alg: the RNG algorithm.\n\n    Returns:\n      The new generator.\n    '
        return cls(alg=alg, state=state)

    @classmethod
    def from_seed(cls, seed, alg=None):
        if False:
            return 10
        'Creates a generator from a seed.\n\n    A seed is a 1024-bit unsigned integer represented either as a Python\n    integer or a vector of integers. Seeds shorter than 1024-bit will be\n    padded. The padding, the internal structure of a seed and the way a seed\n    is converted to a state are all opaque (unspecified). The only semantics\n    specification of seeds is that two different seeds are likely to produce\n    two independent generators (but no guarantee).\n\n    Args:\n      seed: the seed for the RNG.\n      alg: (optional) the RNG algorithm. If None, it will be auto-selected. See\n        `__init__` for its possible values.\n\n    Returns:\n      The new generator.\n    '
        if alg is None:
            alg = DEFAULT_ALGORITHM
        alg = random_ops_util.convert_alg_to_int(alg)
        state = create_rng_state(seed, alg)
        return cls(state=state, alg=alg)

    @classmethod
    def from_non_deterministic_state(cls, alg=None):
        if False:
            print('Hello World!')
        'Creates a generator by non-deterministically initializing its state.\n\n    The source of the non-determinism will be platform- and time-dependent.\n\n    Args:\n      alg: (optional) the RNG algorithm. If None, it will be auto-selected. See\n        `__init__` for its possible values.\n\n    Returns:\n      The new generator.\n    '
        if config.is_op_determinism_enabled():
            raise RuntimeError('"from_non_deterministic_state" cannot be called when determinism is enabled.')
        if alg is None:
            alg = DEFAULT_ALGORITHM
        alg = random_ops_util.convert_alg_to_int(alg)
        state = non_deterministic_ints(shape=[_get_state_size(alg)], dtype=SEED_TYPE)
        return cls(state=state, alg=alg)

    @classmethod
    def from_key_counter(cls, key, counter, alg):
        if False:
            while True:
                i = 10
        'Creates a generator from a key and a counter.\n\n    This constructor only applies if the algorithm is a counter-based algorithm.\n    See method `key` for the meaning of "key" and "counter".\n\n    Args:\n      key: the key for the RNG, a scalar of type STATE_TYPE.\n      counter: a vector of dtype STATE_TYPE representing the initial counter for\n        the RNG, whose length is algorithm-specific.,\n      alg: the RNG algorithm. If None, it will be auto-selected. See\n        `__init__` for its possible values.\n\n    Returns:\n      The new generator.\n    '
        counter = _convert_to_state_tensor(counter)
        key = _convert_to_state_tensor(key)
        alg = random_ops_util.convert_alg_to_int(alg)
        counter.shape.assert_is_compatible_with([_get_state_size(alg) - 1])
        key.shape.assert_is_compatible_with([])
        key = array_ops.reshape(key, [1])
        state = array_ops.concat([counter, key], 0)
        return cls(state=state, alg=alg)

    def __init__(self, copy_from=None, state=None, alg=None):
        if False:
            while True:
                i = 10
        'Creates a generator.\n\n    The new generator will be initialized by one of the following ways, with\n    decreasing precedence:\n    (1) If `copy_from` is not None, the new generator is initialized by copying\n        information from another generator.\n    (2) If `state` and `alg` are not None (they must be set together), the new\n        generator is initialized by a state.\n\n    Args:\n      copy_from: a generator to be copied from.\n      state: a vector of dtype STATE_TYPE representing the initial state of the\n        RNG, whose length and semantics are algorithm-specific. If it\'s a\n        variable, the generator will reuse it instead of creating a new\n        variable.\n      alg: the RNG algorithm. Possible values are\n        `tf.random.Algorithm.PHILOX` for the Philox algorithm and\n        `tf.random.Algorithm.THREEFRY` for the ThreeFry algorithm\n        (see paper \'Parallel Random Numbers: As Easy as 1, 2, 3\'\n        [https://www.thesalmons.org/john/random123/papers/random123sc11.pdf]).\n        The string names `"philox"` and `"threefry"` can also be used.\n        Note `PHILOX` guarantees the same numbers are produced (given\n        the same random state) across all architectures (CPU, GPU, XLA etc).\n    '
        if distribute_lib.has_strategy():
            self._distribution_strategy = distribute_lib.get_strategy()
        else:
            self._distribution_strategy = None
        if copy_from is not None:
            assert (alg or state) is None
            self._state_var = self._create_variable(copy_from.state, dtype=STATE_TYPE, trainable=False)
            self._alg = copy_from.algorithm
        else:
            assert alg is not None and state is not None
            alg = random_ops_util.convert_alg_to_int(alg)
            if isinstance(state, variables.Variable):
                _check_state_shape(state.shape, alg)
                self._state_var = state
            else:
                state = _convert_to_state_tensor(state)
                _check_state_shape(state.shape, alg)
                self._state_var = self._create_variable(state, dtype=STATE_TYPE, trainable=False)
            self._alg = alg

    def _create_variable(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Creates a variable.\n\n    Args:\n      *args: positional arguments passed along to `variables.Variable.\n      **kwargs: keyword arguments passed along to `variables.Variable.\n\n    Returns:\n      The created variable.\n    '
        with ops.name_scope('random_generator'):
            kwargs['name'] = 'StateVar'
            v = variables.Variable(*args, **kwargs)
        if isinstance(v, sharded_variable.ShardedVariable):
            raise ValueError("tf.random.Generator state is sharded, which is not allowed. When creating a tf.distribute.experimental.ParameterServerStrategy, please make sure that the `variable_partitioner` argument won't shard a small variable of shape [2] or [3]. Ways to avoid sharding small variables include setting `variable_partitioner` to None or to tf.distribute.experimental.partitioners.MinSizePartitioner with a large enough `min_shard_bytes`.")
        return v

    def reset(self, state):
        if False:
            print('Hello World!')
        'Resets the generator by a new state.\n\n    See `__init__` for the meaning of "state".\n\n    Args:\n      state: the new state.\n    '
        state = _convert_to_state_tensor(state)
        state.shape.assert_is_compatible_with([_get_state_size(self.algorithm)])
        self._state_var.assign(state)

    def reset_from_seed(self, seed):
        if False:
            return 10
        'Resets the generator by a new seed.\n\n    See `from_seed` for the meaning of "seed".\n\n    Args:\n      seed: the new seed.\n    '
        state = create_rng_state(seed, self.algorithm)
        self._state_var.assign(state)

    def reset_from_key_counter(self, key, counter):
        if False:
            for i in range(10):
                print('nop')
        'Resets the generator by a new key-counter pair.\n\n    See `from_key_counter` for the meaning of "key" and "counter".\n\n    Args:\n      key: the new key.\n      counter: the new counter.\n    '
        counter = _convert_to_state_tensor(counter)
        key = _convert_to_state_tensor(key)
        counter.shape.assert_is_compatible_with([_get_state_size(self.algorithm) - 1])
        key.shape.assert_is_compatible_with([])
        key = array_ops.reshape(key, [1])
        state = array_ops.concat([counter, key], 0)
        self._state_var.assign(state)

    @property
    def state(self):
        if False:
            i = 10
            return i + 15
        'The internal state of the RNG.'
        return self._state_var

    @property
    def algorithm(self):
        if False:
            for i in range(10):
                print('nop')
        'The RNG algorithm id (a Python integer or scalar integer Tensor).'
        return self._alg

    def _standard_normal(self, shape, dtype):
        if False:
            while True:
                i = 10
        (key, counter) = self._prepare_key_counter(shape)
        return gen_stateless_random_ops_v2.stateless_random_normal_v2(shape, key=key, counter=counter, dtype=dtype, alg=self.algorithm)

    @property
    def key(self):
        if False:
            i = 10
            return i + 15
        "The 'key' part of the state of a counter-based RNG.\n\n    For a counter-base RNG algorithm such as Philox and ThreeFry (as\n    described in paper 'Parallel Random Numbers: As Easy as 1, 2, 3'\n    [https://www.thesalmons.org/john/random123/papers/random123sc11.pdf]),\n    the RNG state consists of two parts: counter and key. The output is\n    generated via the formula: output=hash(key, counter), i.e. a hashing of\n    the counter parametrized by the key. Two RNGs with two different keys can\n    be thought as generating two independent random-number streams (a stream\n    is formed by increasing the counter).\n\n    Returns:\n      A scalar which is the 'key' part of the state, if the RNG algorithm is\n        counter-based; otherwise it raises a ValueError.\n    "
        alg = self.algorithm
        if alg in (a.value for a in random_ops_util.Algorithm):
            return self._state_var[-1]
        else:
            raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))

    def _skip_single_var(self, var, delta):
        if False:
            while True:
                i = 10
        resource_variable_ops.variable_accessed(var)
        return gen_stateful_random_ops.rng_read_and_skip(var.handle, alg=math_ops.cast(self.algorithm, dtypes.int32), delta=math_ops.cast(delta, dtypes.uint64))

    def skip(self, delta):
        if False:
            while True:
                i = 10
        'Advance the counter of a counter-based RNG.\n\n    Args:\n      delta: the amount of advancement. The state of the RNG after\n        `skip(n)` will be the same as that after `normal([n])`\n        (or any other distribution). The actual increment added to the\n        counter is an unspecified implementation detail.\n\n    Returns:\n      A `Tensor` of type `int64`.\n    '

        def update_fn(v):
            if False:
                return 10
            return self._skip_single_var(v, delta)
        if values_util.is_saving_non_distributed():
            return update_fn(self.state)
        if self._distribution_strategy is not None:
            with distribute_lib.enter_or_assert_strategy(self._distribution_strategy):
                if distribute_lib.in_cross_replica_context():
                    values_util.mark_as_unsaveable()
                if distribute_lib.in_cross_replica_context() or 'CentralStorage' in type(self._distribution_strategy).__name__:
                    return distribute_lib.get_strategy().extended.update(self.state, update_fn)
        return update_fn(self.state)

    def _preprocess_key(self, key):
        if False:
            return 10
        if self._distribution_strategy is None:
            return key
        with distribute_lib.enter_or_assert_strategy(self._distribution_strategy):
            replica_id = get_replica_id()
            if replica_id is not None:
                replica_id = array_ops_stack.stack([replica_id, 0], axis=0)
                replica_id = math_ops.cast(replica_id, dtypes.uint64)
                key = gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2(shape=[1], key=key, counter=replica_id, dtype=dtypes.uint64, alg=self.algorithm)
            return key

    def _prepare_key_counter(self, shape):
        if False:
            while True:
                i = 10
        delta = math_ops.reduce_prod(shape)
        counter_key = self.skip(delta)
        counter_size = _get_counter_size(self.algorithm)
        counter = array_ops.bitcast(counter_key[:counter_size], dtypes.uint64)
        key = array_ops.bitcast(counter_key[counter_size:counter_size + 1], dtypes.uint64)
        key = self._preprocess_key(key)
        return (key, counter)

    def normal(self, shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, name=None):
        if False:
            i = 10
            return i + 15
        'Outputs random values from a normal distribution.\n\n    Args:\n      shape: A 1-D integer Tensor or Python array. The shape of the output\n        tensor.\n      mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal\n        distribution.\n      stddev: A 0-D Tensor or Python value of type `dtype`. The standard\n        deviation of the normal distribution.\n      dtype: The type of the output.\n      name: A name for the operation (optional).\n\n    Returns:\n      A tensor of the specified shape filled with random normal values.\n    '
        with ops.name_scope(name, 'stateful_normal', [shape, mean, stddev]) as name:
            shape = _shape_tensor(shape)
            mean = ops.convert_to_tensor(mean, dtype=dtype, name='mean')
            stddev = ops.convert_to_tensor(stddev, dtype=dtype, name='stddev')
            rnd = self._standard_normal(shape, dtype=dtype)
            return math_ops.add(rnd * stddev, mean, name=name)

    def _truncated_normal(self, shape, dtype):
        if False:
            i = 10
            return i + 15
        (key, counter) = self._prepare_key_counter(shape)
        return gen_stateless_random_ops_v2.stateless_truncated_normal_v2(shape=shape, key=key, counter=counter, dtype=dtype, alg=self.algorithm)

    def truncated_normal(self, shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, name=None):
        if False:
            while True:
                i = 10
        'Outputs random values from a truncated normal distribution.\n\n    The generated values follow a normal distribution with specified mean and\n    standard deviation, except that values whose magnitude is more than\n    2 standard deviations from the mean are dropped and re-picked.\n\n    Args:\n      shape: A 1-D integer Tensor or Python array. The shape of the output\n        tensor.\n      mean: A 0-D Tensor or Python value of type `dtype`. The mean of the\n        truncated normal distribution.\n      stddev: A 0-D Tensor or Python value of type `dtype`. The standard\n        deviation of the normal distribution, before truncation.\n      dtype: The type of the output.\n      name: A name for the operation (optional).\n\n    Returns:\n      A tensor of the specified shape filled with random truncated normal\n        values.\n    '
        with ops.name_scope(name, 'truncated_normal', [shape, mean, stddev]) as name:
            shape_tensor = _shape_tensor(shape)
            mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name='mean')
            stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name='stddev')
            rnd = self._truncated_normal(shape_tensor, dtype=dtype)
            mul = rnd * stddev_tensor
            return math_ops.add(mul, mean_tensor, name=name)

    def _uniform(self, shape, dtype):
        if False:
            print('Hello World!')
        (key, counter) = self._prepare_key_counter(shape)
        return gen_stateless_random_ops_v2.stateless_random_uniform_v2(shape=shape, key=key, counter=counter, dtype=dtype, alg=self.algorithm)

    def _uniform_full_int(self, shape, dtype, name=None):
        if False:
            for i in range(10):
                print('nop')
        (key, counter) = self._prepare_key_counter(shape)
        return gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2(shape=shape, key=key, counter=counter, dtype=dtype, alg=self.algorithm, name=name)

    def uniform(self, shape, minval=0, maxval=None, dtype=dtypes.float32, name=None):
        if False:
            i = 10
            return i + 15
        'Outputs random values from a uniform distribution.\n\n    The generated values follow a uniform distribution in the range\n    `[minval, maxval)`. The lower bound `minval` is included in the range, while\n    the upper bound `maxval` is excluded. (For float numbers especially\n    low-precision types like bfloat16, because of\n    rounding, the result may sometimes include `maxval`.)\n\n    For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must\n    be specified explicitly.\n\n    In the integer case, the random integers are slightly biased unless\n    `maxval - minval` is an exact power of two.  The bias is small for values of\n    `maxval - minval` significantly smaller than the range of the output (either\n    `2**32` or `2**64`).\n\n    For full-range random integers, pass `minval=None` and `maxval=None` with an\n    integer `dtype` (for integer dtypes, `minval` and `maxval` must be both\n    `None` or both not `None`).\n\n    Args:\n      shape: A 1-D integer Tensor or Python array. The shape of the output\n        tensor.\n      minval: A Tensor or Python value of type `dtype`, broadcastable with\n        `shape` (for integer types, broadcasting is not supported, so it needs\n        to be a scalar). The lower bound (included) on the range of random\n        values to generate. Pass `None` for full-range integers. Defaults to 0.\n      maxval: A Tensor or Python value of type `dtype`, broadcastable with\n        `shape` (for integer types, broadcasting is not supported, so it needs\n        to be a scalar). The upper bound (excluded) on the range of random\n        values to generate. Pass `None` for full-range integers. Defaults to 1\n        if `dtype` is floating point.\n      dtype: The type of the output.\n      name: A name for the operation (optional).\n\n    Returns:\n      A tensor of the specified shape filled with random uniform values.\n\n    Raises:\n      ValueError: If `dtype` is integral and `maxval` is not specified.\n    '
        dtype = dtypes.as_dtype(dtype)
        if dtype.is_integer:
            if (minval is None) != (maxval is None):
                raise ValueError('For integer dtype {}, minval and maxval must be both `None` or both non-`None`; got minval={} and maxval={}'.format(dtype, minval, maxval))
        elif maxval is None:
            maxval = 1
        with ops.name_scope(name, 'stateful_uniform', [shape, minval, maxval]) as name:
            shape = _shape_tensor(shape)
            if dtype.is_integer and minval is None:
                return self._uniform_full_int(shape=shape, dtype=dtype, name=name)
            minval = ops.convert_to_tensor(minval, dtype=dtype, name='min')
            maxval = ops.convert_to_tensor(maxval, dtype=dtype, name='max')
            if dtype.is_integer:
                (key, counter) = self._prepare_key_counter(shape)
                return gen_stateless_random_ops_v2.stateless_random_uniform_int_v2(shape=shape, key=key, counter=counter, minval=minval, maxval=maxval, alg=self.algorithm, name=name)
            else:
                rnd = self._uniform(shape=shape, dtype=dtype)
                return math_ops.add(rnd * (maxval - minval), minval, name=name)

    def uniform_full_int(self, shape, dtype=dtypes.uint64, name=None):
        if False:
            while True:
                i = 10
        "Uniform distribution on an integer type's entire range.\n\n    This method is the same as setting `minval` and `maxval` to `None` in the\n    `uniform` method.\n\n    Args:\n      shape: the shape of the output.\n      dtype: (optional) the integer type, default to uint64.\n      name: (optional) the name of the node.\n\n    Returns:\n      A tensor of random numbers of the required shape.\n    "
        dtype = dtypes.as_dtype(dtype)
        with ops.name_scope(name, 'stateful_uniform_full_int', [shape]) as name:
            shape = _shape_tensor(shape)
            return self._uniform_full_int(shape=shape, dtype=dtype, name=name)

    def binomial(self, shape, counts, probs, dtype=dtypes.int32, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Outputs random values from a binomial distribution.\n\n    The generated values follow a binomial distribution with specified count and\n    probability of success parameters.\n\n    Example:\n\n    ```python\n    counts = [10., 20.]\n    # Probability of success.\n    probs = [0.8]\n\n    rng = tf.random.Generator.from_seed(seed=234)\n    binomial_samples = rng.binomial(shape=[2], counts=counts, probs=probs)\n\n\n    counts = ... # Shape [3, 1, 2]\n    probs = ...  # Shape [1, 4, 2]\n    shape = [3, 4, 3, 4, 2]\n    rng = tf.random.Generator.from_seed(seed=1717)\n    # Sample shape will be [3, 4, 3, 4, 2]\n    binomial_samples = rng.binomial(shape=shape, counts=counts, probs=probs)\n    ```\n\n\n    Args:\n      shape: A 1-D integer Tensor or Python array. The shape of the output\n        tensor.\n      counts: Tensor. The counts of the binomial distribution. Must be\n        broadcastable with `probs`, and broadcastable with the rightmost\n        dimensions of `shape`.\n      probs: Tensor. The probability of success for the\n        binomial distribution. Must be broadcastable with `counts` and\n        broadcastable with the rightmost dimensions of `shape`.\n      dtype: The type of the output. Default: tf.int32\n      name: A name for the operation (optional).\n\n    Returns:\n      samples: A Tensor of the specified shape filled with random binomial\n        values.  For each i, each samples[i, ...] is an independent draw from\n        the binomial distribution on counts[i] trials with probability of\n        success probs[i].\n    '
        dtype = dtypes.as_dtype(dtype)
        with ops.name_scope(name, 'binomial', [shape, counts, probs]) as name:
            counts = ops.convert_to_tensor(counts, name='counts')
            probs = ops.convert_to_tensor(probs, name='probs')
            shape_tensor = _shape_tensor(shape)
            return gen_stateful_random_ops.stateful_random_binomial(self.state.handle, self.algorithm, shape=shape_tensor, counts=counts, probs=probs, dtype=dtype, name=name)

    def _make_int64_keys(self, shape=()):
        if False:
            for i in range(10):
                print('nop')
        return self.uniform_full_int(shape=shape, dtype=dtypes.int64)

    def make_seeds(self, count=1):
        if False:
            while True:
                i = 10
        'Generates seeds for stateless random ops.\n\n    For example:\n\n    ```python\n    seeds = get_global_generator().make_seeds(count=10)\n    for i in range(10):\n      seed = seeds[:, i]\n      numbers = stateless_random_normal(shape=[2, 3], seed=seed)\n      ...\n    ```\n\n    Args:\n      count: the number of seed pairs (note that stateless random ops need a\n        pair of seeds to invoke).\n\n    Returns:\n      A tensor of shape [2, count] and dtype int64.\n    '
        alg = self.algorithm
        if alg in (a.value for a in random_ops_util.Algorithm):
            keys = self._make_int64_keys(shape=[count])
            zeros = array_ops.zeros_like(keys)
            return array_ops_stack.stack([keys, zeros])
        else:
            raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))

    def split(self, count=1):
        if False:
            return 10
        'Returns a list of independent `Generator` objects.\n\n    Two generators are independent of each other in the sense that the\n    random-number streams they generate don\'t have statistically detectable\n    correlations. The new generators are also independent of the old one.\n    The old generator\'s state will be changed (like other random-number\n    generating methods), so two calls of `split` will return different\n    new generators.\n\n    For example:\n\n    ```python\n    gens = get_global_generator().split(count=10)\n    for gen in gens:\n      numbers = gen.normal(shape=[2, 3])\n      # ...\n    gens2 = get_global_generator().split(count=10)\n    # gens2 will be different from gens\n    ```\n\n    The new generators will be put on the current device (possible different\n    from the old generator\'s), for example:\n\n    ```python\n    with tf.device("/device:CPU:0"):\n      gen = Generator(seed=1234)  # gen is on CPU\n    with tf.device("/device:GPU:0"):\n      gens = gen.split(count=10)  # gens are on GPU\n    ```\n\n    Args:\n      count: the number of generators to return.\n\n    Returns:\n      A list (length `count`) of `Generator` objects independent of each other.\n      The new generators have the same RNG algorithm as the old one.\n    '

        def _key_to_state(alg, key):
            if False:
                return 10
            return [0] * (_get_state_size(alg) - 1) + [key]
        alg = self.algorithm
        if alg in (a.value for a in random_ops_util.Algorithm):
            keys = self._make_int64_keys(shape=[count])
            return [Generator(state=_key_to_state(alg, key), alg=alg) for key in array_ops_stack.unstack(keys, num=count)]
        else:
            raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))
global_generator = None

@tf_export('random.get_global_generator', 'random.experimental.get_global_generator')
def get_global_generator():
    if False:
        print('Hello World!')
    'Retrieves the global generator.\n\n  This function will create the global generator the first time it is called,\n  and the generator will be placed at the default device at that time, so one\n  needs to be careful when this function is first called. Using a generator\n  placed on a less-ideal device will incur performance regression.\n\n  Returns:\n    The global `tf.random.Generator` object.\n  '
    global global_generator
    if global_generator is None:
        if config.is_op_determinism_enabled():
            raise RuntimeError('"get_global_generator" cannot be called if determinism is enabled, unless "set_global_generator" has already been called. Please call "set_global_generator" first.')
        with ops.init_scope():
            global_generator = Generator.from_non_deterministic_state()
    return global_generator

@tf_export('random.set_global_generator', 'random.experimental.set_global_generator')
def set_global_generator(generator):
    if False:
        while True:
            i = 10
    'Replaces the global generator with another `Generator` object.\n\n  This function replaces the global generator with the provided `generator`\n  object.\n  A random number generator utilizes a `tf.Variable` object to store its state.\n  The user shall be aware of caveats how `set_global_generator` interacts with\n  `tf.function`:\n\n  - tf.function puts restrictions on Variable creation thus one cannot freely\n    create a new random generator instance inside `tf.function`.\n    To call `set_global_generator` inside `tf.function`, the generator instance\n    must have already been created eagerly.\n  - tf.function captures the Variable during trace-compilation, thus a compiled\n    f.function will not be affected `set_global_generator` as demonstrated by\n    random_test.py/RandomTest.testResetGlobalGeneratorBadWithDefun .\n\n  For most use cases, avoid calling `set_global_generator` after program\n  initialization, and prefer to reset the state of the existing global generator\n  instead, such as,\n\n  >>> rng = tf.random.get_global_generator()\n  >>> rng.reset_from_seed(30)\n\n\n  Args:\n    generator: the new `Generator` object.\n  '
    global global_generator
    global_generator = generator
"""Stateless random ops which take seed as a tensor input."""
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_random_index_shuffle_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import shape_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
ops.NotDifferentiable('StatelessMultinomial')
ops.NotDifferentiable('StatelessRandomBinomial')
ops.NotDifferentiable('StatelessRandomNormal')
ops.NotDifferentiable('StatelessRandomPoisson')
ops.NotDifferentiable('StatelessRandomUniform')
ops.NotDifferentiable('StatelessRandomUniformInt')
ops.NotDifferentiable('StatelessRandomUniformFullInt')
ops.NotDifferentiable('StatelessTruncatedNormal')
ops.NotDifferentiable('StatelessRandomNormalV2')
ops.NotDifferentiable('StatelessRandomUniformV2')
ops.NotDifferentiable('StatelessRandomUniformIntV2')
ops.NotDifferentiable('StatelessRandomUniformFullIntV2')
ops.NotDifferentiable('StatelessTruncatedNormalV2')
ops.NotDifferentiable('StatelessRandomShuffle')
ops.NotDifferentiable('RandomIndexShuffle')

@tf_export('random.split', 'random.experimental.stateless_split')
@dispatch.add_dispatch_support
def split(seed, num=2, alg='auto_select'):
    if False:
        for i in range(10):
            print('nop')
    "Splits an RNG seed into `num` new seeds by adding a leading axis.\n\n  Example:\n\n  >>> seed = [1, 2]\n  >>> new_seeds = tf.random.split(seed, num=3)\n  >>> print(new_seeds)\n  tf.Tensor(\n  [[1105988140 1738052849]\n   [-335576002  370444179]\n   [  10670227 -246211131]], shape=(3, 2), dtype=int32)\n  >>> tf.random.stateless_normal(shape=[3], seed=new_seeds[0, :])\n  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.59835213, -0.9578608 ,\n  0.9002807 ], dtype=float32)>\n\n  Args:\n    seed: an RNG seed (a tensor with shape [2] and dtype `int32` or `int64`).\n      (When using XLA, only `int32` is allowed.)\n    num: optional, a positive integer or scalar tensor indicating the number of\n      seeds to produce (default 2).\n    alg: The RNG algorithm used to generate the random numbers. See\n      `tf.random.stateless_uniform` for a detailed explanation.\n\n  Returns:\n    A tensor with shape [num, 2] representing `num` new seeds. It will have the\n    same dtype as `seed` (if `seed` doesn't have an explict dtype, the dtype\n    will be determined by `tf.convert_to_tensor`).\n  "
    seed = ops.convert_to_tensor(seed)
    return stateless_random_uniform(shape=[num, 2], seed=seed, dtype=seed.dtype, minval=None, maxval=None, alg=alg)

@tf_export('random.fold_in', 'random.experimental.stateless_fold_in')
@dispatch.add_dispatch_support
def fold_in(seed, data, alg='auto_select'):
    if False:
        while True:
            i = 10
    'Folds in data to an RNG seed to form a new RNG seed.\n\n  For example, in a distributed-training setting, suppose we have a master seed\n  and a replica ID. We want to fold the replica ID into the master seed to\n  form a "replica seed" to be used by that replica later on, so that different\n  replicas will generate different random numbers but the reproducibility of the\n  whole system can still be controlled by the master seed:\n\n  >>> master_seed = [1, 2]\n  >>> replica_id = 3\n  >>> replica_seed = tf.random.experimental.stateless_fold_in(\n  ...   master_seed, replica_id)\n  >>> print(replica_seed)\n  tf.Tensor([1105988140          3], shape=(2,), dtype=int32)\n  >>> tf.random.stateless_normal(shape=[3], seed=replica_seed)\n  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.03197195, 0.8979765 ,\n  0.13253039], dtype=float32)>\n\n  Args:\n    seed: an RNG seed (a tensor with shape [2] and dtype `int32` or `int64`).\n      (When using XLA, only `int32` is allowed.)\n    data: an `int32` or `int64` scalar representing data to be folded in to the\n      seed.\n    alg: The RNG algorithm used to generate the random numbers. See\n      `tf.random.stateless_uniform` for a detailed explanation.\n\n  Returns:\n    A new RNG seed that is a deterministic function of the inputs and is\n    statistically safe for producing a stream of new pseudo-random values. It\n    will have the same dtype as `data` (if `data` doesn\'t have an explict dtype,\n    the dtype will be determined by `tf.convert_to_tensor`).\n  '
    data = ops.convert_to_tensor(data)
    seed1 = stateless_random_uniform(shape=[], seed=seed, dtype=data.dtype, minval=None, maxval=None, alg=alg)
    return array_ops_stack.stack([seed1, data])

@tf_export('random.experimental.index_shuffle')
@dispatch.add_dispatch_support
def index_shuffle(index, seed, max_index):
    if False:
        return 10
    "Outputs the position of `index` in a permutation of `[0, ..., max_index]`.\n\n  For each possible `seed` and `max_index` there is one pseudorandom\n  permutation of the sequence `S=[0, ..., max_index]`. Instead of\n  materializing the full array we can compute the new position of any\n  integer `i` (`0 <= i <= max_index`) in `S`. This can be useful for\n  very large `max_index`s by avoiding allocating large chunks of\n  memory.\n\n  In the simplest case, `index` and `max_index` are scalars, and\n  `seed` is a length-2 vector (as typical for stateless RNGs). But\n  you can add a leading batch dimension to all of them. If some of\n  them don't have the batch dimension while others do, `index_shuffle`\n  will add a batch dimension to the former by broadcasting.\n\n  The input `index` and output can be used as indices to shuffle a\n  vector.  For example:\n\n  >>> vector = tf.constant(['e0', 'e1', 'e2', 'e3'])\n  >>> indices = tf.random.experimental.index_shuffle(\n  ...   index=tf.range(4), seed=[5, 9], max_index=3)\n  >>> print(indices)\n  tf.Tensor([2 0 1 3], shape=(4,), dtype=int32)\n  >>> shuffled_vector = tf.gather(vector, indices)\n  >>> print(shuffled_vector)\n  tf.Tensor([b'e2' b'e0' b'e1' b'e3'], shape=(4,), dtype=string)\n\n  More usefully, it can be used in a streaming (aka online) scenario such as\n  `tf.data`, where each element of `vector` is processed individually and the\n  whole `vector` is never materialized in memory.\n\n  >>> dataset = tf.data.Dataset.range(10)\n  >>> dataset = dataset.map(\n  ...  lambda idx: tf.random.experimental.index_shuffle(idx, [5, 8], 9))\n  >>> print(list(dataset.as_numpy_iterator()))\n  [3, 8, 0, 1, 2, 7, 6, 9, 4, 5]\n\n  This operation is stateless (like the `tf.random.stateless_*`\n  functions), meaning the output is fully determined by the `seed`\n  (other inputs being equal).  Each `seed` choice corresponds to one\n  permutation, so when calling this function multiple times for the\n  same shuffling, please make sure to use the same `seed`. For\n  example:\n\n  >>> seed = [5, 9]\n  >>> idx0 = tf.random.experimental.index_shuffle(0, seed, 3)\n  >>> idx1 = tf.random.experimental.index_shuffle(1, seed, 3)\n  >>> idx2 = tf.random.experimental.index_shuffle(2, seed, 3)\n  >>> idx3 = tf.random.experimental.index_shuffle(3, seed, 3)\n  >>> shuffled_vector = tf.gather(vector, [idx0, idx1, idx2, idx3])\n  >>> print(shuffled_vector)\n  tf.Tensor([b'e2' b'e0' b'e1' b'e3'], shape=(4,), dtype=string)\n\n  Args:\n    index: An integer scalar tensor or vector with values in `[0, max_index]`.\n      It can be seen as either a value `v` in the sequence `S=[0, ...,\n      max_index]` to be permutated, or as an index of an element `e` in a\n      shuffled vector.\n    seed: A tensor of shape [2] or [n, 2] with dtype `int32`, `uint32`, `int64`\n      or `uint64`.  The RNG seed. If the rank is unknown during graph-building\n      time it must be 1 at runtime.\n    max_index: A non-negative tensor with the same shape and dtype as `index`.\n      The upper bound (inclusive).\n\n  Returns:\n    If all inputs were scalar (shape [2] for `seed`), the output will\n    be a scalar with the same dtype as `index`. The output can be seen\n    as the new position of `v` in `S`, or as the index of `e` in the\n    vector before shuffling.  If one or multiple inputs were vectors\n    (shape [n, 2] for `seed`), then the output will be a vector of the\n    same size which each element shuffled independently. Scalar values\n    are broadcasted in this case.\n  "
    seed = ops.convert_to_tensor(seed)
    if seed.shape.rank is None:
        paddings = [[1, 0]]
    else:
        paddings = [[1, 0]] + (seed.shape.rank - 1) * [[0, 0]]
    seed = array_ops.pad(seed, paddings, constant_values=498247692)
    return gen_random_index_shuffle_ops.random_index_shuffle(index, seed=seed, max_index=max_index, rounds=4)

@tf_export('random.experimental.stateless_shuffle')
@dispatch.add_dispatch_support
def stateless_shuffle(value, seed, alg='auto_select', name=None):
    if False:
        i = 10
        return i + 15
    'Randomly and deterministically shuffles a tensor along its first dimension.\n\n  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped\n  to one and only one `output[i]`. For example, a mapping that might occur for a\n  3x2 tensor is:\n\n  ```python\n  [[1, 2],       [[5, 6],\n   [3, 4],  ==>   [1, 2],\n   [5, 6]]        [3, 4]]\n  ```\n\n  >>> v = tf.constant([[1, 2], [3, 4], [5, 6]])\n  >>> shuffled = tf.random.experimental.stateless_shuffle(v, seed=[8, 9])\n  >>> print(shuffled)\n  tf.Tensor(\n  [[5 6]\n    [1 2]\n    [3 4]], shape=(3, 2), dtype=int32)\n\n  This is a stateless version of `tf.random.shuffle`: if run twice with the\n  same `value` and `seed`, it will produce the same result.  The\n  output is consistent across multiple runs on the same hardware (and between\n  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU\n  hardware.\n\n  Args:\n    value: A Tensor to be shuffled.\n    seed: A shape [2] Tensor. The seed to the random number generator. Must have\n      dtype `int32` or `int64`.\n    alg: The RNG algorithm used to generate the random numbers. See\n      `tf.random.stateless_uniform` for a detailed explanation.\n    name: A name for the operation.\n\n  Returns:\n    A tensor of same shape and type as `value`, shuffled along its first\n    dimension.\n  '
    with ops.name_scope(name, 'stateless_shuffle', [value, seed]) as name:
        (key, counter, alg) = random_ops_util.get_key_counter_alg(seed, alg)
        return gen_stateless_random_ops_v2.stateless_shuffle(value, key=key, counter=counter, alg=alg)

@tf_export('random.stateless_uniform')
@dispatch.add_dispatch_support
def stateless_random_uniform(shape, seed, minval=0, maxval=None, dtype=dtypes.float32, name=None, alg='auto_select'):
    if False:
        print('Hello World!')
    'Outputs deterministic pseudorandom values from a uniform distribution.\n\n  This is a stateless version of `tf.random.uniform`: if run twice with the\n  same seeds and shapes, it will produce the same pseudorandom numbers.  The\n  output is consistent across multiple runs on the same hardware (and between\n  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU\n  hardware.\n\n  The generated values follow a uniform distribution in the range\n  `[minval, maxval)`. The lower bound `minval` is included in the range, while\n  the upper bound `maxval` is excluded.\n\n  For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must\n  be specified explicitly.\n\n  In the integer case, the random integers are slightly biased unless\n  `maxval - minval` is an exact power of two.  The bias is small for values of\n  `maxval - minval` significantly smaller than the range of the output (either\n  `2**32` or `2**64`).\n\n  For full-range (i.e. inclusive of both max and min) random integers, pass\n  `minval=None` and `maxval=None` with an integer `dtype`. For an integer dtype\n  either both `minval` and `maxval` must be `None` or neither may be `None`. For\n  example:\n  ```python\n  ints = tf.random.stateless_uniform(\n      [10], seed=(2, 3), minval=None, maxval=None, dtype=tf.int32)\n  ```\n\n  Args:\n    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    minval: A Tensor or Python value of type `dtype`, broadcastable with `shape`\n      (for integer types, broadcasting is not supported, so it needs to be a\n      scalar). The lower bound on the range of random values to generate. Pass\n      `None` for full-range integers.  Defaults to 0.\n    maxval: A Tensor or Python value of type `dtype`, broadcastable with `shape`\n      (for integer types, broadcasting is not supported, so it needs to be a\n      scalar). The upper bound on the range of random values to generate.\n      Defaults to 1 if `dtype` is floating point. Pass `None` for full-range\n      integers.\n    dtype: The type of the output: `float16`, `bfloat16`, `float32`, `float64`,\n      `int32`, or `int64`. For unbounded uniform ints (`minval`, `maxval` both\n      `None`), `uint32` and `uint64` may be used. Defaults to `float32`.\n    name: A name for the operation (optional).\n    alg: The RNG algorithm used to generate the random numbers. Valid choices\n      are `"philox"` for [the Philox\n      algorithm](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf),\n      `"threefry"` for [the ThreeFry\n      algorithm](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf),\n      and `"auto_select"` (default) for the system to automatically select an\n      algorithm based the device type. Values of `tf.random.Algorithm` can also\n      be used. Note that with `"auto_select"`, the outputs of this function may\n      change when it is running on a different device.\n\n  Returns:\n    A tensor of the specified shape filled with random uniform values.\n\n  Raises:\n    ValueError: If `dtype` is integral and only one of `minval` or `maxval` is\n      specified.\n  '
    dtype = dtypes.as_dtype(dtype)
    accepted_dtypes = (dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64, dtypes.uint32, dtypes.uint64)
    if dtype not in accepted_dtypes:
        raise ValueError(f'Argument `dtype` got invalid value {dtype}. Accepted dtypes are {accepted_dtypes}.')
    if dtype.is_integer:
        if (minval is None) != (maxval is None):
            raise ValueError(f'For integer `dtype` argument {dtype}, argument `minval` and `maxval` must be both None or not None. Got `minval`={minval} and `maxval`={maxval}.')
        if minval is not None and dtype in (dtypes.uint32, dtypes.uint64):
            raise ValueError(f"Argument `dtype` got invalid value {dtype} when argument `minval` is not None. Please don't use unsigned integers in this case.")
    elif maxval is None:
        maxval = 1
    with ops.name_scope(name, 'stateless_random_uniform', [shape, seed, minval, maxval]) as name:
        shape = shape_util.shape_tensor(shape)
        if dtype.is_integer and minval is None:
            (key, counter, alg) = random_ops_util.get_key_counter_alg(seed, alg)
            result = gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2(shape, key=key, counter=counter, dtype=dtype, alg=alg, name=name)
        else:
            minval = ops.convert_to_tensor(minval, dtype=dtype, name='min')
            maxval = ops.convert_to_tensor(maxval, dtype=dtype, name='max')
            if dtype.is_integer:
                (key, counter, alg) = random_ops_util.get_key_counter_alg(seed, alg)
                result = gen_stateless_random_ops_v2.stateless_random_uniform_int_v2(shape, key=key, counter=counter, minval=minval, maxval=maxval, alg=alg, name=name)
            else:
                (key, counter, alg) = random_ops_util.get_key_counter_alg(seed, alg)
                rnd = gen_stateless_random_ops_v2.stateless_random_uniform_v2(shape, key=key, counter=counter, dtype=dtype, alg=alg)
                result = math_ops.add(rnd * (maxval - minval), minval, name=name)
        shape_util.maybe_set_static_shape(result, shape)
        return result

@tf_export('random.stateless_binomial')
@dispatch.add_dispatch_support
def stateless_random_binomial(shape, seed, counts, probs, output_dtype=dtypes.int32, name=None):
    if False:
        while True:
            i = 10
    'Outputs deterministic pseudorandom values from a binomial distribution.\n\n  The generated values follow a binomial distribution with specified count and\n  probability of success parameters.\n\n  This is a stateless version of `tf.random.Generator.binomial`: if run twice\n  with the same seeds and shapes, it will produce the same pseudorandom numbers.\n  The output is consistent across multiple runs on the same hardware (and\n  between CPU and GPU), but may change between versions of TensorFlow or on\n  non-CPU/GPU hardware.\n\n  Example:\n\n  ```python\n  counts = [10., 20.]\n  # Probability of success.\n  probs = [0.8]\n\n  binomial_samples = tf.random.stateless_binomial(\n      shape=[2], seed=[123, 456], counts=counts, probs=probs)\n\n  counts = ... # Shape [3, 1, 2]\n  probs = ...  # Shape [1, 4, 2]\n  shape = [3, 4, 3, 4, 2]\n  # Sample shape will be [3, 4, 3, 4, 2]\n  binomial_samples = tf.random.stateless_binomial(\n      shape=shape, seed=[123, 456], counts=counts, probs=probs)\n  ```\n\n  Args:\n    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    counts: Tensor. The counts of the binomial distribution. Must be\n      broadcastable with `probs`, and broadcastable with the rightmost\n      dimensions of `shape`.\n    probs: Tensor. The probability of success for the binomial distribution.\n      Must be broadcastable with `counts` and broadcastable with the rightmost\n      dimensions of `shape`.\n    output_dtype: The type of the output. Default: tf.int32\n    name: A name for the operation (optional).\n\n  Returns:\n    samples: A Tensor of the specified shape filled with random binomial\n      values.  For each i, each samples[..., i] is an independent draw from\n      the binomial distribution on counts[i] trials with probability of\n      success probs[i].\n  '
    with ops.name_scope(name, 'stateless_random_binomial', [shape, seed, counts, probs]) as name:
        shape = shape_util.shape_tensor(shape)
        probs = ops.convert_to_tensor(probs, dtype_hint=dtypes.float32, name='probs')
        counts = ops.convert_to_tensor(counts, dtype_hint=probs.dtype, name='counts')
        result = gen_stateless_random_ops.stateless_random_binomial(shape=shape, seed=seed, counts=counts, probs=probs, dtype=output_dtype)
        shape_util.maybe_set_static_shape(result, shape)
        return result

@tf_export('random.stateless_gamma')
@dispatch.add_dispatch_support
def stateless_random_gamma(shape, seed, alpha, beta=None, dtype=dtypes.float32, name=None):
    if False:
        print('Hello World!')
    'Outputs deterministic pseudorandom values from a gamma distribution.\n\n  The generated values follow a gamma distribution with specified concentration\n  (`alpha`) and inverse scale (`beta`) parameters.\n\n  This is a stateless version of `tf.random.gamma`: if run twice with the same\n  seeds and shapes, it will produce the same pseudorandom numbers. The output is\n  consistent across multiple runs on the same hardware (and between CPU and\n  GPU),\n  but may change between versions of TensorFlow or on non-CPU/GPU hardware.\n\n  A slight difference exists in the interpretation of the `shape` parameter\n  between `stateless_gamma` and `gamma`: in `gamma`, the `shape` is always\n  prepended to the shape of the broadcast of `alpha` with `beta`; whereas in\n  `stateless_gamma` the `shape` parameter must always encompass the shapes of\n  each of `alpha` and `beta` (which must broadcast together to match the\n  trailing dimensions of `shape`).\n\n  Note: Because internal calculations are done using `float64` and casting has\n  `floor` semantics, we must manually map zero outcomes to the smallest\n  possible positive floating-point value, i.e., `np.finfo(dtype).tiny`.  This\n  means that `np.finfo(dtype).tiny` occurs more frequently than it otherwise\n  should.  This bias can only happen for small values of `alpha`, i.e.,\n  `alpha << 1` or large values of `beta`, i.e., `beta >> 1`.\n\n  The samples are differentiable w.r.t. alpha and beta.\n  The derivatives are computed using the approach described in\n  (Figurnov et al., 2018).\n\n  Example:\n\n  ```python\n  samples = tf.random.stateless_gamma([10, 2], seed=[12, 34], alpha=[0.5, 1.5])\n  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents\n  # the samples drawn from each distribution\n\n  samples = tf.random.stateless_gamma([7, 5, 2], seed=[12, 34], alpha=[.5, 1.5])\n  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]\n  # represents the 7x5 samples drawn from each of the two distributions\n\n  alpha = tf.constant([[1.], [3.], [5.]])\n  beta = tf.constant([[3., 4.]])\n  samples = tf.random.stateless_gamma(\n      [30, 3, 2], seed=[12, 34], alpha=alpha, beta=beta)\n  # samples has shape [30, 3, 2], with 30 samples each of 3x2 distributions.\n\n  with tf.GradientTape() as tape:\n    tape.watch([alpha, beta])\n    loss = tf.reduce_mean(tf.square(tf.random.stateless_gamma(\n        [30, 3, 2], seed=[12, 34], alpha=alpha, beta=beta)))\n  dloss_dalpha, dloss_dbeta = tape.gradient(loss, [alpha, beta])\n  # unbiased stochastic derivatives of the loss function\n  alpha.shape == dloss_dalpha.shape  # True\n  beta.shape == dloss_dbeta.shape  # True\n  ```\n\n  Args:\n    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    alpha: Tensor. The concentration parameter of the gamma distribution. Must\n      be broadcastable with `beta`, and broadcastable with the rightmost\n      dimensions of `shape`.\n    beta: Tensor. The inverse scale parameter of the gamma distribution. Must be\n      broadcastable with `alpha` and broadcastable with the rightmost dimensions\n      of `shape`.\n    dtype: Floating point dtype of `alpha`, `beta`, and the output.\n    name: A name for the operation (optional).\n\n  Returns:\n    samples: A Tensor of the specified shape filled with random gamma values.\n      For each i, each `samples[..., i] is an independent draw from the gamma\n      distribution with concentration alpha[i] and scale beta[i].\n  '
    with ops.name_scope(name, 'stateless_random_gamma', [shape, seed, alpha, beta]) as name:
        shape = shape_util.shape_tensor(shape)
        alpha = ops.convert_to_tensor(alpha, dtype=dtype, name='alpha')
        beta = ops.convert_to_tensor(beta if beta is not None else 1, name='beta', dtype=dtype)
        broadcast_shape = array_ops.broadcast_dynamic_shape(array_ops.shape(alpha), array_ops.shape(beta))
        alpha_broadcast = array_ops.broadcast_to(alpha, broadcast_shape)
        alg = 'auto_select'
        (key, counter, alg) = random_ops_util.get_key_counter_alg(seed, alg)
        rnd = gen_stateless_random_ops_v2.stateless_random_gamma_v3(shape, key=key, counter=counter, alg=alg, alpha=alpha_broadcast)
        result = math_ops.maximum(np.finfo(alpha.dtype.as_numpy_dtype).tiny, rnd / beta)
        shape_util.maybe_set_static_shape(result, shape)
        return result

@tf_export('random.stateless_poisson')
@dispatch.add_dispatch_support
def stateless_random_poisson(shape, seed, lam, dtype=dtypes.int32, name=None):
    if False:
        i = 10
        return i + 15
    'Outputs deterministic pseudorandom values from a Poisson distribution.\n\n  The generated values follow a Poisson distribution with specified rate\n  parameter.\n\n  This is a stateless version of `tf.random.poisson`: if run twice with the same\n  seeds and shapes, it will produce the same pseudorandom numbers. The output is\n  consistent across multiple runs on the same hardware, but may change between\n  versions of TensorFlow or on non-CPU/GPU hardware.\n\n  A slight difference exists in the interpretation of the `shape` parameter\n  between `stateless_poisson` and `poisson`: in `poisson`, the `shape` is always\n  prepended to the shape of `lam`; whereas in `stateless_poisson` the shape of\n  `lam` must match the trailing dimensions of `shape`.\n\n  Example:\n\n  ```python\n  samples = tf.random.stateless_poisson([10, 2], seed=[12, 34], lam=[5, 15])\n  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents\n  # the samples drawn from each distribution\n\n  samples = tf.random.stateless_poisson([7, 5, 2], seed=[12, 34], lam=[5, 15])\n  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]\n  # represents the 7x5 samples drawn from each of the two distributions\n\n  rate = tf.constant([[1.], [3.], [5.]])\n  samples = tf.random.stateless_poisson([30, 3, 1], seed=[12, 34], lam=rate)\n  # samples has shape [30, 3, 1], with 30 samples each of 3x1 distributions.\n  ```\n\n  Args:\n    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    lam: Tensor. The rate parameter "lambda" of the Poisson distribution. Shape\n      must match the rightmost dimensions of `shape`.\n    dtype: Dtype of the samples (int or float dtypes are permissible, as samples\n      are discrete). Default: int32.\n    name: A name for the operation (optional).\n\n  Returns:\n    samples: A Tensor of the specified shape filled with random Poisson values.\n      For each i, each `samples[..., i]` is an independent draw from the Poisson\n      distribution with rate `lam[i]`.\n  '
    with ops.name_scope(name, 'stateless_random_poisson', [shape, seed, lam]) as name:
        shape = shape_util.shape_tensor(shape)
        result = gen_stateless_random_ops.stateless_random_poisson(shape, seed=seed, lam=lam, dtype=dtype)
        shape_util.maybe_set_static_shape(result, shape)
        return result

@tf_export('random.stateless_normal')
@dispatch.add_dispatch_support
def stateless_random_normal(shape, seed, mean=0.0, stddev=1.0, dtype=dtypes.float32, name=None, alg='auto_select'):
    if False:
        for i in range(10):
            print('nop')
    'Outputs deterministic pseudorandom values from a normal distribution.\n\n  This is a stateless version of `tf.random.normal`: if run twice with the\n  same seeds and shapes, it will produce the same pseudorandom numbers.  The\n  output is consistent across multiple runs on the same hardware (and between\n  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU\n  hardware.\n\n  Args:\n    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal\n      distribution.\n    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation\n      of the normal distribution.\n    dtype: The float type of the output: `float16`, `bfloat16`, `float32`,\n      `float64`. Defaults to `float32`.\n    name: A name for the operation (optional).\n    alg: The RNG algorithm used to generate the random numbers. See\n      `tf.random.stateless_uniform` for a detailed explanation.\n\n  Returns:\n    A tensor of the specified shape filled with random normal values.\n  '
    with ops.name_scope(name, 'stateless_random_normal', [shape, seed, mean, stddev]) as name:
        shape = shape_util.shape_tensor(shape)
        mean = ops.convert_to_tensor(mean, dtype=dtype, name='mean')
        stddev = ops.convert_to_tensor(stddev, dtype=dtype, name='stddev')
        (key, counter, alg) = random_ops_util.get_key_counter_alg(seed, alg)
        rnd = gen_stateless_random_ops_v2.stateless_random_normal_v2(shape, key=key, counter=counter, dtype=dtype, alg=alg)
        result = math_ops.add(rnd * stddev, mean, name=name)
        shape_util.maybe_set_static_shape(result, shape)
        return result

@tf_export('random.stateless_truncated_normal')
@dispatch.add_dispatch_support
def stateless_truncated_normal(shape, seed, mean=0.0, stddev=1.0, dtype=dtypes.float32, name=None, alg='auto_select'):
    if False:
        return 10
    'Outputs deterministic pseudorandom values, truncated normally distributed.\n\n  This is a stateless version of `tf.random.truncated_normal`: if run twice with\n  the same seeds and shapes, it will produce the same pseudorandom numbers.  The\n  output is consistent across multiple runs on the same hardware (and between\n  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU\n  hardware.\n\n  The generated values follow a normal distribution with specified mean and\n  standard deviation, except that values whose magnitude is more than 2 standard\n  deviations from the mean are dropped and re-picked.\n\n  Args:\n    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the\n      truncated normal distribution.\n    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation\n      of the normal distribution, before truncation.\n    dtype: The type of the output.\n    name: A name for the operation (optional).\n    alg: The RNG algorithm used to generate the random numbers. See\n      `tf.random.stateless_uniform` for a detailed explanation.\n\n  Returns:\n    A tensor of the specified shape filled with random truncated normal values.\n  '
    with ops.name_scope(name, 'stateless_truncated_normal', [shape, seed, mean, stddev]) as name:
        shape = shape_util.shape_tensor(shape)
        mean = ops.convert_to_tensor(mean, dtype=dtype, name='mean')
        stddev = ops.convert_to_tensor(stddev, dtype=dtype, name='stddev')
        (key, counter, alg) = random_ops_util.get_key_counter_alg(seed, alg)
        rnd = gen_stateless_random_ops_v2.stateless_truncated_normal_v2(shape, key=key, counter=counter, dtype=dtype, alg=alg)
        result = math_ops.add(rnd * stddev, mean, name=name)
        shape_util.maybe_set_static_shape(result, shape)
        return result

@tf_export(v1=['random.stateless_multinomial'])
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions='Use `tf.random.stateless_categorical` instead.')
def stateless_multinomial(logits, num_samples, seed, output_dtype=dtypes.int64, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Draws deterministic pseudorandom samples from a multinomial distribution.\n\n  This is a stateless version of `tf.random.categorical`: if run twice with the\n  same seeds and shapes, it will produce the same pseudorandom numbers.  The\n  output is consistent across multiple runs on the same hardware (and between\n  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU\n  hardware.\n\n  Example:\n\n  ```python\n  # samples has shape [1, 5], where each value is either 0 or 1 with equal\n  # probability.\n  samples = tf.random.stateless_categorical(\n      tf.math.log([[0.5, 0.5]]), 5, seed=[7, 17])\n  ```\n\n  Args:\n    logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i,\n      :]` represents the unnormalized log-probabilities for all classes.\n    num_samples: 0-D.  Number of independent samples to draw for each row slice.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    output_dtype: The integer type of the output: `int32` or `int64`. Defaults\n      to `int64`.\n    name: Optional name for the operation.\n\n  Returns:\n    The drawn samples of shape `[batch_size, num_samples]`.\n  '
    with ops.name_scope(name, 'stateless_multinomial', [logits, seed]):
        return stateless_multinomial_categorical_impl(logits, num_samples, output_dtype, seed)

@tf_export('random.stateless_categorical')
@dispatch.add_dispatch_support
def stateless_categorical(logits, num_samples, seed, dtype=dtypes.int64, name=None):
    if False:
        print('Hello World!')
    'Draws deterministic pseudorandom samples from a categorical distribution.\n\n  This is a stateless version of `tf.categorical`: if run twice with the\n  same seeds and shapes, it will produce the same pseudorandom numbers.  The\n  output is consistent across multiple runs on the same hardware (and between\n  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU\n  hardware.\n\n\n  Example:\n\n  ```python\n  # samples has shape [1, 5], where each value is either 0 or 1 with equal\n  # probability.\n  samples = tf.random.stateless_categorical(\n      tf.math.log([[0.5, 0.5]]), 5, seed=[7, 17])\n  ```\n\n  Args:\n    logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i,\n      :]` represents the unnormalized log-probabilities for all classes.\n    num_samples: 0-D.  Number of independent samples to draw for each row slice.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    dtype: The integer type of the output: `int32` or `int64`. Defaults to\n      `int64`.\n    name: Optional name for the operation.\n\n  Returns:\n    The drawn samples of shape `[batch_size, num_samples]`.\n  '
    with ops.name_scope(name, 'stateless_categorical', [logits, seed]):
        return stateless_multinomial_categorical_impl(logits, num_samples, dtype, seed)

def stateless_multinomial_categorical_impl(logits, num_samples, dtype, seed):
    if False:
        return 10
    'Implementation for stateless multinomial/categorical ops (v1/v2).'
    logits = ops.convert_to_tensor(logits, name='logits')
    dtype = dtypes.as_dtype(dtype) if dtype else dtypes.int64
    accepted_dtypes = (dtypes.int32, dtypes.int64)
    if dtype not in accepted_dtypes:
        raise ValueError(f'Argument `dtype` got invalid value {dtype}. Accepted dtypes are {accepted_dtypes}.')
    return gen_stateless_random_ops.stateless_multinomial(logits, num_samples, seed, output_dtype=dtype)

@dispatch.add_dispatch_support
@tf_export('random.stateless_parameterized_truncated_normal')
def stateless_parameterized_truncated_normal(shape, seed, means=0.0, stddevs=1.0, minvals=-2.0, maxvals=2.0, name=None):
    if False:
        i = 10
        return i + 15
    'Outputs random values from a truncated normal distribution.\n\n  The generated values follow a normal distribution with specified mean and\n  standard deviation, except that values whose magnitude is more than 2 standard\n  deviations from the mean are dropped and re-picked.\n\n\n  Examples:\n\n  Sample from a Truncated normal, with deferring shape parameters that\n  broadcast.\n\n  >>> means = 0.\n  >>> stddevs = tf.math.exp(tf.random.uniform(shape=[2, 3]))\n  >>> minvals = [-1., -2., -1000.]\n  >>> maxvals = [[10000.], [1.]]\n  >>> y = tf.random.stateless_parameterized_truncated_normal(\n  ...   shape=[10, 2, 3], seed=[7, 17],\n  ...   means=means, stddevs=stddevs, minvals=minvals, maxvals=maxvals)\n  >>> y.shape\n  TensorShape([10, 2, 3])\n\n  Args:\n    shape: A 1-D integer `Tensor` or Python array. The shape of the output\n      tensor.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    means: A `Tensor` or Python value of type `dtype`. The mean of the truncated\n      normal distribution. This must broadcast with `stddevs`, `minvals` and\n      `maxvals`, and the broadcasted shape must be dominated by `shape`.\n    stddevs: A `Tensor` or Python value of type `dtype`. The standard deviation\n      of the truncated normal distribution. This must broadcast with `means`,\n      `minvals` and `maxvals`, and the broadcasted shape must be dominated by\n      `shape`.\n    minvals: A `Tensor` or Python value of type `dtype`. The minimum value of\n      the truncated normal distribution. This must broadcast with `means`,\n      `stddevs` and `maxvals`, and the broadcasted shape must be dominated by\n      `shape`.\n    maxvals: A `Tensor` or Python value of type `dtype`. The maximum value of\n      the truncated normal distribution. This must broadcast with `means`,\n      `stddevs` and `minvals`, and the broadcasted shape must be dominated by\n      `shape`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A tensor of the specified shape filled with random truncated normal values.\n  '
    with ops.name_scope(name, 'stateless_parameterized_truncated_normal', [shape, means, stddevs, minvals, maxvals]) as name:
        shape_tensor = shape_util.shape_tensor(shape)
        means_tensor = ops.convert_to_tensor(means, name='means')
        stddevs_tensor = ops.convert_to_tensor(stddevs, name='stddevs')
        minvals_tensor = ops.convert_to_tensor(minvals, name='minvals')
        maxvals_tensor = ops.convert_to_tensor(maxvals, name='maxvals')
        rnd = gen_stateless_random_ops.stateless_parameterized_truncated_normal(shape_tensor, seed, means_tensor, stddevs_tensor, minvals_tensor, maxvals_tensor)
        shape_util.maybe_set_static_shape(rnd, shape)
        return rnd
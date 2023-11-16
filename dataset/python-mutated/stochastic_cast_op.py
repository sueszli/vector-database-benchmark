"""Stochastic cast op which stochastically casts input tensors to the desired data type."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_stochastic_cast_op
from tensorflow.python.ops import random_ops_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

def allowed_to_types(is_integer=True):
    if False:
        for i in range(10):
            print('nop')
    if is_integer:
        return {dtypes.int32, dtypes.int16, dtypes.int8}
    else:
        return {dtypes.float16, dtypes.bfloat16, dtypes.float8_e5m2, dtypes.float8_e4m3fn}

@tf_export('random.stochastic_cast')
@dispatch.add_dispatch_support
def stochastic_cast(t, dtype, seed, alg='auto_select', name=None):
    if False:
        return 10
    "Casts input to the desired precision with stochastic rounding.\n\n  This means the value of the cast result will be rounded to two of the closest\n  values with with a probability proportional to the distance between the number\n  and the two closest to the input. For example, if a number falls between 2 and\n  3, and is closer to 2 than to 3, it has a higher probability of being rounded\n  to 2. On the other hand, if it's closer to 3 than to 2, it has a higher\n  probability of being rounded to 3. This is intended to eliminate rounding bias\n  introduced by determinisitc rounding methods. If cast to integers, the values\n  will saturate if out of range, e.g. 254.8 in floating point will become 127 in\n  int8. If inputs are NaN, the results will be zero. Given the same random seed,\n  the results will be deterministic, but not otherwise.\n\n  Args:\n    t: The input tensor. This is the same as the output shape.\n    dtype: The output type, currently int32, int16 and int8 are supported.\n    seed: A shape [2] Tensor, the seed to the random number generator. Must have\n      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)\n    alg: The RNG algorithm used to generate the random numbers. See\n      `tf.random.stateless_uniform` for a detailed explanation.\n    name: A name for the operation (optional).\n\n  Returns:\n    A tensor of the specified data type whose values are rounded to the\n    specified precisions with stochastic rounding.\n  "
    with ops.name_scope(name, 'stochastic_cast', [t, seed]) as name:
        t = ops.convert_to_tensor(t)
        (key, counter, algorithm) = random_ops_util.get_key_counter_alg(seed, alg)
        if dtype in allowed_to_types(is_integer=True):
            return gen_stochastic_cast_op.stochastic_cast_to_int(t, key=key, counter=counter, alg=algorithm, Tout=dtype)
        else:
            raise NotImplementedError(f'Stochastic cast to small float {dtype} has not yet been supported.')
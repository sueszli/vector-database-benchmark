"""Utilities related to distributed training."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.keras import backend
from tensorflow.python.ops import variables

def global_batch_size_supported(distribution_strategy):
    if False:
        return 10
    return distribution_strategy.extended._global_batch_size

def call_replica_local_fn(fn, *args, **kwargs):
    if False:
        print('Hello World!')
    'Call a function that uses replica-local variables.\n\n  This function correctly handles calling `fn` in a cross-replica\n  context.\n\n  Args:\n    fn: The function to call.\n    *args: Positional arguments to the `fn`.\n    **kwargs: Keyword argument to `fn`.\n\n  Returns:\n    The result of calling `fn`.\n  '
    strategy = None
    if 'strategy' in kwargs:
        strategy = kwargs.pop('strategy')
    elif distribute_lib.has_strategy():
        strategy = distribute_lib.get_strategy()
    is_tpu = backend.is_tpu_strategy(strategy)
    if not is_tpu and strategy and distribute_lib.in_cross_replica_context():
        with strategy.scope():
            return strategy.extended.call_for_each_replica(fn, args, kwargs)
    return fn(*args, **kwargs)

def is_distributed_variable(v):
    if False:
        i = 10
        return i + 15
    'Returns whether `v` is a distributed variable.'
    return isinstance(v, values_lib.DistributedValues) and isinstance(v, variables.Variable)
"""A module for interm merge-call related internal APIs."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.util.tf_export import tf_export

@tf_export('__internal__.distribute.strategy_supports_no_merge_call', v1=[])
def strategy_supports_no_merge_call():
    if False:
        return 10
    'Returns if the current `Strategy` can operate in pure replica context.'
    if not distribute_lib.has_strategy():
        return True
    strategy = distribute_lib.get_strategy()
    return not strategy.extended._use_merge_call()

@tf_export('__internal__.distribute.interim.maybe_merge_call', v1=[])
def maybe_merge_call(fn, strategy, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    "Maybe invoke `fn` via `merge_call` which may or may not be fulfilled.\n\n  The caller of this utility function requests to invoke `fn` via `merge_call`\n  at `tf.distribute.Strategy`'s best efforts. It is `tf.distribute`'s internal\n  whether the request is honored, depending on the `Strategy`. See\n  `tf.distribute.ReplicaContext.merge_call()` for more information.\n\n  This is an interim API which is subject to removal and does not guarantee\n  backward-compatibility.\n\n  Args:\n    fn: the function to be invoked.\n    strategy: the `tf.distribute.Strategy` to call `fn` with.\n    *args: the positional arguments to be passed in to `fn`.\n    **kwargs: the keyword arguments to be passed in to `fn`.\n\n  Returns:\n    The return value of the `fn` call.\n  "
    if strategy_supports_no_merge_call():
        return fn(strategy, *args, **kwargs)
    else:
        return distribute_lib.get_replica_context().merge_call(fn, args=args, kwargs=kwargs)
"""Utilities for collectives."""
import copy
import enum
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@tf_export('distribute.experimental.CommunicationImplementation', 'distribute.experimental.CollectiveCommunication')
class CommunicationImplementation(enum.Enum):
    """Cross device communication implementation.

  Warning: The alias `tf.distribute.experimental.CollectiveCommunication` is
  deprecated and will be removed in a future version. Use
  `tf.distribute.experimental.CommunicationImplementation` instead.

  * `AUTO`: Automatically chosen by Tensorflow.
  * `RING`: TensorFlow's ring algorithms for all-reduce and
    all-gather.
  * `NCCL`: NVIDIA®'s NCCL library. This is now only used for all-reduce on
    GPUs; all-reduce on CPU, all-gather and broadcast fallbacks to RING.
  """
    AUTO = 'AUTO'
    RING = 'RING'
    NCCL = 'NCCL'
CollectiveCommunication = CommunicationImplementation

@tf_export('distribute.experimental.CommunicationOptions')
class _OptionsExported(object):
    """Options for cross device communications like All-reduce.

  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.

  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.

  Examples:

  ```python
  options = tf.distribute.experimental.CommunicationOptions(
      bytes_per_pack=50 * 1024 * 1024,
      timeout_seconds=120.0,
      implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
  )
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, options=options)
  optimizer.apply_gradients(zip(grads, vars),
      experimental_aggregate_gradients=False)
  ```

  """

    def __new__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        return Options(*args, **kwargs)

    def __init__(self, bytes_per_pack=0, timeout_seconds=None, implementation=CommunicationImplementation.AUTO):
        if False:
            return 10
        "Creates a CollectiveHints.\n\n    Args:\n      bytes_per_pack: a non-negative integer. Breaks collective operations into\n        packs of certain size. If it's zero, the value is determined\n        automatically. This hint is respected by all multi-replica strategies\n        except `TPUStrategy`.\n      timeout_seconds: a float or None, timeout in seconds. If not None, the\n        collective raises `tf.errors.DeadlineExceededError` if it takes longer\n        than this timeout. Zero disables timeout. This can be useful when\n        debugging hanging issues.  This should only be used for debugging since\n        it creates a new thread for each collective, i.e. an overhead of\n        `timeout_seconds * num_collectives_per_second` more threads. This only\n        works for `tf.distribute.experimental.MultiWorkerMirroredStrategy`.\n      implementation: a\n        `tf.distribute.experimental.CommunicationImplementation`. This is a hint\n        on the preferred communication implementation. Possible values include\n        `AUTO`, `RING`, and `NCCL`. NCCL is generally more performant for GPU,\n        but doesn't work for CPU. This only works for\n        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.\n\n    Raises:\n      ValueError: When arguments have invalid value.\n    "
        pass

class Options(object):
    """Implementation of OptionsInterface."""

    def __init__(self, bytes_per_pack=0, timeout_seconds=None, implementation=CommunicationImplementation.AUTO):
        if False:
            return 10
        if bytes_per_pack < 0:
            raise ValueError(f'Argument `bytes_per_pack` must be >=0, Received {bytes_per_pack}.')
        if isinstance(implementation, str):
            implementation = CommunicationImplementation(implementation.upper())
        if not isinstance(implementation, CommunicationImplementation):
            raise ValueError('Argument `implementation` must be instance of `tf.distribute.experimental.CommunicationImplementation`.')
        self.bytes_per_pack = bytes_per_pack
        self.timeout_seconds = timeout_seconds
        self.implementation = implementation
    __init__.__doc__ = _OptionsExported.__init__.__doc__

    def merge(self, options):
        if False:
            print('Hello World!')
        "Merges with another options and returns a new one.\n\n    Values specified in the `options` takes precedence if they're not the\n    default.\n\n    Args:\n      options: a `tf.distribute.experimental.CollectiveCommunication`.\n\n    Returns:\n      A new `tf.distribute.experimental.CollectiveCommunication`.\n    "
        merged = copy.deepcopy(self)
        if options is None:
            return merged
        if options.bytes_per_pack != 0:
            merged.bytes_per_pack = options.bytes_per_pack
        if options.timeout_seconds is not None:
            merged.timeout_seconds = options.timeout_seconds
        if options.implementation != CommunicationImplementation.AUTO:
            merged.implementation = options.implementation
        return merged

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'Options(bytes_per_pack={self.bytes_per_pack},timeout_seconds={self.timeout_seconds}, implementation={self.implementation})'

@tf_export('distribute.experimental.CollectiveHints')
class Hints(object):
    """Hints for collective operations like AllReduce.

  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.

  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.

  Examples:

  - bytes_per_pack

  ```python
  hints = tf.distribute.experimental.CollectiveHints(
      bytes_per_pack=50 * 1024 * 1024)
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, experimental_hints=hints)
  optimizer.apply_gradients(zip(grads, vars),
      experimental_aggregate_gradients=False)
  ```

  - timeout_seconds

  ```python
  strategy = tf.distribute.MirroredStrategy()
  hints = tf.distribute.experimental.CollectiveHints(
      timeout_seconds=120.0)
  try:
    strategy.reduce("sum", v, axis=None, experimental_hints=hints)
  except tf.errors.DeadlineExceededError:
    do_something()
  ```

  """

    @deprecation.deprecated(None, 'use distribute.experimental.CommunicationOptions instead')
    def __new__(cls, bytes_per_pack=0, timeout_seconds=None):
        if False:
            return 10
        return Options(bytes_per_pack=bytes_per_pack, timeout_seconds=timeout_seconds)

    def __init__(self, bytes_per_pack=0, timeout_seconds=None):
        if False:
            return 10
        "Creates a CollectiveHints.\n\n    Args:\n      bytes_per_pack: a non-negative integer. Breaks collective operations into\n        packs of certain size. If it's zero, the value is determined\n        automatically. This only applies to all-reduce with\n        `MultiWorkerMirroredStrategy` currently.\n      timeout_seconds: a float or None, timeout in seconds. If not None, the\n        collective raises `tf.errors.DeadlineExceededError` if it takes longer\n        than this timeout. This can be useful when debugging hanging issues.\n        This should only be used for debugging since it creates a new thread for\n        each collective, i.e. an overhead of `timeout_seconds *\n        num_collectives_per_second` more threads.  This only works for\n        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.\n\n    Raises:\n      ValueError: When arguments have invalid value.\n    "
        pass
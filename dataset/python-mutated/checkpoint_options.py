"""Options for saving Checkpoints."""
import copy
import inspect
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export

@tf_export('train.CheckpointOptions')
class CheckpointOptions(object):
    """Options for constructing a Checkpoint.

  Used as the `options` argument to either `tf.train.Checkpoint.save()` or
  `tf.train.Checkpoint.restore()` methods to adjust how variables are
  saved/restored.

  Example: Run IO ops on "localhost" while saving a checkpoint:

  ```
  step = tf.Variable(0, name="step")
  checkpoint = tf.train.Checkpoint(step=step)
  options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
  checkpoint.save("/tmp/ckpt", options=options)
  ```
  """
    __slots__ = ('experimental_io_device', 'experimental_enable_async_checkpoint', 'experimental_write_callbacks', 'enable_async')

    @deprecated_args(None, 'Use enable_async instead', 'experimental_enable_async_checkpoint')
    def __init__(self, experimental_io_device=None, experimental_enable_async_checkpoint=False, experimental_write_callbacks=None, enable_async=False):
        if False:
            for i in range(10):
                print('nop')
        'Creates an object that stores options for a Checkpoint.\n\n    Args:\n      experimental_io_device: string. Applies in a distributed setting.\n        Tensorflow device to use to access the filesystem. If `None` (default)\n        then for each variable the filesystem is accessed from the CPU:0 device\n        of the host where that variable is assigned. If specified, the\n        filesystem is instead accessed from that device for all variables.\n\n        This is for example useful if you want to save to a local directory,\n        such as "/tmp" when running in a distributed setting. In that case pass\n        a device for the host where the "/tmp" directory is accessible.\n\n      experimental_enable_async_checkpoint: bool Type. Deprecated, please use\n        the enable_async option.\n\n      experimental_write_callbacks: List[Callable]. A list of callback functions\n        that will be executed after each saving event finishes (i.e. after\n        `save()` or `write()`). For async checkpoint, the callbacks will be\n        executed only after the async thread finishes saving.\n\n        The return values of the callback(s) will be ignored. The callback(s)\n        can optionally take the `save_path` (the result of `save()` or\n        `write()`) as an argument. The callbacks will be executed in the same\n        order of this list after the checkpoint has been written.\n\n      enable_async: bool Type. Indicates whether async checkpointing is enabled.\n        Default is False, i.e., no async checkpoint.\n\n        Async checkpoint moves the checkpoint file writing off the main thread,\n        so that the model can continue to train while the checkpoing file\n        writing runs in the background. Async checkpoint reduces TPU device idle\n        cycles and speeds up model training process, while memory consumption\n        may increase.\n    '
        self.experimental_io_device = experimental_io_device
        self.enable_async = experimental_enable_async_checkpoint or enable_async
        self.experimental_enable_async_checkpoint = self.enable_async
        if experimental_write_callbacks is not None:
            for callback in experimental_write_callbacks:
                assert len(inspect.signature(callback).parameters) <= 1
        self.experimental_write_callbacks = experimental_write_callbacks

    def __copy__(self):
        if False:
            return 10
        result = copy.copy(super())
        result.experimental_write_callbacks = copy.copy(self.experimental_write_callbacks)
        return result
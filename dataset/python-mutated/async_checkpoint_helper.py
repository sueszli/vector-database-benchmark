"""Utilities for saving/loading Trackable objects asynchronously."""
import atexit
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base
from tensorflow.python.util import object_identity
_END_TIME_OF_LAST_ASYNC_WRITE = None
_END_TIME_OF_LAST_ASYNC_WRITE_LOCK = threading.Lock()
_ASYNC_CHECKPOINT = 'async_checkpoint'
_TPU_EMBEDDING_ATTR = '_create_copy_for_async_checkpoint'

def _get_duration_microseconds(start_time_seconds, end_time_seconds):
    if False:
        return 10
    'Calculate the duration between start and end time.\n\n  Args:\n    start_time_seconds: The start time in seconds.\n    end_time_seconds: The end time in seconds.\n\n  Returns:\n    The duration between the start and the end time. Return 0 if\n    end_time_seconds < start_time_seconds.\n  '
    if end_time_seconds < start_time_seconds:
        return 0
    return round((end_time_seconds - start_time_seconds) * 1000000)

def _get_all_trackables(root, exclude_set):
    if False:
        for i in range(10):
            print('nop')
    'Return the list of checkpointable trackables dependent on `root`.\n\n  Args:\n    root: The root trackable from where we get all its dependent trackables.\n    exclude_set: An ObjectIdentitySet of Trackables to exclude before returning.\n        Each element in `exclude_set` is a specific instance of a `Trackable`\n        and appears precisely once in `TrackableView(root).descendants()`.\n\n  Returns:\n    saveable_trackables: All trackables that are saveable in `all_trackables`\n        (see definition of "saveable" in `_trackable_needs_to_be_saved()`). A\n        subset of `all_trackables`.\n    all_trackables: All trackables returned by `TrackableView`\'s `descendants()`\n        after excluding `exclude_set`. A superset of `saveable_trackables`.\n  '
    all_trackables = trackable_view.TrackableView(root=root).descendants()
    trackable_index = 0
    while trackable_index < len(all_trackables) and exclude_set:
        if all_trackables[trackable_index] in exclude_set:
            exclude_set.discard(all_trackables[trackable_index])
            all_trackables.pop(trackable_index)
        else:
            trackable_index += 1

    def _trackable_needs_to_be_saved(obj):
        if False:
            print('Hello World!')
        "Returns whether a trackable needs to be saved.\n\n    Returns a bool to indicate whether obj's class has `_serialize_to_tensors`,\n    `gather_saveables_for_checkpoint`, or `_copy_trackable_to_cpu` defined.\n\n    Args:\n      obj: A Trackable object.\n    "
        if hasattr(obj, '__dict__'):
            if '_serialize_to_tensors' in obj.__dict__ or '_gather_saveables_for_checkpoint' in obj.__dict__ or '_copy_trackable_to_cpu' in obj.__dict__:
                return True
        for t in type(obj).mro():
            if t is base.Trackable:
                continue
            elif '_serialize_to_tensors' in t.__dict__ or '_gather_saveables_for_checkpoint' in t.__dict__ or '_copy_trackable_to_cpu' in t.__dict__:
                return True
        return False
    saveable_trackables = [x for x in all_trackables if _trackable_needs_to_be_saved(x)]
    return (saveable_trackables, all_trackables)

class AsyncCheckpointHelper:
    """Helper class for async checkpoint."""

    def __init__(self, checkpointer_impl, root=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Initialize AsyncCheckpoint.\n\n    Args:\n      checkpointer_impl: The Checkpoint class to power the AsyncCheckpoint.\n      root: The root object to checkpoint. `root` may be a trackable object or\n        `WeakRef` of a trackable object.\n      **kwargs: The keyword arguments representing the checkpointed variables.\n\n    Raises:\n      AttributeError: when checkpointer_impl is None.\n    '
        if root:
            trackable_root = root() if isinstance(root, weakref.ref) else root
            kwargs['root'] = trackable_root
            trackable_root._maybe_initialize_trackable()
        if checkpointer_impl is None:
            raise AttributeError('checkpointer_impl cannot be None for AsyncCheckpointHelper.')
        self._checkpointer_impl = checkpointer_impl
        self._checkpoint_items = kwargs
        self._checkpoint = None
        self.checkpointer()
        self._checkpoint_options = None
        self._initialized = False
        self._original_nodes = None
        self._object_map = None
        self._tpu_embedding_objects = None
        self._saveable_trackables = None
        self._default_device = device_util.current() or 'CPU:0'
        self._default_device = device_util.canonicalize(self._default_device)
        self._save_file_prefix = None
        self._use_checkpoint_save = False
        self._async_save_thread = None
        self._queue = queue.Queue(maxsize=1)
        atexit.register(self._join_async_save_thread)
        self._async_error = None
        global _END_TIME_OF_LAST_ASYNC_WRITE
        with _END_TIME_OF_LAST_ASYNC_WRITE_LOCK:
            if _END_TIME_OF_LAST_ASYNC_WRITE is None:
                _END_TIME_OF_LAST_ASYNC_WRITE = time.time()

    @def_function.function
    def _copy_to_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        'Copy the checkpointed variables from the accelerator to the host CPU.\n\n    TODO(chienchunh): Get the concrete function before firstly called to avoid\n                      hangining the accelerators idle during function tracing.\n    '
        for t in self._saveable_trackables:
            try:
                t._copy_trackable_to_cpu(object_map=self._object_map)
            except NotImplementedError as e:
                logging.warning('Trackable %s skipped due to: %s', t, e)
        for tpu_embedding in self._tpu_embedding_objects:
            tpu_embedding._retrieve_variables()

    def checkpointer(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets or creates the underlying Checkpoint instance.'
        if self._checkpoint is None:
            self._checkpoint = self._checkpointer_impl(**self._checkpoint_items)
        return self._checkpoint

    def _ensure_initialized(self):
        if False:
            print('Hello World!')
        'Initialize the async checkpoint internal state.'
        self._object_map = object_identity.ObjectIdentityDictionary()
        self._tpu_embedding_objects = []
        exclude_set = object_identity.ObjectIdentitySet()
        exclude_set.add(self.checkpointer())
        exclude_set.add(self.checkpointer().save_counter)
        (self._saveable_trackables, all_trackables) = _get_all_trackables(root=self.checkpointer(), exclude_set=exclude_set)
        for t in all_trackables:
            if hasattr(type(t), _TPU_EMBEDDING_ATTR):
                self._handle_tpu_embedding(t)
            if 'get_slot_names' in dir(t):
                slot_names = t.get_slot_names()
                for slot_name in slot_names:
                    for original_variable in all_trackables:
                        if not isinstance(original_variable, variables.Variable):
                            continue
                        try:
                            original_slot_variable = t.get_slot(original_variable, slot_name)
                        except (AttributeError, KeyError):
                            continue
                        if isinstance(original_slot_variable, base.Trackable):
                            self._saveable_trackables.append(original_slot_variable)
        save_counter = self.checkpointer().save_counter.numpy()
        logging.info("Initializing async checkpoint's save_counter: %d", save_counter)
        self.checkpointer()._saver._object_map = self._object_map
        for t in self._saveable_trackables:
            try:
                t._copy_trackable_to_cpu(object_map=self._object_map)
            except NotImplementedError as e:
                logging.warning('Trackable %s skipped due to: %s', t, e)
        for tpu_embedding in self._tpu_embedding_objects:
            tpu_embedding._retrieve_variables()
        self._async_save_thread = threading.Thread(target=self._async_save, daemon=True)
        self._async_save_thread.start()
        self._initialized = True

    def _check_async_thread_error(self):
        if False:
            i = 10
            return i + 15
        'Expose the most recent error from the async saving thread to the caller.\n    '
        if self._async_error:
            e = self._async_error
            self._async_error = None
            logging.error('Propagating the most recent error from the async thread before joining: %s', str(e))
            raise e

    def _join_async_save_thread(self):
        if False:
            return 10
        "Join the async save thread.\n\n    The steps for terminating the async save thread:\n    1). Put will succeed when the last async save event is done. Putting a false\n        triggers the async save thread's while loop to end. We use put instead\n        of sync because sync does not have a timeout argument.\n    2). Join the async save thread. (The thread may finish before joining.)\n    "
        try:
            self._queue.put(False, timeout=300)
            logging.info('Joining the async save thread.')
            if self._async_save_thread is not None:
                self._async_save_thread.join()
        except queue.Full:
            logging.error('Timeout waiting for the async save thread; terminating the thread instead. The last checkpoint may be incomeplete.')
        finally:
            self._check_async_thread_error()

    def _async_save(self):
        if False:
            for i in range(10):
                print('nop')
        'The thread function for the async checkpoint save.'
        with context.executor_scope(executor.new_executor(enable_async=False, enable_streaming_enqueue=False)):
            while self._queue.get():
                logging.info('Starting async checkpoint save on the device: %s', self._default_device)
                async_save_start_time = time.time()
                try:
                    with ops.device(self._default_device):
                        with checkpoint_context.async_metrics_context():
                            if self._use_checkpoint_save:
                                self.checkpointer().save(self._save_file_prefix, self._checkpoint_options)
                            else:
                                self.checkpointer()._write(self._save_file_prefix, options=self._checkpoint_options)
                except Exception as e:
                    self._async_error = e
                finally:
                    self._queue.task_done()
                async_save_end_time = time.time()
                metrics.AddAsyncCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(async_save_start_time, async_save_end_time))
                global _END_TIME_OF_LAST_ASYNC_WRITE
                with _END_TIME_OF_LAST_ASYNC_WRITE_LOCK:
                    metrics.AddTrainingTimeSaved(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_ASYNC_WRITE, async_save_start_time))
                    _END_TIME_OF_LAST_ASYNC_WRITE = async_save_start_time
        logging.info('Async save thread reached the end of the execution.')

    def _handle_tpu_embedding(self, tpu_embedding):
        if False:
            return 10
        "Handle TPUEmbedding.\n\n    This is the only place where we populate object map in the class of\n    `AsyncCheckpointHelper`. For all other checkpointable trackables, we\n    populate object map using the trackable's own `_copy_trackable_to_cpu()`.\n\n    Args:\n      tpu_embedding: TPUEmbedding object to be handled.\n\n    Raises:\n      AttributeError: if the input trackable is not TPUEmbedding type.\n    "
        if not hasattr(type(tpu_embedding), _TPU_EMBEDDING_ATTR) or not callable(tpu_embedding._create_copy_for_async_checkpoint):
            raise AttributeError('Expecting TPUEmbedding type; got %s' % type(tpu_embedding))
        new_embedding = tpu_embedding._create_copy_for_async_checkpoint(feature_config=tpu_embedding._feature_config, optimizer=tpu_embedding._table_config[0] if tpu_embedding._table_config else None, pipeline_execution_with_tensor_core=tpu_embedding._pipeline_execution_with_tensor_core)
        self._object_map[tpu_embedding] = new_embedding
        if tpu_embedding not in self._tpu_embedding_objects:
            self._tpu_embedding_objects.append(tpu_embedding)

    @property
    def save_counter(self):
        if False:
            while True:
                i = 10
        'An integer variable numbering the checkpoint events.\n\n    This is maintained by the underlying tf.train.Checkpoing object employed by\n    AsyncCheckpoint class. The number starts at 0 and gets incremented for each\n    checkpoint event.\n\n    Returns:\n      The save counter variable.\n    '
        return self.checkpointer().save_counter

    def write(self, save_path, options=None):
        if False:
            i = 10
            return i + 15
        'Save the checkpointed variables.\n\n    Args:\n      save_path: The file prefix of the checkpoint file.\n      options: Optional CheckpointOption instance.\n\n    Returns:\n      The full path of the checkpoint file.\n    '
        return self._write(save_path, options)

    def _write(self, save_path, options=None):
        if False:
            while True:
                i = 10
        'Save the checkpointed variables.\n\n    This method has exactly the same logic as save(), except it does not\n    increment the underlying save_counter, which is done by the caller, e.g.,\n    CheckpointManager.\n\n    Args:\n      save_path: The file prefix of the checkpoint file.\n      options: Optional CheckpointOption instance.\n\n    Returns:\n      The full path of the checkpoint file.\n    '
        write_start_time = time.time()
        if not self._initialized:
            self._ensure_initialized()
        else:
            self._queue.join()
            self._copy_to_cpu()
        self._check_async_thread_error()
        context.async_wait()
        self._save_file_prefix = save_path
        self._use_checkpoint_save = False
        self._checkpoint_options = copy.copy(options) if options else None
        if self._checkpoint_options:
            self._checkpoint_options.experimental_enable_async_checkpoint = False
        self._queue.put(True)
        write_end_time = time.time()
        metrics.AddCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(write_start_time, write_end_time))
        return save_path

    def save(self, save_path, options=None):
        if False:
            i = 10
            return i + 15
        'Save the checkpointed variables.\n\n    Args:\n      save_path: The file prefix of the checkpoint file.\n      options: Optional CheckpointOption instance.\n\n    Returns:\n      The full path of the checkpoint file.\n    '
        save_start_time = time.time()
        if not self._initialized:
            self._ensure_initialized()
        else:
            self._queue.join()
            self._copy_to_cpu()
        self._check_async_thread_error()
        save_counter = self.checkpointer().save_counter.numpy() + 1
        full_path = '{}-{}'.format(save_path, save_counter)
        context.async_wait()
        self._save_file_prefix = save_path
        self._use_checkpoint_save = True
        self._checkpoint_options = copy.copy(options) if options else None
        if self._checkpoint_options:
            self._checkpoint_options.experimental_enable_async_checkpoint = False
        self._queue.put(True)
        save_end_time = time.time()
        metrics.AddCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(save_start_time, save_end_time))
        return full_path

    def read(self, save_path, options=None):
        if False:
            for i in range(10):
                print('nop')
        'Restore the checkpointed variables.\n\n    This method has exactly the same logic as restore(). This method is\n    implemented only to fulfill the duty of subclassing tf.train.Checkpoint.\n\n    Args:\n      save_path: The full name of the checkpoint file to be restored.\n      options: CheckpointOption instance.\n\n    Returns:\n      A load status object, which can be used to make assertions about the\n      status of a checkpoint restoration. See tf.train.Checkpoint.restore()\n      for more details.\n    '
        return self.restore(save_path, options)

    def restore(self, save_path, options=None):
        if False:
            i = 10
            return i + 15
        'Restore the checkpointed variables.\n\n    Args:\n      save_path: The full name of the checkpoint file to be restored.\n      options: CheckpointOption instance.\n\n    Returns:\n      A load status object, which can be used to make assertions about the\n      status of a checkpoint restoration. See tf.train.Checkpoint.restore()\n      for more details.\n    '
        self._checkpoint_options = copy.copy(options) if options else self._checkpoint_options
        if self._checkpoint_options:
            self._checkpoint_options.experimental_enable_async_checkpoint = False
        self._queue.join()
        status = self.checkpointer().restore(save_path, self._checkpoint_options)
        return status

    def sync(self):
        if False:
            print('Hello World!')
        'Sync on any ongoing save or restore events.'
        self._queue.join()
        logging.info('Sync on ongoing save/restore.')
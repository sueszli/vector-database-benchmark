"""Checkpoint Manager and other utilities for managing checkpoints."""
import collections
import copy
import os.path
import re
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

def _evaluate(tensor):
    if False:
        return 10
    'Returns the numpy value of a tensor.'
    if context.executing_eagerly():
        return tensor.numpy()
    return ops.get_default_session().run(tensor)

def _GetCheckpointFilename(save_dir, latest_filename):
    if False:
        while True:
            i = 10
    "Returns a filename for storing the CheckpointState.\n\n  Args:\n    save_dir: The directory for saving and restoring checkpoints.\n    latest_filename: Name of the file in 'save_dir' that is used\n      to store the CheckpointState.\n\n  Returns:\n    The path of the file that contains the CheckpointState proto.\n  "
    if latest_filename is None:
        latest_filename = 'checkpoint'
    return os.path.join(save_dir, latest_filename)

@tf_export(v1=['train.generate_checkpoint_state_proto'])
def generate_checkpoint_state_proto(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, all_model_checkpoint_timestamps=None, last_preserved_timestamp=None):
    if False:
        print('Hello World!')
    'Generates a checkpoint state proto.\n\n  Args:\n    save_dir: Directory where the model was saved.\n    model_checkpoint_path: The checkpoint file.\n    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted\n      checkpoints, sorted from oldest to newest.  If this is a non-empty list,\n      the last element must be equal to model_checkpoint_path.  These paths\n      are also saved in the CheckpointState proto.\n    all_model_checkpoint_timestamps: A list of floats, indicating the number of\n      seconds since the Epoch when each checkpoint was generated.\n    last_preserved_timestamp: A float, indicating the number of seconds since\n      the Epoch when the last preserved checkpoint was written, e.g. due to a\n      `keep_checkpoint_every_n_hours` parameter (see\n      `tf.train.CheckpointManager` for an implementation).\n  Returns:\n    CheckpointState proto with model_checkpoint_path and\n    all_model_checkpoint_paths updated to either absolute paths or\n    relative paths to the current save_dir.\n\n  Raises:\n    ValueError: If `all_model_checkpoint_timestamps` was provided but its length\n      does not match `all_model_checkpoint_paths`.\n  '
    if all_model_checkpoint_paths is None:
        all_model_checkpoint_paths = []
    if not all_model_checkpoint_paths or all_model_checkpoint_paths[-1] != model_checkpoint_path:
        logging.info('%s is not in all_model_checkpoint_paths. Manually adding it.', model_checkpoint_path)
        all_model_checkpoint_paths.append(model_checkpoint_path)
    if all_model_checkpoint_timestamps and len(all_model_checkpoint_timestamps) != len(all_model_checkpoint_paths):
        raise ValueError('Checkpoint timestamps, if provided, must match checkpoint paths (got paths %s and timestamps %s)' % (all_model_checkpoint_paths, all_model_checkpoint_timestamps))
    if not os.path.isabs(save_dir):
        if not os.path.isabs(model_checkpoint_path):
            model_checkpoint_path = os.path.relpath(model_checkpoint_path, save_dir)
        for (i, p) in enumerate(all_model_checkpoint_paths):
            if not os.path.isabs(p):
                all_model_checkpoint_paths[i] = os.path.relpath(p, save_dir)
    coord_checkpoint_proto = CheckpointState(model_checkpoint_path=model_checkpoint_path, all_model_checkpoint_paths=all_model_checkpoint_paths, all_model_checkpoint_timestamps=all_model_checkpoint_timestamps, last_preserved_timestamp=last_preserved_timestamp)
    return coord_checkpoint_proto

@deprecation.deprecated(date=None, instructions='Use `tf.train.CheckpointManager` to manage checkpoints rather than manually editing the Checkpoint proto.')
@tf_export(v1=['train.update_checkpoint_state'])
def update_checkpoint_state(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, latest_filename=None, all_model_checkpoint_timestamps=None, last_preserved_timestamp=None):
    if False:
        for i in range(10):
            print('nop')
    "Updates the content of the 'checkpoint' file.\n\n  This updates the checkpoint file containing a CheckpointState\n  proto.\n\n  Args:\n    save_dir: Directory where the model was saved.\n    model_checkpoint_path: The checkpoint file.\n    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted\n      checkpoints, sorted from oldest to newest.  If this is a non-empty list,\n      the last element must be equal to model_checkpoint_path.  These paths\n      are also saved in the CheckpointState proto.\n    latest_filename: Optional name of the checkpoint file.  Default to\n      'checkpoint'.\n    all_model_checkpoint_timestamps: Optional list of timestamps (floats,\n      seconds since the Epoch) indicating when the checkpoints in\n      `all_model_checkpoint_paths` were created.\n    last_preserved_timestamp: A float, indicating the number of seconds since\n      the Epoch when the last preserved checkpoint was written, e.g. due to a\n      `keep_checkpoint_every_n_hours` parameter (see\n      `tf.train.CheckpointManager` for an implementation).\n  Raises:\n    RuntimeError: If any of the model checkpoint paths conflict with the file\n      containing CheckpointSate.\n  "
    update_checkpoint_state_internal(save_dir=save_dir, model_checkpoint_path=model_checkpoint_path, all_model_checkpoint_paths=all_model_checkpoint_paths, latest_filename=latest_filename, save_relative_paths=False, all_model_checkpoint_timestamps=all_model_checkpoint_timestamps, last_preserved_timestamp=last_preserved_timestamp)

@tf_export('__internal__.train.update_checkpoint_state', v1=[])
def update_checkpoint_state_internal(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, latest_filename=None, save_relative_paths=False, all_model_checkpoint_timestamps=None, last_preserved_timestamp=None):
    if False:
        return 10
    "Updates the content of the 'checkpoint' file.\n\n  This updates the checkpoint file containing a CheckpointState\n  proto.\n\n  Args:\n    save_dir: Directory where the model was saved.\n    model_checkpoint_path: The checkpoint file.\n    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted\n      checkpoints, sorted from oldest to newest.  If this is a non-empty list,\n      the last element must be equal to model_checkpoint_path.  These paths\n      are also saved in the CheckpointState proto.\n    latest_filename: Optional name of the checkpoint file.  Default to\n      'checkpoint'.\n    save_relative_paths: If `True`, will write relative paths to the checkpoint\n      state file.\n    all_model_checkpoint_timestamps: Optional list of timestamps (floats,\n      seconds since the Epoch) indicating when the checkpoints in\n      `all_model_checkpoint_paths` were created.\n    last_preserved_timestamp: A float, indicating the number of seconds since\n      the Epoch when the last preserved checkpoint was written, e.g. due to a\n      `keep_checkpoint_every_n_hours` parameter (see\n      `tf.train.CheckpointManager` for an implementation).\n\n  Raises:\n    RuntimeError: If any of the model checkpoint paths conflict with the file\n      containing CheckpointSate.\n  "
    coord_checkpoint_filename = _GetCheckpointFilename(save_dir, latest_filename)
    if save_relative_paths:
        if os.path.isabs(model_checkpoint_path):
            rel_model_checkpoint_path = os.path.relpath(model_checkpoint_path, save_dir)
        else:
            rel_model_checkpoint_path = model_checkpoint_path
        rel_all_model_checkpoint_paths = []
        for p in all_model_checkpoint_paths:
            if os.path.isabs(p):
                rel_all_model_checkpoint_paths.append(os.path.relpath(p, save_dir))
            else:
                rel_all_model_checkpoint_paths.append(p)
        ckpt = generate_checkpoint_state_proto(save_dir, rel_model_checkpoint_path, all_model_checkpoint_paths=rel_all_model_checkpoint_paths, all_model_checkpoint_timestamps=all_model_checkpoint_timestamps, last_preserved_timestamp=last_preserved_timestamp)
    else:
        ckpt = generate_checkpoint_state_proto(save_dir, model_checkpoint_path, all_model_checkpoint_paths=all_model_checkpoint_paths, all_model_checkpoint_timestamps=all_model_checkpoint_timestamps, last_preserved_timestamp=last_preserved_timestamp)
    if coord_checkpoint_filename == ckpt.model_checkpoint_path:
        raise RuntimeError("Save path '%s' conflicts with path used for checkpoint state.  Please use a different save path." % model_checkpoint_path)
    file_io.atomic_write_string_to_file(coord_checkpoint_filename, text_format.MessageToString(ckpt))

@tf_export('train.get_checkpoint_state')
def get_checkpoint_state(checkpoint_dir, latest_filename=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns CheckpointState proto from the "checkpoint" file.\n\n  If the "checkpoint" file contains a valid CheckpointState\n  proto, returns it.\n\n  Args:\n    checkpoint_dir: The directory of checkpoints.\n    latest_filename: Optional name of the checkpoint file.  Default to\n      \'checkpoint\'.\n\n  Returns:\n    A CheckpointState if the state was available, None\n    otherwise.\n\n  Raises:\n    ValueError: if the checkpoint read doesn\'t have model_checkpoint_path set.\n  '
    if isinstance(checkpoint_dir, os.PathLike):
        checkpoint_dir = os.fspath(checkpoint_dir)
    ckpt = None
    coord_checkpoint_filename = _GetCheckpointFilename(checkpoint_dir, latest_filename)
    f = None
    try:
        if file_io.file_exists(coord_checkpoint_filename):
            file_content = file_io.read_file_to_string(coord_checkpoint_filename)
            ckpt = CheckpointState()
            text_format.Merge(file_content, ckpt)
            if not ckpt.model_checkpoint_path:
                raise ValueError('Invalid checkpoint state loaded from ' + checkpoint_dir)
            if not os.path.isabs(ckpt.model_checkpoint_path):
                ckpt.model_checkpoint_path = os.path.join(checkpoint_dir, ckpt.model_checkpoint_path)
            for (i, p) in enumerate(ckpt.all_model_checkpoint_paths):
                if not os.path.isabs(p):
                    ckpt.all_model_checkpoint_paths[i] = os.path.join(checkpoint_dir, p)
    except errors.OpError as e:
        logging.warning('%s: %s', type(e).__name__, e)
        logging.warning('%s: Checkpoint ignored', coord_checkpoint_filename)
        return None
    except text_format.ParseError as e:
        logging.warning('%s: %s', type(e).__name__, e)
        logging.warning('%s: Checkpoint ignored', coord_checkpoint_filename)
        return None
    finally:
        if f:
            f.close()
    return ckpt

def _prefix_to_checkpoint_path(prefix, format_version):
    if False:
        while True:
            i = 10
    'Returns the pathname of a checkpoint file, given the checkpoint prefix.\n\n  For V1 checkpoint, simply returns the prefix itself (the data file).  For V2,\n  returns the pathname to the index file.\n\n  Args:\n    prefix: a string, the prefix of a checkpoint.\n    format_version: the checkpoint format version that corresponds to the\n      prefix.\n  Returns:\n    The pathname of a checkpoint file, taking into account the checkpoint\n      format version.\n  '
    if format_version == saver_pb2.SaverDef.V2:
        return prefix + '.index'
    return prefix

@tf_export('train.latest_checkpoint')
def latest_checkpoint(checkpoint_dir, latest_filename=None):
    if False:
        for i in range(10):
            print('nop')
    'Finds the filename of latest saved checkpoint file.\n\n  Gets the checkpoint state given the provided checkpoint_dir and looks for a\n  corresponding TensorFlow 2 (preferred) or TensorFlow 1.x checkpoint path.\n  The latest_filename argument is only applicable if you are saving checkpoint\n  using `v1.train.Saver.save`\n\n\n  See the [Training Checkpoints\n  Guide](https://www.tensorflow.org/guide/checkpoint) for more details and\n  examples.`\n\n  Args:\n    checkpoint_dir: Directory where the variables were saved.\n    latest_filename: Optional name for the protocol buffer file that\n      contains the list of most recent checkpoint filenames.\n      See the corresponding argument to `v1.train.Saver.save`.\n\n  Returns:\n    The full path to the latest checkpoint or `None` if no checkpoint was found.\n  '
    ckpt = get_checkpoint_state(checkpoint_dir, latest_filename)
    if ckpt and ckpt.model_checkpoint_path:
        v2_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path, saver_pb2.SaverDef.V2)
        v1_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path, saver_pb2.SaverDef.V1)
        if file_io.get_matching_files(v2_path) or file_io.get_matching_files(v1_path):
            return ckpt.model_checkpoint_path
        else:
            logging.error("Couldn't match files for checkpoint %s", ckpt.model_checkpoint_path)
    return None

def checkpoint_exists_internal(checkpoint_prefix):
    if False:
        return 10
    'Checks whether a V1 or V2 checkpoint exists with the specified prefix.\n\n  This is an internal function to check if a checkpoint exists,\n  since it takes into account the naming difference between V1 and V2 formats.\n\n  Args:\n    checkpoint_prefix: the prefix of a V1 or V2 checkpoint, with V2 taking\n      priority.  Typically the result of `Saver.save()` or that of\n      `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or\n      V1/V2.\n  Returns:\n    A bool, true if a checkpoint referred to by `checkpoint_prefix` exists.\n  '
    pathname = _prefix_to_checkpoint_path(checkpoint_prefix, saver_pb2.SaverDef.V2)
    if file_io.get_matching_files(pathname):
        return True
    elif file_io.get_matching_files(checkpoint_prefix):
        return True
    else:
        return False

@deprecation.deprecated(date=None, instructions='Use standard file APIs to check for files with this prefix.')
@tf_export(v1=['train.checkpoint_exists'])
def checkpoint_exists(checkpoint_prefix):
    if False:
        return 10
    'Checks whether a V1 or V2 checkpoint exists with the specified prefix.\n\n  This is the recommended way to check if a checkpoint exists, since it takes\n  into account the naming difference between V1 and V2 formats.\n\n  Args:\n    checkpoint_prefix: the prefix of a V1 or V2 checkpoint, with V2 taking\n      priority.  Typically the result of `Saver.save()` or that of\n      `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or\n      V1/V2.\n\n  Returns:\n    A bool, true if a checkpoint referred to by `checkpoint_prefix` exists.\n  '
    return checkpoint_exists_internal(checkpoint_prefix)

@deprecation.deprecated(date=None, instructions='Use standard file utilities to get mtimes.')
@tf_export(v1=['train.get_checkpoint_mtimes'])
def get_checkpoint_mtimes(checkpoint_prefixes):
    if False:
        print('Hello World!')
    'Returns the mtimes (modification timestamps) of the checkpoints.\n\n  Globs for the checkpoints pointed to by `checkpoint_prefixes`.  If the files\n  exist, collect their mtime.  Both V2 and V1 checkpoints are considered, in\n  that priority.\n\n  This is the recommended way to get the mtimes, since it takes into account\n  the naming difference between V1 and V2 formats.\n\n  Note: If not all checkpoints exist, the length of the returned mtimes list\n  will be smaller than the length of `checkpoint_prefixes` list, so mapping\n  checkpoints to corresponding mtimes will not be possible.\n\n  Args:\n    checkpoint_prefixes: a list of checkpoint paths, typically the results of\n      `Saver.save()` or those of `tf.train.latest_checkpoint()`, regardless of\n      sharded/non-sharded or V1/V2.\n  Returns:\n    A list of mtimes (in microseconds) of the found checkpoints.\n  '
    mtimes = []

    def match_maybe_append(pathname):
        if False:
            print('Hello World!')
        fnames = file_io.get_matching_files(pathname)
        if fnames:
            mtimes.append(file_io.stat(fnames[0]).mtime_nsec / 1000000000.0)
            return True
        return False
    for checkpoint_prefix in checkpoint_prefixes:
        pathname = _prefix_to_checkpoint_path(checkpoint_prefix, saver_pb2.SaverDef.V2)
        if match_maybe_append(pathname):
            continue
        match_maybe_append(checkpoint_prefix)
    return mtimes

@deprecation.deprecated(date=None, instructions='Use standard file APIs to delete files with this prefix.')
@tf_export(v1=['train.remove_checkpoint'])
def remove_checkpoint(checkpoint_prefix, checkpoint_format_version=saver_pb2.SaverDef.V2, meta_graph_suffix='meta'):
    if False:
        i = 10
        return i + 15
    "Removes a checkpoint given by `checkpoint_prefix`.\n\n  Args:\n    checkpoint_prefix: The prefix of a V1 or V2 checkpoint. Typically the result\n      of `Saver.save()` or that of `tf.train.latest_checkpoint()`, regardless of\n      sharded/non-sharded or V1/V2.\n    checkpoint_format_version: `SaverDef.CheckpointFormatVersion`, defaults to\n      `SaverDef.V2`.\n    meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.\n  "
    _delete_file_if_exists(meta_graph_filename(checkpoint_prefix, meta_graph_suffix))
    if checkpoint_format_version == saver_pb2.SaverDef.V2:
        _delete_file_if_exists(checkpoint_prefix + '.index')
        _delete_file_if_exists(checkpoint_prefix + '.data-?????-of-?????')
    else:
        _delete_file_if_exists(checkpoint_prefix)

def _delete_file_if_exists(filespec):
    if False:
        print('Hello World!')
    'Deletes files matching `filespec`.'
    for pathname in file_io.get_matching_files(filespec):
        try:
            file_io.delete_file(pathname)
        except errors.NotFoundError:
            logging.warning("Hit NotFoundError when deleting '%s', possibly because another process/thread is also deleting/moving the same file", pathname)

def meta_graph_filename(checkpoint_filename, meta_graph_suffix='meta'):
    if False:
        print('Hello World!')
    "Returns the meta graph filename.\n\n  Args:\n    checkpoint_filename: Name of the checkpoint file.\n    meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.\n\n  Returns:\n    MetaGraph file name.\n  "
    basename = re.sub('-[\\d\\?]+-of-\\d+$', '', checkpoint_filename)
    suffixed_filename = '.'.join([basename, meta_graph_suffix])
    return suffixed_filename

@tf_export('train.CheckpointManager')
class CheckpointManager(object):
    """Manages multiple checkpoints by keeping some and deleting unneeded ones.

  Example usage:

  ```python
  import tensorflow as tf
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(
      checkpoint, directory="/tmp/model", max_to_keep=5)
  status = checkpoint.restore(manager.latest_checkpoint)
  while True:
    # train
    manager.save()
  ```

  `CheckpointManager` preserves its own state across instantiations (see the
  `__init__` documentation for details). Only one should be active in a
  particular directory at a time.
  """

    def __init__(self, checkpoint, directory, max_to_keep, keep_checkpoint_every_n_hours=None, checkpoint_name='ckpt', step_counter=None, checkpoint_interval=None, init_fn=None):
        if False:
            while True:
                i = 10
        'Configure a `CheckpointManager` for use in `directory`.\n\n    If a `CheckpointManager` was previously used in `directory`, its\n    state will be restored. This includes the list of managed checkpoints and\n    the timestamp bookkeeping necessary to support\n    `keep_checkpoint_every_n_hours`. The behavior of the new `CheckpointManager`\n    will be the same as the previous `CheckpointManager`, including cleaning up\n    existing checkpoints if appropriate.\n\n    Checkpoints are only considered for deletion just after a new checkpoint has\n    been added. At that point, `max_to_keep` checkpoints will remain in an\n    "active set". Once a checkpoint is preserved by\n    `keep_checkpoint_every_n_hours` it will not be deleted by this\n    `CheckpointManager` or any future `CheckpointManager` instantiated in\n    `directory` (regardless of the new setting of\n    `keep_checkpoint_every_n_hours`). The `max_to_keep` checkpoints in the\n    active set may be deleted by this `CheckpointManager` or a future\n    `CheckpointManager` instantiated in `directory` (subject to its\n    `max_to_keep` and `keep_checkpoint_every_n_hours` settings).\n\n    `CheckpointManager` can be also used for initializing the model if\n    there is no checkpoints for restoring in `directory`. An example usage is:\n\n    >>> import tempfile\n\n    >>> tmp_dir = tempfile.mkdtemp()\n    >>> checkpoint = tf.train.Checkpoint()\n    >>> init_path = checkpoint.save(os.path.join(tmp_dir, \'init\'))\n\n    >>> def init_fn():\n    ...   # Partially restore the checkpoint from `init_path`.\n    ...   checkpoint.restore(init_path)\n\n    >>> manager = tf.train.CheckpointManager(\n    ...     checkpoint,\n    ...     directory=os.path.join(tmp_dir, \'ckpt\'),\n    ...     max_to_keep=None,\n    ...     init_fn=init_fn)\n    >>> # `restore_or_initialize` will call `init_fn` if there is no existing\n    >>> # checkpoint in `directory`.\n    >>> manager.restore_or_initialize()\n\n    Args:\n      checkpoint: The `tf.train.Checkpoint` instance to save and manage\n        checkpoints for.\n      directory: The path to a directory in which to write checkpoints. A\n        special file named "checkpoint" is also written to this directory (in a\n        human-readable text format) which contains the state of the\n        `CheckpointManager`.\n      max_to_keep: An integer, the number of checkpoints to keep. Unless\n        preserved by `keep_checkpoint_every_n_hours`, checkpoints will be\n        deleted from the active set, oldest first, until only `max_to_keep`\n        checkpoints remain. If `None`, no checkpoints are deleted and everything\n        stays in the active set. Note that `max_to_keep=None` will keep all\n        checkpoint paths in memory and in the checkpoint state protocol buffer\n        on disk.\n      keep_checkpoint_every_n_hours: Upon removal from the active set, a\n        checkpoint will be preserved if it has been at least\n        `keep_checkpoint_every_n_hours` since the last preserved checkpoint. The\n        default setting of `None` does not preserve any checkpoints in this way.\n      checkpoint_name: Custom name for the checkpoint file.\n      step_counter: A `tf.Variable` instance for checking the current step\n        counter value, in case users want to save checkpoints every N steps.\n      checkpoint_interval: An integer, indicates the minimum step interval\n        between two checkpoints.\n      init_fn: Callable. A function to do customized intialization if no\n        checkpoints are in the directory.\n\n    Raises:\n      ValueError: If `max_to_keep` is not a positive integer.\n    '
        self._checkpoint = checkpoint
        self._save_counter_assign = None
        if max_to_keep is not None and max_to_keep <= 0:
            raise ValueError('Expected a positive integer or `None` for `max_to_keep`, got %d.' % (max_to_keep,))
        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        if isinstance(directory, os.PathLike):
            directory = os.fspath(directory)
        self._directory = directory
        self._checkpoint_prefix = os.path.join(directory, checkpoint_name)
        self._init_fn = init_fn
        if checkpoint_interval is not None:
            if step_counter is None:
                raise ValueError('`step_counter` should be passed if `checkpoint_interval` is not None.')
            self._last_checkpoint_step = None
            self._step_counter = step_counter
        self._checkpoint_interval = checkpoint_interval
        recovered_state = get_checkpoint_state(directory)
        current_clock = time.time()
        self._maybe_delete = collections.OrderedDict()
        if recovered_state is None:
            self._latest_checkpoint = None
            self._last_preserved_timestamp = current_clock - 1.0
        else:
            self._latest_checkpoint = recovered_state.model_checkpoint_path
            self._last_preserved_timestamp = recovered_state.last_preserved_timestamp
            if current_clock < self._last_preserved_timestamp:
                logging.warning('time.time() returned a value %f seconds behind the last preserved checkpoint timestamp.' % (self._last_preserved_timestamp - current_clock,))
                self._last_preserved_timestamp = current_clock
            all_timestamps = recovered_state.all_model_checkpoint_timestamps
            all_paths = recovered_state.all_model_checkpoint_paths
            del recovered_state
            if not all_timestamps:
                all_timestamps = [self._last_preserved_timestamp] * len(all_paths)
            for (filename, timestamp) in zip(all_paths, all_timestamps):
                timestamp = min(timestamp, current_clock)
                if timestamp > self._last_preserved_timestamp:
                    self._maybe_delete[filename] = timestamp

    @property
    def directory(self):
        if False:
            for i in range(10):
                print('nop')
        return self._directory

    @property
    def checkpoint_interval(self):
        if False:
            i = 10
            return i + 15
        return self._checkpoint_interval

    @property
    def latest_checkpoint(self):
        if False:
            print('Hello World!')
        'The prefix of the most recent checkpoint in `directory`.\n\n    Equivalent to `tf.train.latest_checkpoint(directory)` where `directory` is\n    the constructor argument to `CheckpointManager`.\n\n    Suitable for passing to `tf.train.Checkpoint.restore` to resume training.\n\n    Returns:\n      The checkpoint prefix. If there are no checkpoints, returns `None`.\n    '
        return self._latest_checkpoint

    @property
    def checkpoints(self):
        if False:
            print('Hello World!')
        'A list of managed checkpoints.\n\n    Note that checkpoints saved due to `keep_checkpoint_every_n_hours` will not\n    show up in this list (to avoid ever-growing filename lists).\n\n    Returns:\n      A list of filenames, sorted from oldest to newest.\n    '
        return list(self._maybe_delete.keys())

    def _sweep(self):
        if False:
            for i in range(10):
                print('nop')
        'Deletes or preserves managed checkpoints.'
        if not self._max_to_keep:
            return
        while len(self._maybe_delete) > self._max_to_keep:
            (filename, timestamp) = self._maybe_delete.popitem(last=False)
            if self._keep_checkpoint_every_n_hours and timestamp - self._keep_checkpoint_every_n_hours * 3600.0 >= self._last_preserved_timestamp:
                self._last_preserved_timestamp = timestamp
                continue
            _delete_file_if_exists(filename + '.index')
            _delete_file_if_exists(filename + '.data-?????-of-?????')

    def _record_state(self):
        if False:
            print('Hello World!')
        "Saves the `CheckpointManager`'s state in `directory`."
        (filenames, timestamps) = zip(*self._maybe_delete.items())
        update_checkpoint_state_internal(self._directory, model_checkpoint_path=self.latest_checkpoint, all_model_checkpoint_paths=filenames, all_model_checkpoint_timestamps=timestamps, last_preserved_timestamp=self._last_preserved_timestamp, save_relative_paths=True)

    @property
    def _prefix(self):
        if False:
            i = 10
            return i + 15
        'A common prefix for all checkpoints saved with this manager.\n\n    For example, if `directory` (a constructor argument) were `"/tmp/tf-model"`,\n    `prefix` would be `"/tmp/tf-model/ckpt"` and checkpoints would generally be\n    numbered `"/tmp/tf-model/ckpt-1"`, `"/tmp/tf-model/ckpt-2"`, and so on. Each\n    checkpoint has several associated files\n    (e.g. `"/tmp/tf-model/ckpt-2.index"`).\n\n    Returns:\n      A string prefix.\n    '
        return self._checkpoint_prefix

    @property
    def checkpoint(self):
        if False:
            while True:
                i = 10
        'Returns the `tf.train.Checkpoint` object.'
        return self._checkpoint

    def save(self, checkpoint_number=None, check_interval=True, options=None):
        if False:
            i = 10
            return i + 15
        "Creates a new checkpoint and manages it.\n\n    Args:\n      checkpoint_number: An optional integer, or an integer-dtype `Variable` or\n        `Tensor`, used to number the checkpoint. If `None` (default),\n        checkpoints are numbered using `checkpoint.save_counter`. Even if\n        `checkpoint_number` is provided, `save_counter` is still incremented. A\n        user-provided `checkpoint_number` is not incremented even if it is a\n        `Variable`.\n      check_interval: An optional boolean. The argument is only effective when\n        `checkpoint_interval` is passed into the manager. If `True`, the manager\n        will only save the checkpoint if the interval between checkpoints is\n        larger than `checkpoint_interval`. Otherwise it will always save the\n        checkpoint unless a checkpoint has already been saved for the current\n        step.\n      options: Optional `tf.train.CheckpointOptions` object. This argument only\n        works with TF2 checkpoint objects. For example, options =\n        tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')\n\n    Returns:\n      The path to the new checkpoint. It is also recorded in the `checkpoints`\n      and `latest_checkpoint` properties. `None` if no checkpoint is saved.\n    "
        if self._checkpoint_interval is not None:
            current_step = _evaluate(self._step_counter)
            if self._last_checkpoint_step is not None:
                if current_step == self._last_checkpoint_step:
                    return None
                if check_interval and current_step < self._last_checkpoint_step + self._checkpoint_interval:
                    return None
            self._last_checkpoint_step = current_step
        if context.executing_eagerly():
            save_counter = self._checkpoint.save_counter
            save_counter.assign_add(1)
            session = None
        else:
            session = ops.get_default_session()

            def _initializing_creator(next_creator, **kwargs):
                if False:
                    i = 10
                    return i + 15
                'Initialize the save counter if it has been newly created.'
                v = next_creator(**kwargs)
                session.run(v.initializer)
                return v
            with variable_scope.variable_creator_scope(_initializing_creator):
                save_counter = self._checkpoint.save_counter
            if self._save_counter_assign is None:
                self._save_counter_assign = save_counter.assign_add(1, read_value=False)
            session.run(self._save_counter_assign)
        if checkpoint_number is None:
            checkpoint_number = save_counter
        if not isinstance(checkpoint_number, compat.integral_types):
            checkpoint_number = training_util.global_step(sess=session, global_step_tensor=checkpoint_number)
        prefix = '%s-%d' % (self._prefix, checkpoint_number)

        def _record_and_sweep_state(save_path):
            if False:
                print('Hello World!')
            timestamp = time.time()
            if save_path in self._maybe_delete:
                del self._maybe_delete[save_path]
            self._maybe_delete[save_path] = timestamp
            self._latest_checkpoint = save_path
            self._record_state()
            self._sweep()
            self._record_state()
        if options is None:
            options = checkpoint_options.CheckpointOptions(experimental_write_callbacks=[_record_and_sweep_state])
        else:
            options = copy.copy(options)
            if options.experimental_write_callbacks is None:
                options.experimental_write_callbacks = [_record_and_sweep_state]
            else:
                options.experimental_write_callbacks.append(_record_and_sweep_state)
        save_path = self._checkpoint._write(prefix, options=options)
        return save_path

    def restore_or_initialize(self):
        if False:
            print('Hello World!')
        "Restore items in `checkpoint` from the latest checkpoint file.\n\n    This method will first try to restore from the most recent checkpoint in\n    `directory`. If no checkpoints exist in `directory`, and `init_fn` is\n    specified, this method will call `init_fn` to do customized\n    initialization. This can be used to support initialization from pretrained\n    models.\n\n    Note that unlike `tf.train.Checkpoint.restore()`, this method doesn't return\n    a load status object that users can run assertions on\n    (e.g. assert_consumed()). Thus to run assertions, users should directly use\n    `tf.train.Checkpoint.restore()` method.\n\n    Returns:\n      The restored checkpoint path if the lastest checkpoint is found and\n      restored. Otherwise None.\n    "
        if self._latest_checkpoint is not None:
            self._checkpoint.restore(self._latest_checkpoint)
            if self._checkpoint_interval is not None:
                self._last_checkpoint_step = _evaluate(self._step_counter)
            return self._latest_checkpoint
        if self._init_fn is not None:
            self._init_fn()
            logging.info('Customized initialization is done through the passed `init_fn`.')
        return None

    def sync(self):
        if False:
            while True:
                i = 10
        'Wait for any outstanding save or restore operations.'
        if self._checkpoint:
            self._checkpoint.sync()
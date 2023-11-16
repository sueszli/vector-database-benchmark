"""Save and restore variables.

Symbols in this file are deprecated. See replacements in
tensorflow/python/training/trackable and tensorflow/python/training/saving.
"""
import collections
import glob
import os.path
import threading
import time
import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
get_checkpoint_state = checkpoint_management.get_checkpoint_state
update_checkpoint_state = checkpoint_management.update_checkpoint_state
generate_checkpoint_state_proto = checkpoint_management.generate_checkpoint_state_proto
latest_checkpoint = checkpoint_management.latest_checkpoint
checkpoint_exists = checkpoint_management.checkpoint_exists
get_checkpoint_mtimes = checkpoint_management.get_checkpoint_mtimes
remove_checkpoint = checkpoint_management.remove_checkpoint
_END_TIME_OF_LAST_WRITE = None
_END_TIME_OF_LAST_WRITE_LOCK = threading.Lock()
_SAVER_LABEL = 'saver_v1'

def _get_duration_microseconds(start_time_seconds, end_time_seconds):
    if False:
        i = 10
        return i + 15
    if end_time_seconds < start_time_seconds:
        return 0
    return round((end_time_seconds - start_time_seconds) * 1000000)

def _get_checkpoint_size(prefix):
    if False:
        return 10
    'Calculates filesize of checkpoint based on prefix.'
    size = 0
    files = glob.glob('{}*'.format(prefix))
    for file in files:
        size += metrics.CalculateFileSize(file)
    return size

class BaseSaverBuilder:
    """Base class for Savers.

  Can be extended to create different Ops.
  """
    SaveSpec = saveable_object.SaveSpec
    SaveableObject = saveable_object.SaveableObject
    VariableSaveable = saveable_object_util.ReferenceVariableSaveable
    ResourceVariableSaveable = saveable_object_util.ResourceVariableSaveable

    def __init__(self, write_version=saver_pb2.SaverDef.V2):
        if False:
            for i in range(10):
                print('nop')
        self._write_version = write_version

    def save_op(self, filename_tensor, saveables):
        if False:
            print('Hello World!')
        'Create an Op to save \'saveables\'.\n\n    This is intended to be overridden by subclasses that want to generate\n    different Ops.\n\n    Args:\n      filename_tensor: String Tensor.\n      saveables: A list of BaseSaverBuilder.SaveableObject objects.\n\n    Returns:\n      An Operation that save the variables.\n\n    Raises:\n      RuntimeError: (implementation detail) if "self._write_version" is an\n        unexpected value.\n    '
        tensor_names = []
        tensors = []
        tensor_slices = []
        for saveable in saveables:
            for spec in saveable.specs:
                tensor_names.append(spec.name)
                tensors.append(spec.tensor)
                tensor_slices.append(spec.slice_spec)
        if self._write_version == saver_pb2.SaverDef.V1:
            return io_ops._save(filename=filename_tensor, tensor_names=tensor_names, tensors=tensors, tensor_slices=tensor_slices)
        elif self._write_version == saver_pb2.SaverDef.V2:
            return io_ops.save_v2(filename_tensor, tensor_names, tensor_slices, tensors)
        else:
            raise RuntimeError('Unexpected write_version: ' + self._write_version)

    def bulk_restore(self, filename_tensor, saveables, preferred_shard, restore_sequentially):
        if False:
            for i in range(10):
                print('nop')
        "Restore all tensors contained in saveables.\n\n    By default, this issues separate calls to `restore_op` for each saveable.\n    Subclasses may override to load multiple saveables in a single call.\n\n    Args:\n      filename_tensor: String Tensor.\n      saveables: List of BaseSaverBuilder.SaveableObject objects.\n      preferred_shard: Int.  Shard to open first when loading a sharded file.\n      restore_sequentially: Unused.  Bool.  If true, each restore is sequential.\n\n    Returns:\n      A list of Tensors resulting from reading 'saveable' from\n        'filename'.\n\n    "
        del restore_sequentially
        all_tensors = []
        for saveable in saveables:
            if saveable.device:
                device = saveable_object_util.set_cpu0(saveable.device)
            else:
                device = None
            with ops.device(device):
                all_tensors.extend(self.restore_op(filename_tensor, saveable, preferred_shard))
        return all_tensors

    def restore_op(self, filename_tensor, saveable, preferred_shard):
        if False:
            while True:
                i = 10
        "Create ops to restore 'saveable'.\n\n    This is intended to be overridden by subclasses that want to generate\n    different Ops.\n\n    Args:\n      filename_tensor: String Tensor.\n      saveable: A BaseSaverBuilder.SaveableObject object.\n      preferred_shard: Int.  Shard to open first when loading a sharded file.\n\n    Returns:\n      A list of Tensors resulting from reading 'saveable' from\n        'filename'.\n    "
        tensors = []
        for spec in saveable.specs:
            tensors.append(io_ops.restore_v2(filename_tensor, [spec.name], [spec.slice_spec], [spec.dtype])[0])
        return tensors

    def sharded_filename(self, filename_tensor, shard, num_shards):
        if False:
            i = 10
            return i + 15
        'Append sharding information to a filename.\n\n    Args:\n      filename_tensor: A string tensor.\n      shard: Integer.  The shard for the filename.\n      num_shards: An int Tensor for the number of shards.\n\n    Returns:\n      A string tensor.\n    '
        return gen_io_ops.sharded_filename(filename_tensor, shard, num_shards)

    def _AddSaveOps(self, filename_tensor, saveables):
        if False:
            for i in range(10):
                print('nop')
        'Add ops to save variables that are on the same shard.\n\n    Args:\n      filename_tensor: String Tensor.\n      saveables: A list of SaveableObject objects.\n\n    Returns:\n      A tensor with the filename used to save.\n    '
        save = self.save_op(filename_tensor, saveables)
        return control_flow_ops.with_dependencies([save], filename_tensor)

    def _AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device):
        if False:
            return 10
        'Add ops to save the params per shard, for the V2 format.\n\n    Note that the sharded save procedure for the V2 format is different from\n    V1: there is a special "merge" step that merges the small metadata produced\n    from each device.\n\n    Args:\n      checkpoint_prefix: scalar String Tensor.  Interpreted *NOT AS A FILENAME*,\n        but as a prefix of a V2 checkpoint;\n      per_device: A list of (device, BaseSaverBuilder.VarToSave) pairs, as\n        returned by _GroupByDevices().\n\n    Returns:\n      An op to save the variables, which, when evaluated, returns the prefix\n        "<user-fed prefix>" only and does not include the sharded spec suffix.\n    '
        with ops.device('CPU'):
            _SHARDED_SUFFIX = array_ops.where(string_ops.regex_full_match(checkpoint_prefix, '^s3://.*'), constant_op.constant('.part'), constant_op.constant(os.path.normpath('_temp/part')))
            tmp_checkpoint_prefix = string_ops.string_join([checkpoint_prefix, _SHARDED_SUFFIX])
        num_shards = len(per_device)
        sharded_saves = []
        sharded_prefixes = []
        num_shards_tensor = constant_op.constant(num_shards, name='num_shards')
        last_device = None
        for (shard, (device, saveables)) in enumerate(per_device):
            last_device = device
            with ops.device(saveable_object_util.set_cpu0(device)):
                sharded_filename = self.sharded_filename(tmp_checkpoint_prefix, shard, num_shards_tensor)
                sharded_prefixes.append(sharded_filename)
                sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
        with ops.control_dependencies([x.op for x in sharded_saves]):
            with ops.device(saveable_object_util.set_cpu0(last_device)):
                merge_step = gen_io_ops.merge_v2_checkpoints(sharded_prefixes, checkpoint_prefix, delete_old_dirs=True)
                with ops.control_dependencies([merge_step]):
                    return array_ops.identity(checkpoint_prefix)

    def _AddShardedSaveOps(self, filename_tensor, per_device):
        if False:
            while True:
                i = 10
        'Add ops to save the params per shard.\n\n    Args:\n      filename_tensor: a scalar String Tensor.\n      per_device: A list of (device, BaseSaverBuilder.SaveableObject) pairs, as\n        returned by _GroupByDevices().\n\n    Returns:\n      An op to save the variables.\n    '
        if self._write_version == saver_pb2.SaverDef.V2:
            return self._AddShardedSaveOpsForV2(filename_tensor, per_device)
        num_shards = len(per_device)
        sharded_saves = []
        num_shards_tensor = constant_op.constant(num_shards, name='num_shards')
        for (shard, (device, saveables)) in enumerate(per_device):
            with ops.device(device):
                sharded_filename = self.sharded_filename(filename_tensor, shard, num_shards_tensor)
                sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
        with ops.control_dependencies([x.op for x in sharded_saves]):
            return gen_io_ops.sharded_filespec(filename_tensor, num_shards_tensor)

    def _AddRestoreOps(self, filename_tensor, saveables, restore_sequentially, reshape, preferred_shard=-1, name='restore_all'):
        if False:
            print('Hello World!')
        'Add operations to restore saveables.\n\n    Args:\n      filename_tensor: Tensor for the path of the file to load.\n      saveables: A list of SaveableObject objects.\n      restore_sequentially: True if we want to restore variables sequentially\n        within a shard.\n      reshape: True if we want to reshape loaded tensors to the shape of the\n        corresponding variable.\n      preferred_shard: Shard to open first when loading a sharded file.\n      name: Name for the returned op.\n\n    Returns:\n      An Operation that restores the variables.\n    '
        all_tensors = self.bulk_restore(filename_tensor, saveables, preferred_shard, restore_sequentially)
        assign_ops = []
        idx = 0
        for saveable in saveables:
            shapes = None
            if reshape:
                shapes = []
                for spec in saveable.specs:
                    v = spec.tensor
                    shape = v.get_shape()
                    if not shape.is_fully_defined():
                        shape = array_ops.shape(v)
                    shapes.append(shape)
            saveable_tensors = all_tensors[idx:idx + len(saveable.specs)]
            idx += len(saveable.specs)
            assign_ops.append(saveable.restore(saveable_tensors, shapes))
        return control_flow_ops.group(*assign_ops, name=name)

    def _AddShardedRestoreOps(self, filename_tensor, per_device, restore_sequentially, reshape):
        if False:
            i = 10
            return i + 15
        'Add Ops to restore variables from multiple devices.\n\n    Args:\n      filename_tensor: Tensor for the path of the file to load.\n      per_device: A list of (device, SaveableObject) pairs, as returned by\n        _GroupByDevices().\n      restore_sequentially: True if we want to restore variables sequentially\n        within a shard.\n      reshape: True if we want to reshape loaded tensors to the shape of the\n        corresponding variable.\n\n    Returns:\n      An Operation that restores the variables.\n    '
        sharded_restores = []
        for (shard, (device, saveables)) in enumerate(per_device):
            with ops.device(device):
                sharded_restores.append(self._AddRestoreOps(filename_tensor, saveables, restore_sequentially, reshape, preferred_shard=shard, name='restore_shard'))
        return control_flow_ops.group(*sharded_restores, name='restore_all')

    def _GroupByDevices(self, saveables):
        if False:
            for i in range(10):
                print('nop')
        'Group Variable tensor slices per device.\n\n    TODO(touts): Make sure that all the devices found are on different\n    job/replica/task/cpu|gpu.  It would be bad if 2 were on the same device.\n    It can happen if the devices are unspecified.\n\n    Args:\n      saveables: A list of BaseSaverBuilder.SaveableObject objects.\n\n    Returns:\n      A list of tuples: (device_name, BaseSaverBuilder.SaveableObject) tuples.\n      The list is sorted by ascending device_name.\n\n    Raises:\n      ValueError: If the tensors of a saveable are on different devices.\n    '
        per_device = collections.defaultdict(lambda : [])
        for saveable in saveables:
            canonical_device = set((pydev.canonical_name(spec.device) for spec in saveable.specs))
            if len(canonical_device) != 1:
                raise ValueError('All tensors of a saveable object must be on the same device: %s' % saveable.name)
            per_device[canonical_device.pop()].append(saveable)
        return sorted(per_device.items(), key=lambda t: t[0])

    def build(self, names_to_saveables, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, filename='model'):
        if False:
            print('Hello World!')
        'Builds save/restore graph nodes or runs save/restore in eager mode.\n\n    Args:\n      names_to_saveables: A dictionary mapping name to a Variable or\n        SaveableObject. Each name will be associated with the corresponding\n        variable in the checkpoint.\n      reshape: If True, allow restoring parameters from a checkpoint that where\n        the parameters have a different shape.  This is only needed when you try\n        to restore from a Dist-Belief checkpoint, and only some times.\n      sharded: If True, shard the checkpoints, one per device that has Variable\n        nodes.\n      max_to_keep: Maximum number of checkpoints to keep.  As new checkpoints\n        are created, old ones are deleted.  If None or 0, no checkpoints are\n        deleted from the filesystem but only the last one is kept in the\n        `checkpoint` file.  Presently the number is only roughly enforced.  For\n        example in case of restarts more than max_to_keep checkpoints may be\n        kept.\n      keep_checkpoint_every_n_hours: How often checkpoints should be kept.\n        Defaults to 10,000 hours.\n      name: String.  Optional name to use as a prefix when adding operations.\n      restore_sequentially: A Bool, which if true, causes restore of different\n        variables to happen sequentially within each device.\n      filename: If known at graph construction time, filename used for variable\n        loading/saving. If None, then the default name "model" will be used.\n\n    Returns:\n      A SaverDef proto.\n\n    Raises:\n      TypeError: If \'names_to_saveables\' is not a dictionary mapping string\n        keys to variable Tensors.\n      ValueError: If any of the keys or values in \'names_to_saveables\' is not\n        unique.\n    '
        return self._build_internal(names_to_saveables=names_to_saveables, reshape=reshape, sharded=sharded, max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, name=name, restore_sequentially=restore_sequentially, filename=filename)

    def _build_internal(self, names_to_saveables, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, filename='model', build_save=True, build_restore=True):
        if False:
            for i in range(10):
                print('nop')
        'build() with option to only perform save and restore.'
        if not context.executing_eagerly() and (not build_save or not build_restore):
            raise ValueError('save and restore operations need to be built together  when eager execution is not enabled.')
        if not isinstance(names_to_saveables, dict):
            names_to_saveables = saveable_object_util.op_list_to_dict(names_to_saveables)
        saveables = saveable_object_util.validate_and_slice_inputs(names_to_saveables)
        if max_to_keep is None:
            max_to_keep = 0
        with ops.name_scope(name, 'save', [saveable.op for saveable in saveables]) as name:
            filename_tensor = array_ops.placeholder_with_default(filename or 'model', shape=(), name='filename')
            filename_tensor = array_ops.placeholder_with_default(filename_tensor, shape=(), name='Const')
            if sharded:
                per_device = self._GroupByDevices(saveables)
                if build_save:
                    save_tensor = self._AddShardedSaveOps(filename_tensor, per_device)
                if build_restore:
                    restore_op = self._AddShardedRestoreOps(filename_tensor, per_device, restore_sequentially, reshape)
            else:
                if build_save:
                    save_tensor = self._AddSaveOps(filename_tensor, saveables)
                if build_restore:
                    restore_op = self._AddRestoreOps(filename_tensor, saveables, restore_sequentially, reshape)
        if context.executing_eagerly():
            save_tensor_name = save_tensor.numpy() if build_save else ''
            return saver_pb2.SaverDef(filename_tensor_name=filename_tensor.numpy(), save_tensor_name=save_tensor_name, restore_op_name='', max_to_keep=max_to_keep, sharded=sharded, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, version=self._write_version)
        else:
            graph = ops.get_default_graph()
            check_collection_list = graph.get_all_collection_keys()
            for collection_type in check_collection_list:
                for element in graph.get_collection(collection_type):
                    if isinstance(element, variables.PartitionedVariable):
                        try:
                            graph.get_operation_by_name(element.name)
                        except KeyError:
                            element.as_tensor()
            return saver_pb2.SaverDef(filename_tensor_name=filename_tensor.name, save_tensor_name=save_tensor.name, restore_op_name=restore_op.name, max_to_keep=max_to_keep, sharded=sharded, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, version=self._write_version)

class BulkSaverBuilder(BaseSaverBuilder):
    """SaverBuilder with support for bulk restoring multiple saveables."""

    def bulk_restore(self, filename_tensor, saveables, preferred_shard, restore_sequentially):
        if False:
            while True:
                i = 10
        del restore_sequentially
        restore_specs = []
        for saveable in saveables:
            for spec in saveable.specs:
                restore_specs.append((spec.name, spec.slice_spec, spec.dtype))
        (names, slices, dtypes) = zip(*restore_specs)
        with ops.device('cpu:0'):
            return io_ops.restore_v2(filename_tensor, names, slices, dtypes)

def _get_saver_or_default():
    if False:
        i = 10
        return i + 15
    'Returns the saver from SAVERS collection, or creates a default one.\n\n  This method is used by other members of the training module, such as\n  `Scaffold`, or `CheckpointSaverHook`.\n\n  Returns:\n    `Saver`.\n\n  Raises:\n    RuntimeError: If the SAVERS collection already has more than one items.\n  '
    collection_key = ops.GraphKeys.SAVERS
    savers = ops.get_collection(collection_key)
    if savers:
        if len(savers) > 1:
            raise RuntimeError('More than one item in collection {}. Please indicate which one to use by passing it to the constructor.'.format(collection_key))
        return savers[0]
    saver = Saver(sharded=True, allow_empty=True)
    if saver is not None:
        ops.add_to_collection(collection_key, saver)
    return saver

@tf_export(v1=['train.Saver'])
class Saver:
    """Saves and restores variables.

  @compatibility(TF2)
  `tf.compat.v1.train.Saver` is not supported for saving and restoring
  checkpoints in TF2. Please switch to `tf.train.Checkpoint` or
  `tf.keras.Model.save_weights`, which perform a more robust [object-based
  saving](https://www.tensorflow.org/guide/checkpoint#loading_mechanics).

  ### How to Rewrite Checkpoints

  Please rewrite your checkpoints immediately using the object-based checkpoint
  APIs.

  You can load a name-based checkpoint written by `tf.compat.v1.train.Saver`
  using `tf.train.Checkpoint.restore` or `tf.keras.Model.load_weights`. However,
  you may have to change the names of the variables in your model to match the
  variable names in the name-based checkpoint, which can be viewed with
  `tf.train.list_variables(path)`.

  Another option is to create an `assignment_map` that maps the name of the
  variables in the name-based checkpoint to the variables in your model, eg:
  ```
  {
      'sequential/dense/bias': model.variables[0],
      'sequential/dense/kernel': model.variables[1]
  }
  ```
  and use `tf.compat.v1.train.init_from_checkpoint(path, assignment_map)` to
  restore the name-based checkpoint.

  After restoring, re-encode your checkpoint
  using `tf.train.Checkpoint.save` or `tf.keras.Model.save_weights`.

  See the [Checkpoint compatibility](
  https://www.tensorflow.org/guide/migrate#checkpoint_compatibility)
  section of the migration guide for more details.


  ### Checkpoint Management in TF2

  Use `tf.train.CheckpointManager` to manage checkpoints in TF2.
  `tf.train.CheckpointManager` offers equivalent `keep_checkpoint_every_n_hours`
  and `max_to_keep` parameters.

  To recover the latest checkpoint,

  ```
  checkpoint = tf.train.Checkpoint(model)
  manager = tf.train.CheckpointManager(checkpoint)
  status = checkpoint.restore(manager.latest_checkpoint)
  ```

  `tf.train.CheckpointManager` also writes a [`CheckpointState` proto]
  (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/checkpoint_state.proto)
  which contains the timestamp when each checkpoint was created.

  ### Writing `MetaGraphDef`s in TF2

  To replace, `tf.compat.v1.train.Saver.save(write_meta_graph=True)`, use
  `tf.saved_model.save` to write the `MetaGraphDef` (which is contained in
  `saved_model.pb`).

  @end_compatibility

  See [Variables](https://tensorflow.org/guide/variables)
  for an overview of variables, saving and restoring.

  The `Saver` class adds ops to save and restore variables to and from
  *checkpoints*.  It also provides convenience methods to run these ops.

  Checkpoints are binary files in a proprietary format which map variable names
  to tensor values.  The best way to examine the contents of a checkpoint is to
  load it using a `Saver`.

  Savers can automatically number checkpoint filenames with a provided counter.
  This lets you keep multiple checkpoints at different steps while training a
  model.  For example you can number the checkpoint filenames with the training
  step number.  To avoid filling up disks, savers manage checkpoint files
  automatically. For example, they can keep only the N most recent files, or
  one checkpoint for every N hours of training.

  You number checkpoint filenames by passing a value to the optional
  `global_step` argument to `save()`:

  ```python
  saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
  ...
  saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
  ```

  Additionally, optional arguments to the `Saver()` constructor let you control
  the proliferation of checkpoint files on disk:

  * `max_to_keep` indicates the maximum number of recent checkpoint files to
    keep.  As new files are created, older files are deleted.   If None or 0,
    no checkpoints are deleted from the filesystem but only the last one is
    kept in the `checkpoint` file.  Defaults to 5 (that is, the 5 most recent
    checkpoint files are kept.)

  * `keep_checkpoint_every_n_hours`: In addition to keeping the most recent
    `max_to_keep` checkpoint files, you might want to keep one checkpoint file
    for every N hours of training.  This can be useful if you want to later
    analyze how a model progressed during a long training session.  For
    example, passing `keep_checkpoint_every_n_hours=2` ensures that you keep
    one checkpoint file for every 2 hours of training.  The default value of
    10,000 hours effectively disables the feature.

  Note that you still have to call the `save()` method to save the model.
  Passing these arguments to the constructor will not save variables
  automatically for you.

  A training program that saves regularly looks like:

  ```python
  ...
  # Create a saver.
  saver = tf.compat.v1.train.Saver(...variables...)
  # Launch the graph and train, saving the model every 1,000 steps.
  sess = tf.compat.v1.Session()
  for step in range(1000000):
      sess.run(..training_op..)
      if step % 1000 == 0:
          # Append the step number to the checkpoint name:
          saver.save(sess, 'my-model', global_step=step)
  ```

  In addition to checkpoint files, savers keep a protocol buffer on disk with
  the list of recent checkpoints. This is used to manage numbered checkpoint
  files and by `latest_checkpoint()`, which makes it easy to discover the path
  to the most recent checkpoint. That protocol buffer is stored in a file named
  'checkpoint' next to the checkpoint files.

  If you create several savers, you can specify a different filename for the
  protocol buffer file in the call to `save()`.
  """

    def __init__(self, var_list=None, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, saver_def=None, builder=None, defer_build=False, allow_empty=False, write_version=saver_pb2.SaverDef.V2, pad_step_number=False, save_relative_paths=False, filename=None):
        if False:
            while True:
                i = 10
        "Creates a `Saver`.\n\n    The constructor adds ops to save and restore variables.\n\n    `var_list` specifies the variables that will be saved and restored. It can\n    be passed as a `dict` or a list:\n\n    * A `dict` of names to variables: The keys are the names that will be\n      used to save or restore the variables in the checkpoint files.\n    * A list of variables: The variables will be keyed with their op name in\n      the checkpoint files.\n\n    For example:\n\n    ```python\n    v1 = tf.Variable(..., name='v1')\n    v2 = tf.Variable(..., name='v2')\n\n    # Pass the variables as a dict:\n    saver = tf.compat.v1.train.Saver({'v1': v1, 'v2': v2})\n\n    # Or pass them as a list.\n    saver = tf.compat.v1.train.Saver([v1, v2])\n    # Passing a list is equivalent to passing a dict with the variable op names\n    # as keys:\n    saver = tf.compat.v1.train.Saver({v.op.name: v for v in [v1, v2]})\n    ```\n\n    Note: the newer `AutoTrackable` API is not supported by `Saver`. In this\n    case, the `tf.train.Checkpoint` class should be used.\n\n    The optional `reshape` argument, if `True`, allows restoring a variable from\n    a save file where the variable had a different shape, but the same number\n    of elements and type.  This is useful if you have reshaped a variable and\n    want to reload it from an older checkpoint.\n\n    The optional `sharded` argument, if `True`, instructs the saver to shard\n    checkpoints per device.\n\n    Args:\n      var_list: A list of `Variable`/`SaveableObject`, or a dictionary mapping\n        names to `SaveableObject`s. If `None`, defaults to the list of all\n        saveable objects.\n      reshape: If `True`, allows restoring parameters from a checkpoint where\n        the variables have a different shape.\n      sharded: If `True`, shard the checkpoints, one per device.\n      max_to_keep: Maximum number of recent checkpoints to keep. Defaults to 5.\n      keep_checkpoint_every_n_hours: How often to keep checkpoints. Defaults to\n        10,000 hours.\n      name: String.  Optional name to use as a prefix when adding operations.\n      restore_sequentially: A `Bool`, which if true, causes restore of different\n        variables to happen sequentially within each device.  This can lower\n        memory usage when restoring very large models.\n      saver_def: Optional `SaverDef` proto to use instead of running the\n        builder. This is only useful for specialty code that wants to recreate a\n        `Saver` object for a previously built `Graph` that had a `Saver`. The\n        `saver_def` proto should be the one returned by the `as_saver_def()`\n        call of the `Saver` that was created for that `Graph`.\n      builder: Optional `SaverBuilder` to use if a `saver_def` was not provided.\n        Defaults to `BulkSaverBuilder()`.\n      defer_build: If `True`, defer adding the save and restore ops to the\n        `build()` call. In that case `build()` should be called before\n        finalizing the graph or using the saver.\n      allow_empty: If `False` (default) raise an error if there are no variables\n        in the graph. Otherwise, construct the saver anyway and make it a no-op.\n      write_version: controls what format to use when saving checkpoints.  It\n        also affects certain filepath matching logic.  The V2 format is the\n        recommended choice: it is much more optimized than V1 in terms of memory\n        required and latency incurred during restore.  Regardless of this flag,\n        the Saver is able to restore from both V2 and V1 checkpoints.\n      pad_step_number: if True, pads the global step number in the checkpoint\n        filepaths to some fixed width (8 by default).  This is turned off by\n        default.\n      save_relative_paths: If `True`, will write relative paths to the\n        checkpoint state file. This is needed if the user wants to copy the\n        checkpoint directory and reload from the copied directory.\n      filename: If known at graph construction time, filename used for variable\n        loading/saving.\n\n    Raises:\n      TypeError: If `var_list` is invalid.\n      ValueError: If any of the keys or values in `var_list` are not unique.\n      RuntimeError: If eager execution is enabled and`var_list` does not specify\n        a list of variables to save.\n\n    @compatibility(eager)\n    When eager execution is enabled, `var_list` must specify a `list` or `dict`\n    of variables to save. Otherwise, a `RuntimeError` will be raised.\n\n    Although Saver works in some cases when executing eagerly, it is\n    fragile. Please switch to `tf.train.Checkpoint` or\n    `tf.keras.Model.save_weights`, which perform a more robust object-based\n    saving. These APIs will load checkpoints written by `Saver`.\n    @end_compatibility\n    "
        global _END_TIME_OF_LAST_WRITE
        with _END_TIME_OF_LAST_WRITE_LOCK:
            if _END_TIME_OF_LAST_WRITE is None:
                _END_TIME_OF_LAST_WRITE = time.time()
        if defer_build and var_list:
            raise ValueError('If `var_list` is provided then build cannot be deferred. Either set defer_build=False or var_list=None.')
        if context.executing_eagerly():
            logging.warning('Saver is deprecated, please switch to tf.train.Checkpoint or tf.keras.Model.save_weights for training checkpoints. When executing eagerly variables do not necessarily have unique names, and so the variable.name-based lookups Saver performs are error-prone.')
            if var_list is None:
                raise RuntimeError('When eager execution is enabled, `var_list` must specify a list or dict of variables to save')
        self._var_list = var_list
        self._reshape = reshape
        self._sharded = sharded
        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self._name = name
        self._restore_sequentially = restore_sequentially
        self.saver_def = saver_def
        self._builder = builder
        self._is_built = False
        self._allow_empty = allow_empty
        self._is_empty = None
        self._write_version = write_version
        self._pad_step_number = pad_step_number
        self._filename = filename
        self._last_checkpoints = []
        self._checkpoints_to_be_deleted = []
        if context.executing_eagerly():
            self._next_checkpoint_time = time.time() + self._keep_checkpoint_every_n_hours * 3600
        elif not defer_build:
            self.build()
        if self.saver_def:
            self._check_saver_def()
            self._write_version = self.saver_def.version
        self._save_relative_paths = save_relative_paths
        self._object_restore_saver = None

    def build(self):
        if False:
            i = 10
            return i + 15
        if context.executing_eagerly():
            raise RuntimeError('Use save/restore instead of build in eager mode.')
        self._build(self._filename, build_save=True, build_restore=True)

    def _build_eager(self, checkpoint_path, build_save, build_restore):
        if False:
            while True:
                i = 10
        self._build(checkpoint_path, build_save=build_save, build_restore=build_restore)

    def _build(self, checkpoint_path, build_save, build_restore):
        if False:
            print('Hello World!')
        'Builds saver_def.'
        if not context.executing_eagerly():
            if self._is_built:
                return
            self._is_built = True
        if not self.saver_def or context.executing_eagerly():
            if self._builder is None:
                self._builder = BulkSaverBuilder(self._write_version)
            if self._var_list is None:
                self._var_list = variables._all_saveable_objects()
            if not self._var_list:
                if self._allow_empty:
                    self._is_empty = True
                    return
                else:
                    raise ValueError('No variables to save')
            self._is_empty = False
            self.saver_def = self._builder._build_internal(self._var_list, reshape=self._reshape, sharded=self._sharded, max_to_keep=self._max_to_keep, keep_checkpoint_every_n_hours=self._keep_checkpoint_every_n_hours, name=self._name, restore_sequentially=self._restore_sequentially, filename=checkpoint_path, build_save=build_save, build_restore=build_restore)
        elif self.saver_def and self._name:
            self.saver_def.filename_tensor_name = ops.prepend_name_scope(self.saver_def.filename_tensor_name, self._name)
            self.saver_def.save_tensor_name = ops.prepend_name_scope(self.saver_def.save_tensor_name, self._name)
            self.saver_def.restore_op_name = ops.prepend_name_scope(self.saver_def.restore_op_name, self._name)
        self._check_saver_def()
        if not context.executing_eagerly():
            self._next_checkpoint_time = time.time() + self.saver_def.keep_checkpoint_every_n_hours * 3600

    def _check_saver_def(self):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(self.saver_def, saver_pb2.SaverDef):
            raise ValueError('saver_def must be a saver_pb2.SaverDef: %s' % self.saver_def)
        if not context.executing_eagerly():
            if not self.saver_def.save_tensor_name:
                raise ValueError('saver_def must specify the save_tensor_name: %s' % str(self.saver_def))
            if not self.saver_def.restore_op_name:
                raise ValueError('saver_def must specify the restore_op_name: %s' % str(self.saver_def))

    def _CheckpointFilename(self, p):
        if False:
            print('Hello World!')
        'Returns the checkpoint filename given a `(filename, time)` pair.\n\n    Args:\n      p: (filename, time) pair.\n\n    Returns:\n      Checkpoint file name.\n    '
        (name, _) = p
        return name

    def _RecordLastCheckpoint(self, latest_save_path):
        if False:
            while True:
                i = 10
        'Manages the list of the latest checkpoints.'
        if not self.saver_def.max_to_keep:
            return
        for p in self._last_checkpoints:
            if latest_save_path == self._CheckpointFilename(p):
                self._last_checkpoints.remove(p)
        self._last_checkpoints.append((latest_save_path, time.time()))
        if len(self._last_checkpoints) > self.saver_def.max_to_keep:
            self._checkpoints_to_be_deleted.append(self._last_checkpoints.pop(0))

    def _MaybeDeleteOldCheckpoints(self, meta_graph_suffix='meta'):
        if False:
            print('Hello World!')
        "Deletes old checkpoints if necessary.\n\n    `self._checkpoints_to_be_deleted` is going to contain checkpoints that are\n    over `max_to_keep`.  They are going to be deleted.  If\n    `keep_checkpoint_every_n_hours` was specified, keep an additional checkpoint\n    every `N` hours. For example, if `N` is 0.5, an additional checkpoint is\n    kept for every 0.5 hours of training; if `N` is 10, an additional\n    checkpoint is kept for every 10 hours of training.\n\n    Args:\n      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.\n    "
        if self._checkpoints_to_be_deleted:
            p = self._checkpoints_to_be_deleted.pop(0)
            should_keep = p[1] > self._next_checkpoint_time
            if should_keep:
                self._next_checkpoint_time += self.saver_def.keep_checkpoint_every_n_hours * 3600
                return
            try:
                checkpoint_management.remove_checkpoint(self._CheckpointFilename(p), self.saver_def.version, meta_graph_suffix)
            except Exception as e:
                logging.warning('Ignoring: %s', str(e))

    def as_saver_def(self):
        if False:
            print('Hello World!')
        'Generates a `SaverDef` representation of this saver.\n\n    Returns:\n      A `SaverDef` proto.\n    '
        return self.saver_def

    def to_proto(self, export_scope=None):
        if False:
            while True:
                i = 10
        'Converts this `Saver` to a `SaverDef` protocol buffer.\n\n    Args:\n      export_scope: Optional `string`. Name scope to remove.\n\n    Returns:\n      A `SaverDef` protocol buffer.\n    '
        if export_scope is None:
            return self.saver_def
        if not (self.saver_def.filename_tensor_name.startswith(export_scope) and self.saver_def.save_tensor_name.startswith(export_scope) and self.saver_def.restore_op_name.startswith(export_scope)):
            return None
        saver_def = saver_pb2.SaverDef()
        saver_def.CopyFrom(self.saver_def)
        saver_def.filename_tensor_name = ops.strip_name_scope(saver_def.filename_tensor_name, export_scope)
        saver_def.save_tensor_name = ops.strip_name_scope(saver_def.save_tensor_name, export_scope)
        saver_def.restore_op_name = ops.strip_name_scope(saver_def.restore_op_name, export_scope)
        return saver_def

    @staticmethod
    def from_proto(saver_def, import_scope=None):
        if False:
            while True:
                i = 10
        'Returns a `Saver` object created from `saver_def`.\n\n    Args:\n      saver_def: a `SaverDef` protocol buffer.\n      import_scope: Optional `string`. Name scope to use.\n\n    Returns:\n      A `Saver` built from saver_def.\n    '
        return Saver(saver_def=saver_def, name=import_scope)

    @property
    def last_checkpoints(self):
        if False:
            i = 10
            return i + 15
        'List of not-yet-deleted checkpoint filenames.\n\n    You can pass any of the returned values to `restore()`.\n\n    Returns:\n      A list of checkpoint filenames, sorted from oldest to newest.\n    '
        return list((self._CheckpointFilename(p) for p in self._last_checkpoints))

    def set_last_checkpoints(self, last_checkpoints):
        if False:
            return 10
        'DEPRECATED: Use set_last_checkpoints_with_time.\n\n    Sets the list of old checkpoint filenames.\n\n    Args:\n      last_checkpoints: A list of checkpoint filenames.\n\n    Raises:\n      AssertionError: If last_checkpoints is not a list.\n    '
        assert isinstance(last_checkpoints, list)
        self._last_checkpoints = [(s, np.inf) for s in last_checkpoints]

    def set_last_checkpoints_with_time(self, last_checkpoints_with_time):
        if False:
            while True:
                i = 10
        'Sets the list of old checkpoint filenames and timestamps.\n\n    Args:\n      last_checkpoints_with_time: A list of tuples of checkpoint filenames and\n        timestamps.\n\n    Raises:\n      AssertionError: If last_checkpoints_with_time is not a list.\n    '
        assert isinstance(last_checkpoints_with_time, list)
        self._last_checkpoints = last_checkpoints_with_time

    def recover_last_checkpoints(self, checkpoint_paths):
        if False:
            print('Hello World!')
        'Recovers the internal saver state after a crash.\n\n    This method is useful for recovering the "self._last_checkpoints" state.\n\n    Globs for the checkpoints pointed to by `checkpoint_paths`.  If the files\n    exist, use their mtime as the checkpoint timestamp.\n\n    Args:\n      checkpoint_paths: a list of checkpoint paths.\n    '
        checkpoints_with_mtimes = []
        for checkpoint_path in checkpoint_paths:
            try:
                mtime = checkpoint_management.get_checkpoint_mtimes([checkpoint_path])
            except errors.NotFoundError:
                continue
            if mtime:
                checkpoints_with_mtimes.append((checkpoint_path, mtime[0]))
        self.set_last_checkpoints_with_time(checkpoints_with_mtimes)

    def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True, write_state=True, strip_default_attrs=False, save_debug_info=False):
        if False:
            while True:
                i = 10
        "Saves variables.\n\n    This method runs the ops added by the constructor for saving variables.\n    It requires a session in which the graph was launched.  The variables to\n    save must also have been initialized.\n\n    The method returns the path prefix of the newly created checkpoint files.\n    This string can be passed directly to a call to `restore()`.\n\n    Args:\n      sess: A Session to use to save the variables.\n      save_path: String.  Prefix of filenames created for the checkpoint.\n      global_step: If provided the global step number is appended to `save_path`\n        to create the checkpoint filenames. The optional argument can be a\n        `Tensor`, a `Tensor` name or an integer.\n      latest_filename: Optional name for the protocol buffer file that will\n        contains the list of most recent checkpoints.  That file, kept in the\n        same directory as the checkpoint files, is automatically managed by the\n        saver to keep track of recent checkpoints.  Defaults to 'checkpoint'.\n      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.\n      write_meta_graph: `Boolean` indicating whether or not to write the meta\n        graph file.\n      write_state: `Boolean` indicating whether or not to write the\n        `CheckpointStateProto`.\n      strip_default_attrs: Boolean. If `True`, default-valued attributes will be\n        removed from the NodeDefs. For a detailed guide, see [Stripping\n        Default-Valued\n        Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).\n      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,\n        which in the same directory of save_path and with `_debug` added before\n        the file extension. This is only enabled when `write_meta_graph` is\n        `True`\n\n    Returns:\n      A string: path prefix used for the checkpoint files.  If the saver is\n        sharded, this string ends with: '-?????-of-nnnnn' where 'nnnnn'\n        is the number of shards created.\n      If the saver is empty, returns None.\n\n    Raises:\n      TypeError: If `sess` is not a `Session`.\n      ValueError: If `latest_filename` contains path components, or if it\n        collides with `save_path`.\n      RuntimeError: If save and restore ops weren't built.\n    "
        start_time = time.time()
        if not self._is_built and (not context.executing_eagerly()):
            raise RuntimeError('`build()` should be called before save if defer_build==True')
        if latest_filename is None:
            latest_filename = 'checkpoint'
        if self._write_version != saver_pb2.SaverDef.V2:
            logging.warning('*******************************************************')
            logging.warning("TensorFlow's V1 checkpoint format has been deprecated.")
            logging.warning('Consider switching to the more efficient V2 format:')
            logging.warning('   `tf.train.Saver(write_version=tf.train.SaverDef.V2)`')
            logging.warning('now on by default.')
            logging.warning('*******************************************************')
        if os.path.split(latest_filename)[0]:
            raise ValueError("'latest_filename' must not contain path components")
        save_path = compat.as_str(save_path)
        if global_step is not None:
            if not isinstance(global_step, compat.integral_types):
                global_step = training_util.global_step(sess, global_step)
            checkpoint_file = '%s-%d' % (save_path, global_step)
            if self._pad_step_number:
                checkpoint_file = '%s-%s' % (save_path, '{:08d}'.format(global_step))
        else:
            checkpoint_file = save_path
            if os.path.basename(save_path) == latest_filename and (not self._sharded):
                raise ValueError("'latest_filename' collides with 'save_path': '%s' and '%s'" % (latest_filename, save_path))
        if not context.executing_eagerly() and (not isinstance(sess, session.SessionInterface)):
            raise TypeError("'sess' must be a Session; %s" % sess)
        save_path_parent = os.path.dirname(save_path)
        if not self._is_empty:
            try:
                if context.executing_eagerly():
                    self._build_eager(checkpoint_file, build_save=True, build_restore=False)
                    model_checkpoint_path = self.saver_def.save_tensor_name
                else:
                    model_checkpoint_path = sess.run(self.saver_def.save_tensor_name, {self.saver_def.filename_tensor_name: checkpoint_file})
                model_checkpoint_path = compat.as_str(model_checkpoint_path)
                if write_state:
                    self._RecordLastCheckpoint(model_checkpoint_path)
                    checkpoint_management.update_checkpoint_state_internal(save_dir=save_path_parent, model_checkpoint_path=model_checkpoint_path, all_model_checkpoint_paths=self.last_checkpoints, latest_filename=latest_filename, save_relative_paths=self._save_relative_paths)
                    self._MaybeDeleteOldCheckpoints(meta_graph_suffix=meta_graph_suffix)
            except (errors.FailedPreconditionError, errors.NotFoundError) as exc:
                if not gfile.IsDirectory(save_path_parent):
                    exc = ValueError("Parent directory of {} doesn't exist, can't save.".format(save_path))
                raise exc
        end_time = time.time()
        metrics.AddCheckpointWriteDuration(api_label=_SAVER_LABEL, microseconds=_get_duration_microseconds(start_time, end_time))
        global _END_TIME_OF_LAST_WRITE
        with _END_TIME_OF_LAST_WRITE_LOCK:
            metrics.AddTrainingTimeSaved(api_label=_SAVER_LABEL, microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE, end_time))
            _END_TIME_OF_LAST_WRITE = end_time
        if write_meta_graph:
            meta_graph_filename = checkpoint_management.meta_graph_filename(checkpoint_file, meta_graph_suffix=meta_graph_suffix)
            if not context.executing_eagerly():
                with sess.graph.as_default():
                    self.export_meta_graph(meta_graph_filename, strip_default_attrs=strip_default_attrs, save_debug_info=save_debug_info)
        if self._is_empty:
            return None
        else:
            metrics.RecordCheckpointSize(api_label=_SAVER_LABEL, filesize=_get_checkpoint_size(model_checkpoint_path))
            return model_checkpoint_path

    def export_meta_graph(self, filename=None, collection_list=None, as_text=False, export_scope=None, clear_devices=False, clear_extraneous_savers=False, strip_default_attrs=False, save_debug_info=False):
        if False:
            for i in range(10):
                print('nop')
        'Writes `MetaGraphDef` to save_path/filename.\n\n    Args:\n      filename: Optional meta_graph filename including the path.\n      collection_list: List of string keys to collect.\n      as_text: If `True`, writes the meta_graph as an ASCII proto.\n      export_scope: Optional `string`. Name scope to remove.\n      clear_devices: Whether or not to clear the device field for an `Operation`\n        or `Tensor` during export.\n      clear_extraneous_savers: Remove any Saver-related information from the\n        graph (both Save/Restore ops and SaverDefs) that are not associated with\n        this Saver.\n      strip_default_attrs: Boolean. If `True`, default-valued attributes will be\n        removed from the NodeDefs. For a detailed guide, see [Stripping\n        Default-Valued\n        Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).\n      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,\n        which in the same directory of filename and with `_debug` added before\n        the file extension.\n\n    Returns:\n      A `MetaGraphDef` proto.\n    '
        return export_meta_graph(filename=filename, graph_def=ops.get_default_graph().as_graph_def(add_shapes=True, use_pybind11_proto=True), saver_def=self.saver_def, collection_list=collection_list, as_text=as_text, export_scope=export_scope, clear_devices=clear_devices, clear_extraneous_savers=clear_extraneous_savers, strip_default_attrs=strip_default_attrs, save_debug_info=save_debug_info)

    def restore(self, sess, save_path):
        if False:
            i = 10
            return i + 15
        'Restores previously saved variables.\n\n    This method runs the ops added by the constructor for restoring variables.\n    It requires a session in which the graph was launched.  The variables to\n    restore do not have to have been initialized, as restoring is itself a way\n    to initialize variables.\n\n    The `save_path` argument is typically a value previously returned from a\n    `save()` call, or a call to `latest_checkpoint()`.\n\n    Args:\n      sess: A `Session` to use to restore the parameters. None in eager mode.\n      save_path: Path where parameters were previously saved.\n\n    Raises:\n      ValueError: If save_path is None or not a valid checkpoint.\n    '
        start_time = time.time()
        if self._is_empty:
            return
        if save_path is None:
            raise ValueError("Can't load save_path when it is None.")
        checkpoint_prefix = compat.as_text(save_path)
        if not checkpoint_management.checkpoint_exists_internal(checkpoint_prefix):
            raise ValueError('The passed save_path is not a valid checkpoint: ' + checkpoint_prefix)
        logging.info('Restoring parameters from %s', checkpoint_prefix)
        try:
            if context.executing_eagerly():
                self._build_eager(save_path, build_save=False, build_restore=True)
            else:
                sess.run(self.saver_def.restore_op_name, {self.saver_def.filename_tensor_name: save_path})
        except errors.NotFoundError as err:
            try:
                names_to_keys = object_graph_key_mapping(save_path)
            except errors.NotFoundError:
                raise _wrap_restore_error_with_msg(err, 'a Variable name or other graph key that is missing')
            logging.warning('Restoring an object-based checkpoint using a name-based saver. This may be somewhat fragile, and will re-build the Saver. Instead, consider loading object-based checkpoints using tf.train.Checkpoint().')
            self._object_restore_saver = saver_from_object_based_checkpoint(checkpoint_path=save_path, var_list=self._var_list, builder=self._builder, names_to_keys=names_to_keys, cached_saver=self._object_restore_saver)
            self._object_restore_saver.restore(sess=sess, save_path=save_path)
        except errors.InvalidArgumentError as err:
            raise _wrap_restore_error_with_msg(err, 'a mismatch between the current graph and the graph')
        metrics.AddCheckpointReadDuration(api_label=_SAVER_LABEL, microseconds=_get_duration_microseconds(start_time, time.time()))

    @staticmethod
    def _add_collection_def(meta_graph_def, key, export_scope=None):
        if False:
            for i in range(10):
                print('nop')
        'Adds a collection to MetaGraphDef protocol buffer.\n\n    Args:\n      meta_graph_def: MetaGraphDef protocol buffer.\n      key: One of the GraphKeys or user-defined string.\n      export_scope: Optional `string`. Name scope to remove.\n    '
        meta_graph.add_collection_def(meta_graph_def, key, export_scope=export_scope)

@tf_export(v1=['train.import_meta_graph'])
def import_meta_graph(meta_graph_or_file, clear_devices=False, import_scope=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Recreates a Graph saved in a `MetaGraphDef` proto.\n\n  This function takes a `MetaGraphDef` protocol buffer as input. If\n  the argument is a file containing a `MetaGraphDef` protocol buffer ,\n  it constructs a protocol buffer from the file content. The function\n  then adds all the nodes from the `graph_def` field to the\n  current graph, recreates all the collections, and returns a saver\n  constructed from the `saver_def` field.\n\n  In combination with `export_meta_graph()`, this function can be used to\n\n  * Serialize a graph along with other Python objects such as `QueueRunner`,\n    `Variable` into a `MetaGraphDef`.\n\n  * Restart training from a saved graph and checkpoints.\n\n  * Run inference from a saved graph and checkpoints.\n\n  ```Python\n  ...\n  # Create a saver.\n  saver = tf.compat.v1.train.Saver(...variables...)\n  # Remember the training_op we want to run by adding it to a collection.\n  tf.compat.v1.add_to_collection(\'train_op\', train_op)\n  sess = tf.compat.v1.Session()\n  for step in range(1000000):\n      sess.run(train_op)\n      if step % 1000 == 0:\n          # Saves checkpoint, which by default also exports a meta_graph\n          # named \'my-model-global_step.meta\'.\n          saver.save(sess, \'my-model\', global_step=step)\n  ```\n\n  Later we can continue training from this saved `meta_graph` without building\n  the model from scratch.\n\n  ```Python\n  with tf.Session() as sess:\n    new_saver =\n    tf.train.import_meta_graph(\'my-save-dir/my-model-10000.meta\')\n    new_saver.restore(sess, \'my-save-dir/my-model-10000\')\n    # tf.get_collection() returns a list. In this example we only want\n    # the first one.\n    train_op = tf.get_collection(\'train_op\')[0]\n    for step in range(1000000):\n      sess.run(train_op)\n  ```\n\n  NOTE: Restarting training from saved `meta_graph` only works if the\n  device assignments have not changed.\n\n  Example:\n  Variables, placeholders, and independent operations can also be stored, as\n  shown in the following example.\n\n  ```Python\n  # Saving contents and operations.\n  v1 = tf.placeholder(tf.float32, name="v1")\n  v2 = tf.placeholder(tf.float32, name="v2")\n  v3 = tf.math.multiply(v1, v2)\n  vx = tf.Variable(10.0, name="vx")\n  v4 = tf.add(v3, vx, name="v4")\n  saver = tf.train.Saver([vx])\n  sess = tf.Session()\n  sess.run(tf.global_variables_initializer())\n  sess.run(vx.assign(tf.add(vx, vx)))\n  result = sess.run(v4, feed_dict={v1:12.0, v2:3.3})\n  print(result)\n  saver.save(sess, "./model_ex1")\n  ```\n\n  Later this model can be restored and contents loaded.\n\n  ```Python\n  # Restoring variables and running operations.\n  saver = tf.train.import_meta_graph("./model_ex1.meta")\n  sess = tf.Session()\n  saver.restore(sess, "./model_ex1")\n  result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 3.3})\n  print(result)\n  ```\n\n  Args:\n    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including\n      the path) containing a `MetaGraphDef`.\n    clear_devices: Whether or not to clear the device field for an `Operation`\n      or `Tensor` during import.\n    import_scope: Optional `string`. Name scope to add. Only used when\n      initializing from protocol buffer.\n    **kwargs: Optional keyed arguments.\n\n  Returns:\n    A saver constructed from `saver_def` in `MetaGraphDef` or None.\n\n    A None value is returned if no variables exist in the `MetaGraphDef`\n    (i.e., there are no variables to restore).\n\n  Raises:\n    RuntimeError: If called with eager execution enabled.\n\n  @compatibility(eager)\n  Exporting/importing meta graphs is not supported. No graph exists when eager\n  execution is enabled.\n  @end_compatibility\n  '
    return _import_meta_graph_with_return_elements(meta_graph_or_file, clear_devices, import_scope, **kwargs)[0]

def _import_meta_graph_with_return_elements(meta_graph_or_file, clear_devices=False, import_scope=None, return_elements=None, **kwargs):
    if False:
        while True:
            i = 10
    'Import MetaGraph, and return both a saver and returned elements.'
    if context.executing_eagerly():
        raise RuntimeError('Exporting/importing meta graphs is not supported when eager execution is enabled. No graph exists when eager execution is enabled.')
    if not isinstance(meta_graph_or_file, meta_graph_pb2.MetaGraphDef):
        meta_graph_def = meta_graph.read_meta_graph_file(meta_graph_or_file)
    else:
        meta_graph_def = meta_graph_or_file
    (imported_vars, imported_return_elements) = meta_graph.import_scoped_meta_graph_with_return_elements(meta_graph_def, clear_devices=clear_devices, import_scope=import_scope, return_elements=return_elements, **kwargs)
    saver = _create_saver_from_imported_meta_graph(meta_graph_def, import_scope, imported_vars)
    return (saver, imported_return_elements)

def _create_saver_from_imported_meta_graph(meta_graph_def, import_scope, imported_vars):
    if False:
        i = 10
        return i + 15
    'Return a saver for restoring variable values to an imported MetaGraph.'
    if meta_graph_def.HasField('saver_def'):
        scope = import_scope
        var_names = list(imported_vars.keys())
        if var_names:
            sample_key = var_names[0]
            sample_var = imported_vars[sample_key]
            scope = sample_var.name[:-len(sample_key)]
        return Saver(saver_def=meta_graph_def.saver_def, name=scope)
    elif variables._all_saveable_objects(scope=import_scope):
        return Saver()
    else:
        logging.info('Saver not created because there are no variables in the graph to restore')
        return None

@tf_export(v1=['train.export_meta_graph'])
def export_meta_graph(filename=None, meta_info_def=None, graph_def=None, saver_def=None, collection_list=None, as_text=False, graph=None, export_scope=None, clear_devices=False, clear_extraneous_savers=False, strip_default_attrs=False, save_debug_info=False, **kwargs):
    if False:
        print('Hello World!')
    'Returns `MetaGraphDef` proto.\n\n  Optionally writes it to filename.\n\n  This function exports the graph, saver, and collection objects into\n  `MetaGraphDef` protocol buffer with the intention of it being imported\n  at a later time or location to restart training, run inference, or be\n  a subgraph.\n\n  Args:\n    filename: Optional filename including the path for writing the generated\n      `MetaGraphDef` protocol buffer.\n    meta_info_def: `MetaInfoDef` protocol buffer.\n    graph_def: `GraphDef` protocol buffer.\n    saver_def: `SaverDef` protocol buffer.\n    collection_list: List of string keys to collect.\n    as_text: If `True`, writes the `MetaGraphDef` as an ASCII proto.\n    graph: The `Graph` to export. If `None`, use the default graph.\n    export_scope: Optional `string`. Name scope under which to extract the\n      subgraph. The scope name will be striped from the node definitions for\n      easy import later into new name scopes. If `None`, the whole graph is\n      exported. graph_def and export_scope cannot both be specified.\n    clear_devices: Whether or not to clear the device field for an `Operation`\n      or `Tensor` during export.\n    clear_extraneous_savers: Remove any Saver-related information from the graph\n      (both Save/Restore ops and SaverDefs) that are not associated with the\n      provided SaverDef.\n    strip_default_attrs: Boolean. If `True`, default-valued attributes will be\n      removed from the NodeDefs. For a detailed guide, see [Stripping\n      Default-Valued\n      Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).\n    save_debug_info: If `True`, save the GraphDebugInfo to a separate file,\n      which in the same directory of filename and with `_debug` added before the\n      file extend.\n    **kwargs: Optional keyed arguments.\n\n  Returns:\n    A `MetaGraphDef` proto.\n\n  Raises:\n    ValueError: When the `GraphDef` is larger than 2GB.\n    RuntimeError: If called with eager execution enabled.\n\n  @compatibility(eager)\n  Exporting/importing meta graphs is not supported unless both `graph_def` and\n  `graph` are provided. No graph exists when eager execution is enabled.\n  @end_compatibility\n  '
    if context.executing_eagerly() and (not (graph_def is not None and graph is not None)):
        raise RuntimeError('Exporting/importing meta graphs is not supported when eager execution is enabled. No graph exists when eager execution is enabled.')
    (meta_graph_def, _) = meta_graph.export_scoped_meta_graph(filename=filename, meta_info_def=meta_info_def, graph_def=graph_def, saver_def=saver_def, collection_list=collection_list, as_text=as_text, graph=graph, export_scope=export_scope, clear_devices=clear_devices, clear_extraneous_savers=clear_extraneous_savers, strip_default_attrs=strip_default_attrs, save_debug_info=save_debug_info, **kwargs)
    return meta_graph_def

def _wrap_restore_error_with_msg(err, extra_verbiage):
    if False:
        i = 10
        return i + 15
    err_msg = 'Restoring from checkpoint failed. This is most likely due to {} from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:\n\n{}'.format(extra_verbiage, err.message)
    return err.__class__(err.node_def, err.op, err_msg)
ops.register_proto_function(ops.GraphKeys.SAVERS, proto_type=saver_pb2.SaverDef, to_proto=Saver.to_proto, from_proto=Saver.from_proto)

def object_graph_key_mapping(checkpoint_path):
    if False:
        i = 10
        return i + 15
    'Return name to key mappings from the checkpoint.\n\n  Args:\n    checkpoint_path: string, path to object-based checkpoint\n\n  Returns:\n    Dictionary mapping tensor names to checkpoint keys.\n  '
    reader = py_checkpoint_reader.NewCheckpointReader(checkpoint_path)
    object_graph_string = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
    object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
    object_graph_proto.ParseFromString(object_graph_string)
    names_to_keys = {}
    for node in object_graph_proto.nodes:
        for attribute in node.attributes:
            names_to_keys[attribute.full_name] = attribute.checkpoint_key
    return names_to_keys

def saver_from_object_based_checkpoint(checkpoint_path, var_list=None, builder=None, names_to_keys=None, cached_saver=None):
    if False:
        return 10
    "Return a `Saver` which reads from an object-based checkpoint.\n\n  This function validates that all variables in the variables list are remapped\n  in the object-based checkpoint (or `names_to_keys` dict if provided). A\n  saver will be created with the list of remapped variables.\n\n  The `cached_saver` argument allows the user to pass in a previously created\n  saver, so multiple `saver.restore()` calls don't pollute the graph when graph\n  building. This assumes that keys are consistent, meaning that the\n    1) `checkpoint_path` checkpoint, and\n    2) checkpoint used to create the `cached_saver`\n  are the same type of object-based checkpoint. If this argument is set, this\n  function will simply validate that all variables have been remapped by the\n  checkpoint at `checkpoint_path`.\n\n  Note that in general, `tf.train.Checkpoint` should be used to restore/save an\n  object-based checkpoint.\n\n  Args:\n    checkpoint_path: string, path to object-based checkpoint\n    var_list: list of `Variables` that appear in the checkpoint. If `None`,\n      `var_list` will be set to all saveable objects.\n    builder: a `BaseSaverBuilder` instance. If `None`, a new `BulkSaverBuilder`\n      will be created.\n    names_to_keys: dict mapping string tensor names to checkpoint keys. If\n      `None`, this dict will be generated from the checkpoint file.\n    cached_saver: Cached `Saver` object with remapped variables.\n\n  Returns:\n    `Saver` with remapped variables for reading from an object-based checkpoint.\n\n  Raises:\n    ValueError if the checkpoint provided is not an object-based checkpoint.\n    NotFoundError: If one of the variables in `var_list` can not be found in the\n      checkpoint. This could mean the checkpoint or `names_to_keys` mapping is\n      missing the variable.\n  "
    if names_to_keys is None:
        try:
            names_to_keys = object_graph_key_mapping(checkpoint_path)
        except errors.NotFoundError:
            raise ValueError('Checkpoint in %s not an object-based checkpoint.' % checkpoint_path)
    if var_list is None:
        var_list = variables._all_saveable_objects()
    if builder is None:
        builder = BulkSaverBuilder()
    if not isinstance(var_list, dict):
        var_list = saveable_object_util.op_list_to_dict(var_list)
    saveables = saveable_object_util.validate_and_slice_inputs(var_list)
    current_names = set()
    for saveable in saveables:
        for spec in saveable.specs:
            current_names.add(spec.name)
    previous_names = set(names_to_keys.keys())
    missing_names = current_names - previous_names
    if missing_names:
        extra_names = previous_names - current_names
        intersecting_names = previous_names.intersection(current_names)
        raise errors.NotFoundError(None, None, message="\n\nExisting variables not in the checkpoint: %s\n\nVariables names when this checkpoint was written which don't exist now: %s\n\n(%d variable name(s) did match)\n\nCould not find some variables in the checkpoint (see names above). Saver was attempting to load an object-based checkpoint (saved using tf.train.Checkpoint or tf.keras.Model.save_weights) using variable names. If the checkpoint was written with eager execution enabled, it's possible that variable names have changed (for example missing a '_1' suffix). It's also possible that there are new variables which did not exist when the checkpoint was written. You can construct a Saver(var_list=...) with only the variables which previously existed, and if variable names have changed you may need to make this a dictionary with the old names as keys. If you're using an Estimator, you'll need to return a tf.train.Saver inside a tf.train.Scaffold from your model_fn." % (', '.join(sorted(missing_names)), ', '.join(sorted(extra_names)), len(intersecting_names)))
    for saveable in saveables:
        for spec in saveable.specs:
            spec.name = names_to_keys[spec.name]
    if cached_saver is None:
        return Saver(saveables)
    return cached_saver
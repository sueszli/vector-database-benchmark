"""SavedModel builder implementation."""
import functools
import os
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
_SAVE_BUILDER_LABEL = 'save_v1_builder'

@tf_export('__internal__.saved_model.SavedModelBuilder', v1=[])
class _SavedModelBuilder(object):
    """Builds the `SavedModel` protocol buffer and saves variables and assets.

  The `SavedModelBuilder` class provides the functionality to build a
  `SavedModel` protocol buffer. Specifically, this allows multiple meta
  graphs to be saved as part of a single language-neutral `SavedModel`,
  while sharing variables and assets.

  To build a SavedModel, the first meta graph must be saved with variables.
  Subsequent meta graphs will simply be saved with their graph definitions. If
  assets need to be saved and written or copied to disk, they can be provided
  when the meta graph def is added. If multiple meta graph defs are associated
  an asset of the same name, only the first version is retained.

  Each meta graph added to the SavedModel must be annotated with tags. The tags
  provide a means to identify the specific meta graph to load and restore, along
  with the shared set of variables and assets.

  Typical usage for the `SavedModelBuilder`:

  ```python
  ...
  builder = tf.compat.v1.saved_model.Builder(export_dir)

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph_and_variables(sess,
                                    ["foo-tag"],
                                    signature_def_map=foo_signatures,
                                    assets_list=foo_assets)
  ...

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph(["bar-tag", "baz-tag"])
  ...

  builder.save()
  ```

  Note: This function will only be available through the v1 compatibility
  library as tf.compat.v1.saved_model.builder.SavedModelBuilder or
  tf.compat.v1.saved_model.Builder. Tensorflow 2.0 will introduce a new
  object-based method of creating SavedModels.
  """

    def __init__(self, export_dir):
        if False:
            return 10
        self._saved_model = saved_model_pb2.SavedModel()
        self._saved_model.saved_model_schema_version = constants.SAVED_MODEL_SCHEMA_VERSION
        self._export_dir = export_dir
        if file_io.file_exists(export_dir):
            if file_io.list_directory(export_dir):
                raise AssertionError(f"Export directory {export_dir} already exists, and isn't empty. Please choose a different export directory, or delete all the contents of the specified directory.")
        else:
            file_io.recursive_create_dir(self._export_dir)
        self._has_saved_variables = False
        self._saved_asset_files = set()

    def _save_and_write_assets(self, meta_graph_def, assets_list=None):
        if False:
            print('Hello World!')
        'Saves asset to the meta graph and writes asset files to disk.\n\n    Args:\n      meta_graph_def: The meta graph def to which the assets will be added.\n      assets_list: The list where the asset paths are setup.\n    '
        write_fn = functools.partial(_add_asset_to_metagraph, meta_graph_def)
        asset_filename_map = _maybe_save_assets(write_fn, assets_list)
        if not asset_filename_map:
            tf_logging.info('No assets to write.')
            return
        copy_assets_to_destination_dir(asset_filename_map, self._export_dir, self._saved_asset_files)

    def _tag_and_add_meta_graph(self, meta_graph_def, tags, signature_def_map):
        if False:
            for i in range(10):
                print('nop')
        'Tags the meta graph def and adds it to the SavedModel.\n\n    Tags the meta graph def with the supplied tags, adds signature defs to it if\n    provided and appends the meta graph def to the SavedModel proto.\n\n    Args:\n      meta_graph_def: The meta graph def to add to the SavedModel.\n      tags: The set of tags to annotate the meta graph def with.\n      signature_def_map: The map of signature defs to be added to the meta graph\n        def.\n    '
        for tag in tags:
            meta_graph_def.meta_info_def.tags.append(tag)
        if signature_def_map is not None:
            for key in signature_def_map:
                meta_graph_def.signature_def[key].CopyFrom(signature_def_map[key])
        proto_meta_graph_def = self._saved_model.meta_graphs.add()
        proto_meta_graph_def.CopyFrom(meta_graph_def)

    def _validate_tensor_info(self, tensor_info):
        if False:
            i = 10
            return i + 15
        'Validates the `TensorInfo` proto.\n\n    Checks if the `encoding` (`name` or `coo_sparse` or `type_spec`) and\n    `dtype` fields exist and are non-empty.\n\n    Args:\n      tensor_info: `TensorInfo` protocol buffer to validate.\n\n    Raises:\n      AssertionError: If the `encoding` or `dtype` fields of the supplied\n          `TensorInfo` proto are not populated.\n    '
        if tensor_info is None:
            raise AssertionError('All TensorInfo protos used in the SignatureDefs must have the name and dtype fields set.')
        if tensor_info.WhichOneof('encoding') is None:
            raise AssertionError(f"Invalid `tensor_info`: {tensor_info}. All TensorInfo protos used in the SignatureDefs must have one of the 'encoding' fields (e.g., name or coo_sparse) set.")
        if tensor_info.WhichOneof('encoding') == 'composite_tensor':
            for component in tensor_info.composite_tensor.components:
                self._validate_tensor_info(component)
        elif tensor_info.dtype == types_pb2.DT_INVALID:
            raise AssertionError(f'Invalid `tensor_info`: {tensor_info}. All TensorInfo protos used in the SignatureDefs must have the dtype field set.')

    def _validate_signature_def_map(self, signature_def_map):
        if False:
            return 10
        'Validates the `SignatureDef` entries in the signature def map.\n\n    Validation of entries in the signature def map includes ensuring that the\n    `name` and `dtype` fields of the TensorInfo protos of the `inputs` and\n    `outputs` of each `SignatureDef` are populated. Also ensures that reserved\n    SignatureDef keys for the initialization and train ops are not used.\n\n    Args:\n      signature_def_map: The map of signature defs to be validated.\n\n    Raises:\n      AssertionError: If a TensorInfo is not valid.\n      KeyError: If a reserved signature key is used in the map.\n    '
        for signature_def_key in signature_def_map:
            signature_def = signature_def_map[signature_def_key]
            inputs = signature_def.inputs
            outputs = signature_def.outputs
            for inputs_key in inputs:
                self._validate_tensor_info(inputs[inputs_key])
            for outputs_key in outputs:
                self._validate_tensor_info(outputs[outputs_key])
        if constants.INIT_OP_SIGNATURE_KEY in signature_def_map:
            raise KeyError(f'SignatureDef map key "{constants.INIT_OP_SIGNATURE_KEY}" is reserved for initialization. Please use a different key.')
        if constants.TRAIN_OP_SIGNATURE_KEY in signature_def_map:
            raise KeyError(f'SignatureDef map key "{constants.TRAIN_OP_SIGNATURE_KEY}" is reserved for the train op. Please use a different key.')

    def _maybe_create_saver(self, saver=None):
        if False:
            while True:
                i = 10
        'Creates a sharded saver if one does not already exist.'
        if not saver:
            saver = tf_saver.Saver(variables._all_saveable_objects(), sharded=True, write_version=saver_pb2.SaverDef.V2, allow_empty=True)
        return saver

    def add_meta_graph(self, tags, signature_def_map=None, assets_list=None, clear_devices=False, init_op=None, train_op=None, saver=None):
        if False:
            for i in range(10):
                print('nop')
        'Adds the current meta graph to the SavedModel.\n\n    Creates a Saver in the current scope and uses the Saver to export the meta\n    graph def. Invoking this API requires the `add_meta_graph_and_variables()`\n    API to have been invoked before.\n\n    Args:\n      tags: The set of tags to annotate the meta graph def with.\n      signature_def_map: The map of signature defs to be added to the meta graph\n        def.\n      assets_list: Assets to be saved with SavedModel. Note\n          that this list should be a subset of the assets saved as part of\n          the first meta graph in the SavedModel.\n      clear_devices: Set to true if the device info on the default graph should\n        be cleared.\n      init_op: Op or group of ops to execute when the graph is loaded. Note\n          that when the init_op is specified it is run after the restore op at\n        load-time.\n      train_op: Op or group of opts that trains the model when run. This will\n        not be run automatically when the graph is loaded, instead saved in\n        a SignatureDef accessible through the exported MetaGraph.\n      saver: An instance of tf.compat.v1.train.Saver that will be used to export\n        the metagraph. If None, a sharded Saver that restores all variables will\n        be used.\n\n    Raises:\n      AssertionError: If the variables for the SavedModel have not been saved\n          yet, or if the graph already contains one or more legacy init ops.\n    '
        if not self._has_saved_variables:
            raise AssertionError('Graph state including variables and assets has not been saved yet. Please invoke `add_meta_graph_and_variables()` first.')
        signature_def_map = signature_def_map or {}
        self._validate_signature_def_map(signature_def_map)
        _add_op_to_signature_def_map(signature_def_map, init_op, constants.INIT_OP_SIGNATURE_KEY)
        _add_op_to_signature_def_map(signature_def_map, train_op, constants.TRAIN_OP_SIGNATURE_KEY)
        saver = self._maybe_create_saver(saver)
        meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices, strip_default_attrs=True)
        self._save_and_write_assets(meta_graph_def, assets_list)
        self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

    def add_meta_graph_and_variables(self, sess, tags, signature_def_map=None, assets_list=None, clear_devices=False, init_op=None, train_op=None, strip_default_attrs=False, saver=None):
        if False:
            while True:
                i = 10
        'Adds the current meta graph to the SavedModel and saves variables.\n\n    Creates a Saver to save the variables from the provided session. Exports the\n    corresponding meta graph def. This function assumes that the variables to be\n    saved have been initialized. For a given `SavedModelBuilder`, this API must\n    be called exactly once and for the first meta graph to save. For subsequent\n    meta graph defs to be added, the `add_meta_graph()` API must be used.\n\n    Args:\n      sess: The TensorFlow session from which to save the meta graph and\n        variables.\n      tags: The set of tags with which to save the meta graph.\n      signature_def_map: The map of signature def map to add to the meta graph\n        def.\n      assets_list: Assets to be saved with SavedModel.\n      clear_devices: Set to true if the device info on the default graph should\n        be cleared.\n      init_op: Op or group of ops to execute when the graph is loaded. Note\n          that when the init_op is specified it is run after the restore op at\n        load-time.\n      train_op: Op or group of ops that trains the model when run. This will\n        not be run automatically when the graph is loaded, instead saved in\n        a SignatureDef accessible through the exported MetaGraph.\n      strip_default_attrs: Boolean. If `True`, default-valued attributes will be\n        removed from the NodeDefs. For a detailed guide, see\n        [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).\n      saver: An instance of tf.compat.v1.train.Saver that will be used to export the\n        metagraph and save variables. If None, a sharded Saver that restores\n        all variables will be used.\n\n    '
        if self._has_saved_variables:
            raise AssertionError('Graph state including variables and assets has already been saved. Please invoke `add_meta_graph()` instead.')
        signature_def_map = signature_def_map or {}
        self._validate_signature_def_map(signature_def_map)
        _add_op_to_signature_def_map(signature_def_map, init_op, constants.INIT_OP_SIGNATURE_KEY)
        _add_op_to_signature_def_map(signature_def_map, train_op, constants.TRAIN_OP_SIGNATURE_KEY)
        path_helpers.get_or_create_variables_dir(self._export_dir)
        variables_path = path_helpers.get_variables_path(self._export_dir)
        saver = self._maybe_create_saver(saver)
        saver.save(sess, variables_path, write_meta_graph=False, write_state=False)
        meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)
        self._save_and_write_assets(meta_graph_def, assets_list)
        self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)
        self._has_saved_variables = True

    def save(self, as_text=False, experimental_image_format=False):
        if False:
            return 10
        'Writes a `SavedModel` protocol buffer to disk.\n\n    The function writes the SavedModel protocol buffer to the export directory\n    in a serialized format.\n\n    Args:\n      as_text: Writes the SavedModel protocol buffer in text format to disk.\n        Protocol buffers in text format are useful for debugging, but parsing\n        fails when it encounters an unknown field and so is not forward\n        compatible. This means changes to TensorFlow may prevent deployment of\n        new text format SavedModels to existing serving binaries. Do not deploy\n        `as_text` SavedModels to production.\n      experimental_image_format: Writes the SavedModel protobuf in the\n        experimental image format. See\n      https://www.tensorflow.org/api_docs/python/tf/saved_model/SaveOptions for\n        more details. This allows `SavedModelBuilder` to save models larger than\n        2 GiB.\n    \n    Raises:\n       RuntimeError: When trying to use `proto_splitter` but `proto_splitter` is\n         not imported. This check is here because `proto_splitter` is not \n         available in OSS at the moment. \n\n    Returns:\n      The path to which the SavedModel protocol buffer was written.\n    '
        metrics.IncrementWriteApi(_SAVE_BUILDER_LABEL)
        if not file_io.file_exists(self._export_dir):
            file_io.recursive_create_dir(self._export_dir)
        if as_text:
            path = file_io.join(compat.as_bytes(self._export_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
            file_io.write_string_to_file(path, str(self._saved_model))
        elif experimental_image_format:
            path = file_io.join(self._export_dir, constants.SAVED_MODEL_FILENAME_PREFIX)
            if locals().get('proto_splitter', globals().get('proto_splitter')) is None:
                raise RuntimeError('No proto_splitter is provided, cannot use experimental_image_format.')
            path = proto_splitter.SavedModelSplitter(self._saved_model).write(path)
        else:
            path = file_io.join(compat.as_bytes(self._export_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
            file_io.write_string_to_file(path, self._saved_model.SerializeToString(deterministic=True))
        tf_logging.info('SavedModel written to: %s', compat.as_text(path))
        metrics.IncrementWrite(write_version='1')
        return path

@tf_export(v1=['saved_model.Builder', 'saved_model.builder.SavedModelBuilder'])
class SavedModelBuilder(_SavedModelBuilder):
    __doc__ = _SavedModelBuilder.__doc__.replace('assets_list', 'assets_collection')

    def __init__(self, export_dir):
        if False:
            while True:
                i = 10
        super(SavedModelBuilder, self).__init__(export_dir=export_dir)

    def _add_collections(self, assets_collection, main_op, train_op):
        if False:
            print('Hello World!')
        'Add asset and op collections to be saved.'
        self._save_and_write_assets(assets_collection)
        self._maybe_add_main_op(main_op)
        self._add_train_op(train_op)

    def _save_and_write_assets(self, assets_collection_to_add=None):
        if False:
            print('Hello World!')
        'Saves asset to the meta graph and writes asset files to disk.\n\n    Args:\n      assets_collection_to_add: The collection where the asset paths are setup.\n    '
        asset_filename_map = _maybe_save_assets(_add_asset_to_collection, assets_collection_to_add)
        if not asset_filename_map:
            tf_logging.info('No assets to write.')
            return
        copy_assets_to_destination_dir(asset_filename_map, self._export_dir, self._saved_asset_files)

    def _maybe_add_main_op(self, main_op):
        if False:
            i = 10
            return i + 15
        'Adds main op to the SavedModel.\n\n    Args:\n      main_op: Main op to run as part of graph initialization. If None, no main\n        op will be added to the graph.\n\n    Raises:\n      TypeError: If the main op is provided but is not of type `Operation`.\n      ValueError: if the Graph already contains an init op.\n    '
        if main_op is None:
            return
        if not isinstance(main_op, ops.Operation):
            raise TypeError(f'Expected {main_op} to be an Operation but got type {type(main_op)} instead.')
        for init_op_key in (constants.MAIN_OP_KEY, constants.LEGACY_INIT_OP_KEY):
            if ops.get_collection(init_op_key):
                raise ValueError(f'Graph already contains one or more main ops under the collection {init_op_key}.')
        ops.add_to_collection(constants.MAIN_OP_KEY, main_op)

    def _add_train_op(self, train_op):
        if False:
            for i in range(10):
                print('nop')
        'Add train op to the SavedModel.\n\n    Note that this functionality is in development, and liable to be\n    moved elsewhere.\n\n    Args:\n      train_op: Op or group of ops that are used for training. These are stored\n        as a collection with key TRAIN_OP_KEY, but not executed.\n\n    Raises:\n      TypeError if Train op is not of type `Operation`.\n    '
        if train_op is not None:
            if not isinstance(train_op, tensor.Tensor) and (not isinstance(train_op, ops.Operation)):
                raise TypeError(f'`train_op` {train_op} needs to be a Tensor or Op.')
            ops.add_to_collection(constants.TRAIN_OP_KEY, train_op)

    @deprecated_args(None, 'Pass your op to the equivalent parameter main_op instead.', 'legacy_init_op')
    def add_meta_graph(self, tags, signature_def_map=None, assets_collection=None, legacy_init_op=None, clear_devices=False, main_op=None, strip_default_attrs=False, saver=None):
        if False:
            while True:
                i = 10
        if not self._has_saved_variables:
            raise AssertionError('Graph state including variables and assets has not been saved yet. Please invoke `add_meta_graph_and_variables()` first.')
        signature_def_map = signature_def_map or {}
        self._validate_signature_def_map(signature_def_map)
        main_op = main_op if main_op is not None else legacy_init_op
        self._add_collections(assets_collection, main_op, None)
        saver = self._maybe_create_saver(saver)
        meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)
        self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)

    @deprecated_args(None, 'Pass your op to the equivalent parameter main_op instead.', 'legacy_init_op')
    def add_meta_graph_and_variables(self, sess, tags, signature_def_map=None, assets_collection=None, legacy_init_op=None, clear_devices=False, main_op=None, strip_default_attrs=False, saver=None):
        if False:
            i = 10
            return i + 15
        if self._has_saved_variables:
            raise AssertionError('Graph state including variables and assets has already been saved. Please invoke `add_meta_graph()` instead.')
        signature_def_map = signature_def_map or {}
        self._validate_signature_def_map(signature_def_map)
        main_op = main_op or legacy_init_op
        self._add_collections(assets_collection, main_op, None)
        path_helpers.get_or_create_variables_dir(self._export_dir)
        variables_path = path_helpers.get_variables_path(self._export_dir)
        saver = self._maybe_create_saver(saver)
        saver.save(sess, variables_path, write_meta_graph=False, write_state=False)
        meta_graph_def = saver.export_meta_graph(clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)
        self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)
        self._has_saved_variables = True
    add_meta_graph.__doc__ = _SavedModelBuilder.add_meta_graph.__doc__.replace('assets_list', 'assets_collection')
    add_meta_graph_and_variables.__doc__ = _SavedModelBuilder.add_meta_graph_and_variables.__doc__.replace('assets_list', 'assets_collection')

def _maybe_save_assets(write_fn, assets_to_add=None):
    if False:
        return 10
    'Saves assets to the meta graph.\n\n  Args:\n    write_fn: A function callback that writes assets into meta graph.\n    assets_to_add: The list where the asset paths are setup.\n\n  Returns:\n    A dict of asset basenames for saving to the original full path to the asset.\n\n  Raises:\n    ValueError: Indicating an invalid filepath tensor.\n  '
    asset_filename_map = {}
    if assets_to_add is None:
        tf_logging.info('No assets to save.')
        return asset_filename_map
    for asset_tensor in assets_to_add:
        asset_source_filepath = _asset_path_from_tensor(asset_tensor)
        if not asset_source_filepath:
            raise ValueError(f'Asset filepath tensor {asset_tensor} in is invalid.')
        asset_filename = get_asset_filename_to_add(asset_source_filepath, asset_filename_map)
        write_fn(asset_filename, asset_tensor)
        asset_filename_map[asset_filename] = asset_source_filepath
    tf_logging.info('Assets added to graph.')
    return asset_filename_map

def get_asset_filename_to_add(asset_filepath, asset_filename_map):
    if False:
        while True:
            i = 10
    'Get a unique basename to add to the SavedModel if this file is unseen.\n\n  Assets come from users as full paths, and we save them out to the\n  SavedModel as basenames. In some cases, the basenames collide. Here,\n  we dedupe asset basenames by first checking if the file is the same,\n  and, if different, generate and return an index-suffixed basename\n  that can be used to add the asset to the SavedModel.\n\n  Args:\n    asset_filepath: the full path to the asset that is being saved\n    asset_filename_map: a dict of filenames used for saving the asset in\n      the SavedModel to full paths from which the filenames were derived.\n\n  Returns:\n    Uniquified filename string if the file is not a duplicate, or the original\n    filename if the file has already been seen and saved.\n  '
    asset_filename = os.path.basename(asset_filepath)
    if asset_filename not in asset_filename_map:
        return asset_filename
    other_asset_filepath = asset_filename_map[asset_filename]
    if other_asset_filepath == asset_filepath:
        return asset_filename
    if not file_io.filecmp(asset_filepath, other_asset_filepath):
        return _get_unique_asset_filename(asset_filename, asset_filename_map)
    return asset_filename

def _get_unique_asset_filename(asset_filename, asset_filename_map):
    if False:
        i = 10
        return i + 15
    i = 1
    unique_filename = asset_filename
    while unique_filename in asset_filename_map:
        unique_filename = compat.as_bytes('_').join([compat.as_bytes(asset_filename), compat.as_bytes(str(i))])
        i += 1
    return unique_filename

def _asset_path_from_tensor(path_tensor):
    if False:
        for i in range(10):
            print('nop')
    'Returns the filepath value stored in constant `path_tensor`.\n\n  Args:\n    path_tensor: Tensor of a file-path.\n\n  Returns:\n    The string value i.e. path of the tensor, if valid.\n\n  Raises:\n    TypeError if tensor does not match expected op type, dtype or value.\n  '
    if not isinstance(path_tensor, tensor.Tensor):
        raise TypeError(f'Asset path tensor {path_tensor} must be a Tensor.')
    if path_tensor.op.type != 'Const':
        raise TypeError(f'Asset path tensor {path_tensor} must be of type constant.Has type {path_tensor.op.type} instead.')
    if path_tensor.dtype != dtypes.string:
        raise TypeError(f'Asset path tensor {path_tensor}` must be of dtype string.Has type {path_tensor.dtype} instead.')
    str_values = path_tensor.op.get_attr('value').string_val
    if len(str_values) != 1:
        raise TypeError(f'Asset path tensor {path_tensor} must be a scalar.')
    return str_values[0]

def _add_asset_to_metagraph(meta_graph_def, asset_filename, asset_tensor):
    if False:
        while True:
            i = 10
    'Builds an asset proto and adds it to the meta graph def.\n\n  Args:\n    meta_graph_def: The meta graph def to which the asset will be added.\n    asset_filename: The filename of the asset to be added.\n    asset_tensor: The asset tensor used to populate the tensor info of the asset\n      proto.\n  '
    asset_proto = meta_graph_def.asset_file_def.add()
    asset_proto.filename = asset_filename
    asset_proto.tensor_info.name = asset_tensor.name

def copy_assets_to_destination_dir(asset_filename_map, destination_dir, saved_files=None):
    if False:
        for i in range(10):
            print('nop')
    'Copy all assets from source path to destination path.\n\n  Args:\n    asset_filename_map: a dict of filenames used for saving the asset in\n      the SavedModel to full paths from which the filenames were derived.\n    destination_dir: the destination directory that assets are stored in.\n    saved_files: a set of destination filepaths that have already been copied\n      and will be skipped\n  '
    if saved_files is None:
        saved_files = set()
    assets_destination_dir = path_helpers.get_or_create_assets_dir(destination_dir)
    for (asset_basename, asset_source_filepath) in asset_filename_map.items():
        asset_destination_filepath = file_io.join(compat.as_bytes(assets_destination_dir), compat.as_bytes(asset_basename))
        if file_io.file_exists(asset_source_filepath) and asset_source_filepath != asset_destination_filepath and (asset_destination_filepath not in saved_files):
            file_io.copy(asset_source_filepath, asset_destination_filepath, overwrite=True)
            saved_files.add(asset_destination_filepath)
    tf_logging.info('Assets written to: %s', compat.as_text(assets_destination_dir))

def _add_asset_to_collection(asset_filename, asset_tensor):
    if False:
        for i in range(10):
            print('nop')
    'Builds an asset proto and adds it to the asset collection of the graph.\n\n  Args:\n    asset_filename: The filename of the asset to be added.\n    asset_tensor: The asset tensor used to populate the tensor info of the\n        asset proto.\n  '
    asset_proto = meta_graph_pb2.AssetFileDef()
    asset_proto.filename = asset_filename
    asset_proto.tensor_info.name = asset_tensor.name
    asset_any_proto = Any()
    asset_any_proto.Pack(asset_proto)
    ops.add_to_collection(constants.ASSETS_KEY, asset_any_proto)

def _add_op_to_signature_def_map(signature_def_map, op, key):
    if False:
        print('Hello World!')
    if op is not None:
        signature_def_map[key] = signature_def_utils.op_signature_def(op, key)
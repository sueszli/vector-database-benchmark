"""Loader implementation for SavedModel with hermetic, language-neutral exports.
"""
import os
import sys
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
_LOADER_LABEL = 'loader'

def parse_saved_model_with_debug_info(export_dir):
    if False:
        for i in range(10):
            print('nop')
    'Reads the savedmodel as well as the graph debug info.\n\n  Args:\n    export_dir: Directory containing the SavedModel and GraphDebugInfo files.\n\n  Returns:\n    `SavedModel` and `GraphDebugInfo` protocol buffers.\n\n  Raises:\n    IOError: If the saved model file does not exist, or cannot be successfully\n    parsed. Missing graph debug info file is fine.\n  '
    saved_model = parse_saved_model(export_dir)
    debug_info_path = file_io.join(path_helpers.get_debug_dir(export_dir), constants.DEBUG_INFO_FILENAME_PB)
    debug_info = graph_debug_info_pb2.GraphDebugInfo()
    if file_io.file_exists(debug_info_path):
        with file_io.FileIO(debug_info_path, 'rb') as debug_file:
            try:
                debug_info.ParseFromString(debug_file.read())
            except message.DecodeError as e:
                raise IOError(f'Cannot parse file {debug_info_path}: {e}.')
    return (saved_model, debug_info)

@tf_export('__internal__.saved_model.parse_saved_model', v1=[])
def parse_saved_model(export_dir):
    if False:
        while True:
            i = 10
    'Reads the savedmodel.pb or savedmodel.pbtxt file containing `SavedModel`.\n\n  Args:\n    export_dir: String or Pathlike, path to the directory containing the\n    SavedModel file.\n\n  Returns:\n    A `SavedModel` protocol buffer.\n\n  Raises:\n    IOError: If the file does not exist, or cannot be successfully parsed.\n  '
    path_to_pbtxt = file_io.join(compat.as_bytes(compat.path_to_str(export_dir)), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
    path_to_pb = file_io.join(compat.as_bytes(compat.path_to_str(export_dir)), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
    path_to_cpb = file_io.join(compat.as_bytes(compat.path_to_str(export_dir)), compat.as_bytes(constants.SAVED_MODEL_FILENAME_CPB))
    saved_model = saved_model_pb2.SavedModel()
    if file_io.file_exists(path_to_pb):
        with file_io.FileIO(path_to_pb, 'rb') as f:
            file_content = f.read()
        try:
            saved_model.ParseFromString(file_content)
        except message.DecodeError as e:
            raise IOError(f'Cannot parse file {path_to_pb}: {str(e)}.') from e
    elif file_io.file_exists(path_to_pbtxt):
        with file_io.FileIO(path_to_pbtxt, 'rb') as f:
            file_content = f.read()
        try:
            text_format.Parse(file_content.decode('utf-8'), saved_model)
        except text_format.ParseError as e:
            raise IOError(f'Cannot parse file {path_to_pbtxt}: {str(e)}.') from e
    else:
        raise IOError(f'SavedModel file does not exist at: {export_dir}{os.path.sep}{{{constants.SAVED_MODEL_FILENAME_PBTXT}|{constants.SAVED_MODEL_FILENAME_PB}}}')
    return saved_model

def get_asset_tensors(export_dir, meta_graph_def_to_load, import_scope=None):
    if False:
        print('Hello World!')
    "Gets the asset tensors, if defined in the meta graph def to load.\n\n  Args:\n    export_dir: Directory where the SavedModel is located.\n    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.\n    import_scope: Optional `string` -- if specified, prepend this followed by\n        '/' to all returned asset tensor names.\n\n  Returns:\n    A dictionary of asset tensors, keyed by the name of the asset tensor. The\n    value in the map corresponds to the absolute path of the asset file.\n  "
    collection_def = meta_graph_def_to_load.collection_def
    asset_tensor_dict = {}
    asset_protos = []
    if meta_graph_def_to_load.asset_file_def:
        asset_protos = meta_graph_def_to_load.asset_file_def
    elif constants.ASSETS_KEY in collection_def:
        assets_any_proto = collection_def[constants.ASSETS_KEY].any_list.value
        for asset_any_proto in assets_any_proto:
            asset_proto = meta_graph_pb2.AssetFileDef()
            asset_any_proto.Unpack(asset_proto)
            asset_protos.append(asset_proto)
    assets_directory = file_io.join(compat.as_bytes(export_dir), compat.as_bytes(constants.ASSETS_DIRECTORY))
    for asset_proto in asset_protos:
        tensor_name = asset_proto.tensor_info.name
        if import_scope:
            tensor_name = '%s/%s' % (import_scope, tensor_name)
        asset_tensor_dict[tensor_name] = file_io.join(compat.as_bytes(assets_directory), compat.as_bytes(asset_proto.filename))
    return asset_tensor_dict

def _get_main_op_tensor(meta_graph_def_to_load, init_op_key=constants.MAIN_OP_KEY):
    if False:
        print('Hello World!')
    'Gets the main op tensor, if one exists.\n\n  Args:\n    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.\n    init_op_key: name of the collection to check; should be one of MAIN_OP_KEY\n      or the deprecated LEGACY_INIT_OP_KEY\n\n  Returns:\n    The main op tensor, if it exists and `None` otherwise.\n\n  Raises:\n    RuntimeError: If the collection def corresponding to the main op key has\n        other than exactly one tensor.\n  '
    collection_def = meta_graph_def_to_load.collection_def
    init_op = None
    if init_op_key in collection_def:
        init_op_list = collection_def[init_op_key].node_list.value
        if len(init_op_list) != 1:
            raise RuntimeError(f'Expected exactly one SavedModel init op. Found {len(init_op_list)}: {init_op_list}.')
        init_op = ops.get_collection(init_op_key)[0]
    return init_op

def _get_op_from_collection(meta_graph_def, op_key):
    if False:
        while True:
            i = 10
    return _get_main_op_tensor(meta_graph_def, op_key)

def _get_op_from_signature_def(meta_graph_def, op_signature_key, import_scope):
    if False:
        while True:
            i = 10
    "Retrieve op stored in the imported meta graph's signature def."
    if op_signature_key in meta_graph_def.signature_def:
        return signature_def_utils.load_op_from_signature_def(meta_graph_def.signature_def[op_signature_key], op_signature_key, import_scope)
    else:
        return None

def get_init_op(meta_graph_def, import_scope=None):
    if False:
        return 10
    return _get_op_from_signature_def(meta_graph_def, constants.INIT_OP_SIGNATURE_KEY, import_scope) or _get_op_from_collection(meta_graph_def, constants.MAIN_OP_KEY) or _get_op_from_collection(meta_graph_def, constants.LEGACY_INIT_OP_KEY)

def get_train_op(meta_graph_def, import_scope=None):
    if False:
        for i in range(10):
            print('nop')
    train_op = _get_op_from_signature_def(meta_graph_def, constants.TRAIN_OP_SIGNATURE_KEY, import_scope)
    if train_op is None:
        train_op = _get_op_from_collection(meta_graph_def, constants.TRAIN_OP_KEY)
    return train_op

@tf_export(v1=['saved_model.contains_saved_model', 'saved_model.maybe_saved_model_directory', 'saved_model.loader.maybe_saved_model_directory'])
@deprecation.deprecated_endpoints('saved_model.loader.maybe_saved_model_directory')
def maybe_saved_model_directory(export_dir):
    if False:
        i = 10
        return i + 15
    "Checks whether the provided export directory could contain a SavedModel.\n\n  Note that the method does not load any data by itself. If the method returns\n  `false`, the export directory definitely does not contain a SavedModel. If the\n  method returns `true`, the export directory may contain a SavedModel but\n  provides no guarantee that it can be loaded.\n\n  Args:\n    export_dir: Absolute string path to possible export location. For example,\n                '/my/foo/model'.\n\n  Returns:\n    True if the export directory contains SavedModel files, False otherwise.\n  "
    txt_path = file_io.join(export_dir, constants.SAVED_MODEL_FILENAME_PBTXT)
    pb_path = file_io.join(export_dir, constants.SAVED_MODEL_FILENAME_PB)
    return file_io.file_exists(txt_path) or file_io.file_exists(pb_path)

@tf_export('saved_model.contains_saved_model', v1=[])
def contains_saved_model(export_dir):
    if False:
        while True:
            i = 10
    "Checks whether the provided export directory could contain a SavedModel.\n\n  Note that the method does not load any data by itself. If the method returns\n  `false`, the export directory definitely does not contain a SavedModel. If the\n  method returns `true`, the export directory may contain a SavedModel but\n  provides no guarantee that it can be loaded.\n\n  Args:\n    export_dir: Absolute path to possible export location. For example,\n                '/my/foo/model'.\n\n  Returns:\n    True if the export directory contains SavedModel files, False otherwise.\n  "
    if isinstance(export_dir, os.PathLike):
        export_dir = os.fspath(export_dir)
    return maybe_saved_model_directory(export_dir)

@tf_export(v1=['saved_model.load', 'saved_model.loader.load'])
@deprecation.deprecated(None, 'Use `tf.saved_model.load` instead.')
def load(sess, tags, export_dir, import_scope=None, **saver_kwargs):
    if False:
        return 10
    'Loads the model from a SavedModel as specified by tags.\n\n  Args:\n    sess: The TensorFlow session to restore the variables.\n    tags: Set of string tags to identify the required MetaGraphDef. These should\n        correspond to the tags used when saving the variables using the\n        SavedModel `save()` API.\n    export_dir: Directory in which the SavedModel protocol buffer and variables\n        to be loaded are located.\n    import_scope: Optional `string` -- if specified, prepend this string\n        followed by \'/\' to all loaded tensor names. This scope is applied to\n        tensor instances loaded into the passed session, but it is *not* written\n        through to the static `MetaGraphDef` protocol buffer that is returned.\n    **saver_kwargs: Optional keyword arguments passed through to Saver.\n\n  Returns:\n    The `MetaGraphDef` protocol buffer loaded in the provided session. This\n    can be used to further extract signature-defs, collection-defs, etc.\n\n  Raises:\n    RuntimeError: MetaGraphDef associated with the tags cannot be found.\n\n  @compatibility(TF2)\n\n  `tf.compat.v1.saved_model.load` or `tf.compat.v1.saved_model.loader.load` is\n  not compatible with eager execution. Please use `tf.saved_model.load` instead\n  to load your model. You can refer to the [SavedModel guide]\n  (https://www.tensorflow.org/guide/saved_model) for more information as well as\n  "Importing SavedModels from TensorFlow 1.x" in the [`tf.saved_model.load`]\n  (https://www.tensorflow.org/api_docs/python/tf/saved_model/load) docstring.\n\n  #### How to Map Arguments\n\n  | TF1 Arg Name          | TF2 Arg Name    | Note                       |\n  | :-------------------- | :-------------- | :------------------------- |\n  | `sess`                | Not supported   | -                          |\n  | `tags`                | `tags`          | -                          |\n  | `export_dir`          | `export_dir`    | -                          |\n  | `import_scope`        | Not supported   | Name scopes are not needed.\n  :                       :                 : By default, variables are  :\n  :                       :                 : associated with the loaded :\n  :                       :                 : object and function names  :\n  :                       :                 : are deduped.               :\n  | `saver_kwargs`        | Not supported   | -                          |\n\n  #### Before & After Usage Example\n\n  Before:\n\n  ```\n  with tf.compat.v1.Session(graph=tf.Graph()) as sess:\n    tf.compat.v1.saved_model.loader.load(sess, ["foo-tag"], export_dir)\n  ```\n\n  After:\n\n  ```\n  model = tf.saved_model.load(export_dir, tags=["foo-tag"])\n  ```\n  @end_compatibility\n  '
    loader = SavedModelLoader(export_dir)
    return loader.load(sess, tags, import_scope, **saver_kwargs)

class SavedModelLoader(object):
    """Load graphs and restore variable values from a `SavedModel`."""

    def __init__(self, export_dir):
        if False:
            i = 10
            return i + 15
        'Creates a `SavedModelLoader`.\n\n    Args:\n      export_dir: Directory in which the SavedModel protocol buffer and\n        variables to be loaded are located.\n    '
        self._export_dir = export_dir
        self._variables_path = path_helpers.get_variables_path(export_dir)
        self._saved_model = parse_saved_model(export_dir)

    @property
    def export_dir(self):
        if False:
            i = 10
            return i + 15
        'Directory containing the SavedModel.'
        return self._export_dir

    @property
    def variables_path(self):
        if False:
            i = 10
            return i + 15
        'Path to variable checkpoint files.'
        return self._variables_path

    @property
    def saved_model(self):
        if False:
            print('Hello World!')
        'SavedModel object parsed from the export directory.'
        return self._saved_model

    def get_meta_graph_def_from_tags(self, tags):
        if False:
            for i in range(10):
                print('nop')
        'Return MetaGraphDef with the exact specified tags.\n\n    Args:\n      tags: A list or set of string tags that identify the MetaGraphDef.\n\n    Returns:\n      MetaGraphDef with the same tags.\n\n    Raises:\n      RuntimeError: if no metagraphs were found with the associated tags.\n    '
        found_match = False
        meta_graph_def_to_load = None
        available_tags = []
        for meta_graph_def in self._saved_model.meta_graphs:
            available_tags.append(set(meta_graph_def.meta_info_def.tags))
            if set(meta_graph_def.meta_info_def.tags) == set(tags):
                meta_graph_def_to_load = meta_graph_def
                found_match = True
                break
        if not found_match:
            raise RuntimeError(f"MetaGraphDef associated with tags {str(tags).strip('[]')} could not be found in SavedModel, with available tags '{available_tags}'. To inspect available tag-sets in the SavedModel, please use the SavedModel CLI: `saved_model_cli`.")
        return meta_graph_def_to_load

    def load_graph(self, graph, tags, import_scope=None, **saver_kwargs):
        if False:
            print('Hello World!')
        "Load ops and nodes from SavedModel MetaGraph into graph.\n\n    Args:\n      graph: tf.Graph object.\n      tags: a set of string tags identifying a MetaGraphDef.\n      import_scope: Optional `string` -- if specified, prepend this string\n        followed by '/' to all loaded tensor names. This scope is applied to\n        tensor instances loaded into the passed session, but it is *not* written\n        through to the static `MetaGraphDef` protocol buffer that is returned.\n      **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.\n\n    Returns:\n      A tuple of\n        * Saver defined by the MetaGraph, which can be used to restore the\n          variable values.\n        * List of `Operation`/`Tensor` objects returned from\n          `tf.import_graph_def` (may be `None`).\n    "
        meta_graph_def = self.get_meta_graph_def_from_tags(tags)
        if sys.byteorder == 'big':
            saved_model_utils.swap_function_tensor_content(meta_graph_def, 'little', 'big')
        with graph.as_default():
            return tf_saver._import_meta_graph_with_return_elements(meta_graph_def, import_scope=import_scope, **saver_kwargs)

    def restore_variables(self, sess, saver, import_scope=None):
        if False:
            while True:
                i = 10
        "Restore SavedModel variable values into the session.\n\n    Args:\n      sess: tf.compat.v1.Session to restore variable values.\n      saver: a tf.compat.v1.train.Saver object. Can be None if there are no\n        variables in graph. This may be the saver returned by the load_graph()\n        function, or a default `tf.compat.v1.train.Saver()`.\n      import_scope: Optional `string` -- if specified, prepend this string\n        followed by '/' to all loaded tensor names. This scope is applied to\n        tensor instances loaded into the passed session, but it is *not* written\n        through to the static `MetaGraphDef` protocol buffer that is returned.\n\n    Raises:\n      ValueError: if no saver was passed to the saver argument, and there are\n        variables in the graph.\n    "
        with sess.graph.as_default():
            if saver is None and (not variables._all_saveable_objects(scope=import_scope)):
                tf_logging.info('The specified SavedModel has no variables; no checkpoints were restored.')
            elif isinstance(saver, tf_saver.Saver):
                saver.restore(sess, self._variables_path)
            else:
                raise ValueError('No tf.train.Saver object was passed to the function `SavedModelLoader.restore_variables`. Since there are variables in the graph, a saver is required.')

    def run_init_ops(self, sess, tags, import_scope=None):
        if False:
            i = 10
            return i + 15
        "Run initialization ops defined in the `MetaGraphDef`.\n\n    Args:\n      sess: tf.compat.v1.Session to restore variable values.\n      tags: a set of string tags identifying a MetaGraphDef.\n      import_scope: Optional `string` -- if specified, prepend this string\n        followed by '/' to all loaded tensor names. This scope is applied to\n        tensor instances loaded into the passed session, but it is *not* written\n        through to the static `MetaGraphDef` protocol buffer that is returned.\n    "
        meta_graph_def = self.get_meta_graph_def_from_tags(tags)
        with sess.graph.as_default():
            asset_tensors_dictionary = get_asset_tensors(self._export_dir, meta_graph_def, import_scope=import_scope)
            init_op = get_init_op(meta_graph_def, import_scope)
            if init_op is not None:
                sess.run(fetches=[init_op], feed_dict=asset_tensors_dictionary)

    def load(self, sess, tags, import_scope=None, **saver_kwargs):
        if False:
            i = 10
            return i + 15
        "Load the MetaGraphDef graph and restore variable values into the session.\n\n    Args:\n      sess: tf.compat.v1.Session to restore variable values.\n      tags: a set of string tags identifying a MetaGraphDef.\n      import_scope: Optional `string` -- if specified, prepend this string\n        followed by '/' to all loaded tensor names. This scope is applied to\n        tensor instances loaded into the passed session, but it is *not* written\n        through to the static `MetaGraphDef` protocol buffer that is returned.\n      **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.\n\n    Returns:\n      `MetagraphDef` proto of the graph that was loaded.\n    "
        saved_model_proto = parse_saved_model(self._export_dir)
        metrics.IncrementReadApi(_LOADER_LABEL)
        with sess.graph.as_default():
            (saver, _) = self.load_graph(sess.graph, tags, import_scope, **saver_kwargs)
            self.restore_variables(sess, saver, import_scope)
            self.run_init_ops(sess, tags, import_scope)
        meta_graph_def = self.get_meta_graph_def_from_tags(tags)
        if len(saved_model_proto.meta_graphs) == 1 and saved_model_proto.meta_graphs[0].HasField('object_graph_def'):
            metrics.IncrementRead(write_version='2')
        else:
            metrics.IncrementRead(write_version='1')
        return meta_graph_def
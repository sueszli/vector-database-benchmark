"""SavedModel lib provides a way to read and write SavedModels.

This is an internal Hub utility and not part of the public API.
"""
import collections
import os
import re
from absl import logging
import tensorflow as tf
from tensorflow_hub import module_attachment_pb2
from tensorflow_hub import tf_utils
from google.protobuf import message
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
_SIGNATURE_COLLECTION = ('__saved_model_lib_signatures',)
_ATTACHMENT_COLLECTION_INTERNAL = ('__hub_module_attachments',)
ATTACHMENT_COLLECTION_SAVED = 'hub_module_attachments'

def get_variables_path(export_dir):
    if False:
        for i in range(10):
            print('nop')
    'Returns the path for storing variables checkpoints.'
    return os.path.join(tf.compat.as_bytes(export_dir), tf.compat.as_bytes(tf.compat.v1.saved_model.VARIABLES_DIRECTORY), tf.compat.as_bytes(tf.compat.v1.saved_model.VARIABLES_FILENAME))

def _get_assets_dir(export_dir):
    if False:
        for i in range(10):
            print('nop')
    return os.path.join(tf.compat.as_bytes(export_dir), tf.compat.as_bytes(tf.compat.v1.saved_model.ASSETS_DIRECTORY))

def _get_asset_filename(export_dir, asset_filename):
    if False:
        i = 10
        return i + 15
    assets_dir = _get_assets_dir(export_dir)
    filename = os.path.join(tf.compat.as_bytes(assets_dir), tf.compat.as_bytes(asset_filename))
    if not tf_utils.absolute_path(filename).startswith(tf_utils.absolute_path(assets_dir)):
        raise ValueError('Asset filename (%s) points outside assets_dir' % asset_filename)
    logging.debug('Asset filename: %s', filename)
    return filename

def _get_saved_model_proto_path(export_dir):
    if False:
        return 10
    return os.path.join(tf.compat.as_bytes(export_dir), tf.compat.as_bytes(tf.compat.v1.saved_model.SAVED_MODEL_FILENAME_PB))

def _get_node_name_from_tensor(tensor_name):
    if False:
        for i in range(10):
            print('nop')
    'tensor_name must have format node_name:output_number. Returns node_name.'
    result = re.match('([^:]*):\\d+$', tensor_name)
    if not result:
        raise ValueError('Unexpected format for tensor name. Expected node_name:output_number. Got %r' % tensor_name)
    return result.group(1)

def add_signature(key, inputs, outputs):
    if False:
        return 10
    'Adds a signature to current graph.\n\n  Args:\n    key: Signature key as a string.\n    inputs: Signature inputs as a map from string to Tensor or composite tensor\n      (such as SparseTensor or RaggedTensor).\n    outputs: Signature outputs as a map from string to Tensor or composite\n      tensor. (Recall that a Variable is not a Tensor, but Variable.value() is.)\n\n  Raises:\n    TypeError: if the arguments have the wrong types.\n  '
    _check_dict_maps_to_tensors_or_composite_tensors(inputs)
    _check_dict_maps_to_tensors_or_composite_tensors(outputs)
    input_info = {input_name: tf.compat.v1.saved_model.utils.build_tensor_info(tensor) for (input_name, tensor) in inputs.items()}
    output_info = {output_name: tf.compat.v1.saved_model.utils.build_tensor_info(tensor) for (output_name, tensor) in outputs.items()}
    signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(input_info, output_info)
    tf.compat.v1.add_to_collection(_SIGNATURE_COLLECTION, (key, signature))

def _check_dict_maps_to_tensors_or_composite_tensors(tensor_map):
    if False:
        return 10
    for (key, value) in tensor_map.items():
        if not (isinstance(value, tf.Tensor) or tf_utils.is_composite_tensor(value)):
            raise TypeError("Value for key '%s' should be a Tensor or CompositeTensor object, found %s." % (key, type(value)))

def _export_signatures(meta_graph):
    if False:
        while True:
            i = 10
    'Exports signatures from current graph into a MetaGraphDef.'
    named_signatures = tf.compat.v1.get_collection(_SIGNATURE_COLLECTION)
    if not named_signatures:
        raise ValueError('No signatures present. Please call hub.add_signature(...)at least once in the module_fn.')
    for (key, signature) in named_signatures:
        meta_graph.signature_def[key].CopyFrom(signature)

def attach_bytes(key, the_bytes):
    if False:
        while True:
            i = 10
    'Adds a ModuleAttachment to the current graph.\n\n  Args:\n    key: A string with the unique key of the attachment.\n    the_bytes: A bytes object with the serialized attachment.\n  '
    tf.compat.v1.add_to_collection(_ATTACHMENT_COLLECTION_INTERNAL, module_attachment_pb2.ModuleAttachment(key=key, value=the_bytes))

def _export_module_attachments(meta_graph):
    if False:
        for i in range(10):
            print('nop')
    'Exports ModuleAttachments from the current tf.Graph into `meta_graph`.'
    added_attachments = tf.compat.v1.get_collection(_ATTACHMENT_COLLECTION_INTERNAL)
    if not added_attachments:
        return
    unique_attachments = collections.OrderedDict(((attachment.key, attachment) for attachment in added_attachments))
    meta_graph.collection_def[ATTACHMENT_COLLECTION_SAVED].bytes_list.value[:] = [attachment.SerializeToString() for attachment in unique_attachments.values()]

def get_attached_bytes_map(meta_graph):
    if False:
        for i in range(10):
            print('nop')
    'Returns the dict of ModuleAttachments stored in `meta_graph`.\n\n  Args:\n    meta_graph: A MetaGraphDef, as built by SavedModelHandler.add_graph_copy()\n      from some graph.\n\n  Returns:\n    A dict, containing the `(key, bytes)` items passed to `attach_bytes()`\n    when the graph had been built.\n\n  Raises:\n    ValueError: if `meta-graph` is malformed.\n  '
    result = {}
    if ATTACHMENT_COLLECTION_SAVED not in meta_graph.collection_def:
        return result
    collection_def = meta_graph.collection_def[ATTACHMENT_COLLECTION_SAVED]
    if collection_def.WhichOneof('kind') != 'bytes_list':
        raise ValueError('Internal CollectionDef for attached messages has kind %s, expected bytes_list' % collection_def.WhichOneof('kind'))
    attachment = module_attachment_pb2.ModuleAttachment()
    for value in collection_def.bytes_list.value:
        attachment.ParseFromString(value)
        result[attachment.key] = attachment.value
    return result

def _export_tags(meta_graph, tags):
    if False:
        i = 10
        return i + 15
    'Exports tags into a MetaGraphDef.'
    if tags is not None:
        meta_graph.meta_info_def.tags.extend(tags)

def _check_asset_node_def(node_def):
    if False:
        while True:
            i = 10
    'Raises TypeError if `node_def` does not match the expectations.'
    if node_def.op != 'Const':
        raise TypeError('Asset node must be of type constant.')
    if tf.as_dtype(node_def.attr['dtype'].type) != tf.string:
        raise TypeError('Asset node must be of dtype string.')
    if len(node_def.attr['value'].tensor.string_val) != 1:
        raise TypeError('Asset node must be a scalar.')

def _merge_assets_key_collection(saved_model_proto, path):
    if False:
        return 10
    'Merges the ASSETS_KEY collection into the GraphDefs in saved_model_proto.\n\n  Removes the ASSETS_KEY collection from the GraphDefs in the SavedModel and\n  modifies nodes with the assets filenames to point to the assets in `path`.\n  After this transformation, the SavedModel GraphDefs can be used without\n  feeding asset tensors.\n\n  Args:\n    saved_model_proto: SavedModel proto to be modified.\n    path: path where the SavedModel is being loaded from.\n  '
    for meta_graph in saved_model_proto.meta_graphs:
        node_asset_map = {}
        if tf.compat.v1.saved_model.ASSETS_KEY in meta_graph.collection_def:
            assets_any_proto = meta_graph.collection_def[tf.compat.v1.saved_model.ASSETS_KEY].any_list.value
            for asset_any_proto in assets_any_proto:
                asset_proto = meta_graph_pb2.AssetFileDef()
                asset_any_proto.Unpack(asset_proto)
                asset_filename = _get_asset_filename(path, asset_proto.filename)
                node_asset_map[_get_node_name_from_tensor(asset_proto.tensor_info.name)] = asset_filename
            del meta_graph.collection_def[tf.compat.v1.saved_model.ASSETS_KEY]
        for node in meta_graph.graph_def.node:
            asset_filepath = node_asset_map.get(node.name)
            if asset_filepath:
                _check_asset_node_def(node)
                node.attr['value'].tensor.string_val[0] = asset_filepath

def _make_assets_key_collection(saved_model_proto, export_path):
    if False:
        while True:
            i = 10
    'Creates an ASSETS_KEY collection in the GraphDefs in saved_model_proto.\n\n  Adds an ASSETS_KEY collection to the GraphDefs in the SavedModel and returns\n  a map from original asset filename to filename when exporting the SavedModel\n  to `export_path`.\n\n  This is roughly the inverse operation of `_merge_assets_key_collection`.\n\n  Args:\n    saved_model_proto: SavedModel proto to be modified.\n    export_path: string with path where the saved_model_proto will be exported.\n\n  Returns:\n    A map from original asset filename to asset filename when exporting the\n    SavedModel to path.\n\n  Raises:\n    ValueError: on unsuported/unexpected SavedModel.\n  '
    asset_filenames = {}
    used_asset_filenames = set()

    def _make_asset_filename(original_filename):
        if False:
            return 10
        'Returns the asset filename to use for the file.'
        if original_filename in asset_filenames:
            return asset_filenames[original_filename]
        basename = os.path.basename(original_filename)
        suggestion = basename
        index = 0
        while suggestion in used_asset_filenames:
            suggestion = tf.compat.as_bytes(basename) + tf.compat.as_bytes(str(index))
            index += 1
        asset_filenames[original_filename] = suggestion
        used_asset_filenames.add(suggestion)
        return suggestion
    for meta_graph in saved_model_proto.meta_graphs:
        collection_def = meta_graph.collection_def.get(tf.compat.v1.GraphKeys.ASSET_FILEPATHS)
        if collection_def is None:
            continue
        if collection_def.WhichOneof('kind') != 'node_list':
            raise ValueError('MetaGraph collection ASSET_FILEPATHS is not a list of tensors.')
        for tensor in collection_def.node_list.value:
            if not tensor.endswith(':0'):
                raise ValueError('Unexpected tensor in ASSET_FILEPATHS collection.')
        asset_nodes = set([_get_node_name_from_tensor(tensor) for tensor in collection_def.node_list.value])
        tensor_filename_map = {}
        for node in meta_graph.graph_def.node:
            if node.name in asset_nodes:
                _check_asset_node_def(node)
                filename = node.attr['value'].tensor.string_val[0]
                tensor_filename_map[node.name + ':0'] = filename
                logging.debug('Found asset node %s pointing to %s', node.name, filename)
                node.attr['value'].tensor.string_val[0] = tf.compat.as_bytes('SAVEDMODEL-ASSET')
        if tensor_filename_map:
            assets_key_collection = meta_graph.collection_def[tf.compat.v1.saved_model.ASSETS_KEY]
            for (tensor, filename) in sorted(tensor_filename_map.items()):
                asset_proto = meta_graph_pb2.AssetFileDef()
                asset_proto.filename = _make_asset_filename(filename)
                asset_proto.tensor_info.name = tensor
                assets_key_collection.any_list.value.add().Pack(asset_proto)
    return {original_filename: _get_asset_filename(export_path, asset_filename) for (original_filename, asset_filename) in asset_filenames.items()}

class SavedModelHandler(object):
    """SavedModelHandler helps using SavedModel disk format.

  Note: This is a lower level interface than most users need. See SavedModel
  Builder/Loader API for an higher-level API centered around exporting and
  loading Sessions.

  A SavedModel disk format represents a collection of Graphs. To allow these
  graphs to be easy to manipulate, SavedModel extends Graphs with tags and
  signatures. Additionally it packages graphs, assets and variable checkpoints
  into an hermetic directory that can be moved around.

  This class hides the implementation details of SavedModels, in particular
  related with assets and signatures.

  SavedModelHandler deals with assets by:
    - Only supporting asset files as constant ops added to ASSET_FILEPATHS
      collection.
    - Creating a ASSETS_KEY collection only when writing meta_graphs to disk so
      they are never visible to user.
    - Baking the ASSETS_KEY collection in the graphs when loading from disk as
      to hide that the assets point to the packaged assets.

  SavedModelHandler deals with signatures by:
    - Providing `add_signature` API that allows to declare signatures directly
      on a graph.
    - That API is supported by a collection that is not serialized, but instead
      is converted into the right fields of MetaGraphDef when writing and
      loading a SavedModel from disk.
  """

    def __init__(self):
        if False:
            return 10
        self._proto = saved_model_pb2.SavedModel()

    def add_graph_copy(self, graph, tags=None):
        if False:
            i = 10
            return i + 15
        'Adds a copy of Graph with the specified set of tags.'
        with graph.as_default():
            meta_graph = tf.compat.v1.train.export_meta_graph(strip_default_attrs=True)
            _export_tags(meta_graph, tags)
            _export_signatures(meta_graph)
            _export_module_attachments(meta_graph)
        self._proto.meta_graphs.extend([meta_graph])

    def add_meta_graph_copy(self, meta_graph):
        if False:
            return 10
        self._proto.meta_graphs.extend([meta_graph])

    def get_meta_graph_copy(self, tags=None):
        if False:
            return 10
        'Returns a copy of a MetaGraph with the identical set of tags.'
        meta_graph = self.get_meta_graph(tags)
        copy = tf.compat.v1.MetaGraphDef()
        copy.CopyFrom(meta_graph)
        return copy

    @property
    def meta_graphs(self):
        if False:
            return 10
        return iter(self._proto.meta_graphs)

    def get_tags(self):
        if False:
            return 10
        'Returns a list of set of tags.'
        return sorted([frozenset(meta_graph.meta_info_def.tags) for meta_graph in self.meta_graphs])

    def get_attached_bytes_map(self, tags=None):
        if False:
            while True:
                i = 10
        return get_attached_bytes_map(self.get_meta_graph(tags))

    def export(self, path, variables_saver=None):
        if False:
            return 10
        'Exports to SavedModel directory.\n\n    Args:\n      path: path where to export the SavedModel to.\n      variables_saver: lambda that receives a directory path where to\n        export checkpoints of variables.\n    '
        proto = saved_model_pb2.SavedModel()
        proto.CopyFrom(self._proto)
        assets_map = _make_assets_key_collection(proto, path)
        self._save_all_assets(path, assets_map)
        self._save_variables(path, variables_saver)
        self._save_proto(path, proto)

    def get_meta_graph(self, tags=None):
        if False:
            i = 10
            return i + 15
        'Returns the matching MetaGraphDef or raises KeyError.'
        matches = [meta_graph for meta_graph in self.meta_graphs if set(meta_graph.meta_info_def.tags) == set(tags or [])]
        if not matches:
            raise KeyError('SavedModelHandler has no graph with tags: %r' % tags)
        if len(matches) != 1:
            raise KeyError('SavedModelHandler has multiple graphs with tags %r' % tags)
        return matches[0]

    def _save_all_assets(self, path, assets_map):
        if False:
            return 10
        assets_dir = _get_assets_dir(path)
        tf.compat.v1.gfile.MakeDirs(assets_dir)
        for (source, destination) in assets_map.items():
            tf.compat.v1.gfile.Copy(source, destination)

    def _save_variables(self, path, variables_saver):
        if False:
            return 10
        if variables_saver:
            variables_path = get_variables_path(path)
            variables_dir = os.path.dirname(variables_path)
            tf.compat.v1.gfile.MakeDirs(variables_dir)
            logging.debug('Variables saved in: %s', variables_path)
            variables_saver(variables_path)

    def _save_proto(self, path, proto):
        if False:
            i = 10
            return i + 15
        proto_path = _get_saved_model_proto_path(path)
        tf.compat.v1.gfile.MakeDirs(os.path.dirname(proto_path))
        logging.debug('SavedModel saved in: %s', proto_path)
        tf_utils.atomic_write_string_to_file(proto_path, proto.SerializeToString(), overwrite=True)

def _parse_saved_model(path):
    if False:
        i = 10
        return i + 15
    'Reads the savedmodel.pb file containing `SavedModel`.'
    path_to_pb = _get_saved_model_proto_path(path)
    file_content = tf.compat.v1.gfile.Open(path_to_pb, 'rb').read()
    saved_model = saved_model_pb2.SavedModel()
    try:
        saved_model.ParseFromString(file_content)
    except message.DecodeError as e:
        raise IOError('Cannot parse file %s: %s.' % (path_to_pb, str(e)))
    return saved_model

def load(path):
    if False:
        print('Hello World!')
    'Creates a SavedModelHandler from a SavedModel in `path`.'
    proto = _parse_saved_model(path)
    _merge_assets_key_collection(proto, path)
    handler = SavedModelHandler()
    handler._proto = proto
    return handler
"""Functions to convert SavedModel to frozen GraphDefs."""
from tensorflow.lite.python import util
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader

def get_meta_graph_def(saved_model_dir, tag_set):
    if False:
        while True:
            i = 10
    'Validate saved_model and extract MetaGraphDef.\n\n  Args:\n    saved_model_dir: saved_model path to convert.\n    tag_set: Set of tag(s) of the MetaGraphDef to load.\n\n  Returns:\n    The meta_graph_def used for tflite conversion.\n\n  Raises:\n    ValueError: No valid MetaGraphDef for given tag_set.\n  '
    with session.Session(graph=ops.Graph()) as sess:
        return loader.load(sess, tag_set, saved_model_dir)

def get_signature_def(meta_graph, signature_key):
    if False:
        i = 10
        return i + 15
    'Get the signature def from meta_graph with given signature_key.\n\n  Args:\n    meta_graph: meta_graph_def.\n    signature_key: signature_def in the meta_graph_def.\n\n  Returns:\n    The signature_def used for tflite conversion.\n\n  Raises:\n    ValueError: Given signature_key is not valid for this meta_graph.\n  '
    signature_def_map = meta_graph.signature_def
    signature_def_keys = set(signature_def_map.keys())
    logging.info('The given SavedModel MetaGraphDef contains SignatureDefs with the following keys: %s', signature_def_keys)
    if signature_key not in signature_def_keys:
        raise ValueError("No '{}' in the SavedModel's SignatureDefs. Possible values are '{}'.".format(signature_key, ','.join(signature_def_keys)))
    return signature_def_map[signature_key]

def get_inputs_outputs(signature_def):
    if False:
        while True:
            i = 10
    'Get inputs and outputs from SignatureDef.\n\n  Args:\n    signature_def: SignatureDef in the meta_graph_def for conversion.\n\n  Returns:\n    The inputs and outputs in the graph for conversion.\n  '
    inputs_tensor_info = signature_def.inputs
    outputs_tensor_info = signature_def.outputs

    def gather_names(tensor_info):
        if False:
            print('Hello World!')
        return [tensor_info[key].name for key in tensor_info]
    inputs = gather_names(inputs_tensor_info)
    outputs = gather_names(outputs_tensor_info)
    return (inputs, outputs)

def _get_tensors(graph, signature_def_tensor_names=None, user_tensor_names=None):
    if False:
        while True:
            i = 10
    'Gets the tensors associated with the tensor names.\n\n  Either signature_def_tensor_names or user_tensor_names should be provided. If\n  the user provides tensors, the tensors associated with the user provided\n  tensor names are provided. Otherwise, the tensors associated with the names in\n  the SignatureDef are provided.\n\n  Args:\n    graph: GraphDef representing graph.\n    signature_def_tensor_names: Tensor names stored in either the inputs or\n      outputs of a SignatureDef. (default None)\n    user_tensor_names: Tensor names provided by the user. (default None)\n\n  Returns:\n    List of tensors.\n\n  Raises:\n    ValueError:\n      signature_def_tensors and user_tensor_names are undefined or empty.\n      user_tensor_names are not valid.\n  '
    tensors = []
    if user_tensor_names:
        user_tensor_names = sorted(user_tensor_names)
        tensors = util.get_tensors_from_tensor_names(graph, user_tensor_names)
    elif signature_def_tensor_names:
        tensors = [graph.get_tensor_by_name(name) for name in sorted(signature_def_tensor_names)]
    else:
        raise ValueError('Specify either signature_def_tensor_names or user_tensor_names')
    return tensors

@convert_phase(Component.PREPARE_TF_MODEL, SubComponent.FREEZE_SAVED_MODEL)
def freeze_saved_model(saved_model_dir, input_arrays, input_shapes, output_arrays, tag_set, signature_key):
    if False:
        return 10
    'Converts a SavedModel to a frozen graph.\n\n  Args:\n    saved_model_dir: SavedModel directory to convert.\n    input_arrays: List of input tensors to freeze graph with. Uses input arrays\n      from SignatureDef when none are provided.\n    input_shapes: Dict of strings representing input tensor names to list of\n      integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).\n      Automatically determined when input shapes is None (e.g., {"foo" : None}).\n    output_arrays: List of output tensors to freeze graph with. Uses output\n      arrays from SignatureDef when none are provided.\n    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to\n      analyze. All tags in the tag set must be present.\n    signature_key: Key identifying SignatureDef containing inputs and outputs.\n\n  Returns:\n    frozen_graph_def: Frozen GraphDef.\n    in_tensors: List of input tensors for the graph.\n    out_tensors: List of output tensors for the graph.\n    graph: `Graph` object.\n\n  Raises:\n    ValueError:\n      SavedModel doesn\'t contain a MetaGraphDef identified by tag_set.\n      signature_key is not in the MetaGraphDef.\n      assets/ directory is in the MetaGraphDef.\n      input_shapes does not match the length of input_arrays.\n      input_arrays or output_arrays are not valid.\n  '
    meta_graph = get_meta_graph_def(saved_model_dir, tag_set)
    signature_def = get_signature_def(meta_graph, signature_key)
    (inputs, outputs) = get_inputs_outputs(signature_def)
    collection_def = meta_graph.collection_def
    if constants.ASSETS_KEY in collection_def:
        raise ValueError('SavedModels with assets/ directory are not supported.')
    graph = ops.Graph()
    with session.Session(graph=graph) as sess:
        loader.load(sess, meta_graph.meta_info_def.tags, saved_model_dir)
        in_tensors = _get_tensors(graph, inputs, input_arrays)
        out_tensors = _get_tensors(graph, outputs, output_arrays)
        util.set_tensor_shapes(in_tensors, input_shapes)
        frozen_graph_def = util.freeze_graph(sess, in_tensors, out_tensors)
        return (frozen_graph_def, in_tensors, out_tensors, sess.graph)
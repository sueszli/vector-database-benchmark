"""Command-line interface to inspect and execute a graph in a SavedModel.

For detailed usages and examples, please refer to:
https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel

"""
import argparse
import platform
import ast
import os
import re
from absl import app
from absl import flags
from absl.flags import argparse_flags
import numpy as np
from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import meta_graph as meta_graph_lib
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_aot_compile
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.tpu import tpu
from tensorflow.python.util.compat import collections_abc
_XLA_DEBUG_OPTIONS_URL = 'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/debug_options_flags.cc'
_OP_DENYLIST = set(['WriteFile', 'ReadFile', 'PrintV2'])
_SMCLI_DIR = flags.DEFINE_string(name='dir', default=None, help='Directory containing the SavedModel.')
_SMCLI_ALL = flags.DEFINE_bool(name='all', default=False, help='If set, outputs all available information in the given SavedModel.')
_SMCLI_TAG_SET = flags.DEFINE_string(name='tag_set', default=None, help='Comma-separated set of tags that identify variant graphs in the SavedModel.')
_SMCLI_SIGNATURE_DEF = flags.DEFINE_string(name='signature_def', default=None, help='Specifies a SignatureDef (by key) within the SavedModel to display input(s) and output(s) for.')
_SMCLI_LIST_OPS = flags.DEFINE_bool(name='list_ops', default=False, help='If set, will output ops used by a MetaGraphDef specified by tag_set.')
_SMCLI_INPUTS = flags.DEFINE_string(name='inputs', default='', help="Specifies input data files to pass to numpy.load(). Format should be '<input_key>=<filename>' or '<input_key>=<filename>[<variable_name>]', separated by ';'. File formats are limited to .npy, .npz, or pickle.")
_SMCLI_INPUT_EXPRS = flags.DEFINE_string(name='input_exprs', default='', help='Specifies Python literal expressions or numpy functions. Format should be "<input_key>=\'<python_expression>\'", separated by \';\'. Numpy can be accessed with \'np\'. Note that expressions are passed to literal_eval(), making this flag susceptible to code injection. Overrides duplicate input keys provided with the --inputs flag.')
_SMCLI_INPUT_EXAMPLES = flags.DEFINE_string(name='input_examples', default='', help="Specifies tf.train.Example objects as inputs. Format should be '<input_key>=[{{feature0:value_list,feature1:value_list}}]', where input keys are separated by ';'. Overrides duplicate input keys provided with the --inputs and --input_exprs flags.")
_SMCLI_OUTDIR = flags.DEFINE_string(name='outdir', default=None, help='If specified, writes CLI output to the given directory.')
_SMCLI_OVERWRITE = flags.DEFINE_bool(name='overwrite', default=False, help='If set, overwrites output file if it already exists.')
_SMCLI_TF_DEBUG = flags.DEFINE_bool(name='tf_debug', default=False, help='If set, uses the Tensorflow Debugger (tfdbg) to watch intermediate Tensors and runtime GraphDefs while running the SavedModel.')
_SMCLI_WORKER = flags.DEFINE_string(name='worker', default=None, help='If specified, runs the session on the given worker (bns or gRPC path).')
_SMCLI_INIT_TPU = flags.DEFINE_bool(name='init_tpu', default=False, help='If set, calls tpu.initialize_system() on the session. Should only be set if the specified worker is a TPU job.')
_SMCLI_USE_TFRT = flags.DEFINE_bool(name='use_tfrt', default=False, help='If set, runs a TFRT session, instead of a TF1 session.')
_SMCLI_OP_DENYLIST = flags.DEFINE_string(name='op_denylist', default=None, help='If specified, detects and reports the given ops. List of ops should be comma-separated. If not specified, the default list of ops is [WriteFile, ReadFile, PrintV2]. To specify an empty list, pass in the empty string.')
_SMCLI_OUTPUT_DIR = flags.DEFINE_string(name='output_dir', default=None, help='Output directory for the SavedModel.')
_SMCLI_MAX_WORKSPACE_SIZE_BYTES = flags.DEFINE_integer(name='max_workspace_size_bytes', default=2 << 20, help='The maximum temporary GPU memory which the TensorRT engine can use at execution time.')
_SMCLI_PRECISION_MODE = flags.DEFINE_enum(name='precision_mode', default='FP32', enum_values=['FP32', 'FP16', 'INT8'], help='TensorRT data precision. One of FP32, FP16, or INT8.')
_SMCLI_MINIMUM_SEGMENT_SIZE = flags.DEFINE_integer(name='minimum_segment_size', default=3, help='The minimum number of nodes required for a subgraph to be replaced in a TensorRT node.')
_SMCLI_CONVERT_TF1_MODEL = flags.DEFINE_bool(name='convert_tf1_model', default=False, help='Support TensorRT conversion for TF1 models.')
_SMCLI_OUTPUT_PREFIX = flags.DEFINE_string(name='output_prefix', default=None, help='Output directory + filename prefix for the resulting header(s) and object file(s).')
_SMCLI_SIGNATURE_DEF_KEY = flags.DEFINE_string(name='signature_def_key', default=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY, help='SavedModel SignatureDef key to use.')
_SMCLI_CHECKPOINT_PATH = flags.DEFINE_string(name='checkpoint_path', default=None, help='Custom checkpoint to use. Uses SavedModel variables by default.')
_SMCLI_VARIABLES_TO_FEED = flags.DEFINE_string(name='variables_to_feed', default='', help="Names of the variables that will be fed into the SavedModel graph. Pass in '' to feed no variables, 'all' to feed all variables, or a comma-separated list of variable names. Variables not fed will be frozen. *NOTE* Variables passed here must be set *by the user*. These variables will NOT be frozen, and their values will be uninitialized in the compiled object.")
if platform.machine() == 's390x':
    _SMCLI_TARGET_TRIPLE = flags.DEFINE_string(name='target_triple', default='', help="Triple identifying a target variation, containing information suchas processor architecture, vendor, operating system, and environment. Defaults to ''.")
else:
    _SMCLI_TARGET_TRIPLE = flags.DEFINE_string(name='target_triple', default='x86_64-pc-linux', help="Triple identifying a target variation, containing information such as processor architecture, vendor, operating system, and environment. Defaults to 'x86_64-pc-linux'.")
_SMCLI_TARGET_CPU = flags.DEFINE_string(name='target_cpu', default='', help="Target CPU name for LLVM during AOT compilation. Examples include 'x86_64', 'skylake', 'haswell', 'westmere', '' (unknown).")
_SMCLI_CPP_CLASS = flags.DEFINE_string(name='cpp_class', default=None, help='The name of the generated C++ class, wrapping the generated function. Format should be [[<optional_namespace>::],...]<class_name>, i.e. the same syntax as C++ for specifying a class. This class will be generated in the given namespace(s), or, if none are specified, the global namespace.')
_SMCLI_MULTITHREADING = flags.DEFINE_string(name='multithreading', default='False', help="Enable multithreading in the compiled computation. Note that with this flag enabled, the resulting object files may have external dependencies on multithreading libraries, such as 'nsync'.")
command_required_flags = {'show': ['dir'], 'run': ['dir', 'tag_set', 'signature_def'], 'scan': ['dir'], 'convert': ['dir', 'output_dir', 'tag_set'], 'freeze_model': ['dir', 'output_prefix', 'tag_set'], 'aot_compile_cpu': ['cpp_class']}

def _show_tag_sets(saved_model_dir):
    if False:
        return 10
    'Prints the tag-sets stored in SavedModel directory.\n\n  Prints all the tag-sets for MetaGraphs stored in SavedModel directory.\n\n  Args:\n    saved_model_dir: Directory containing the SavedModel to inspect.\n  '
    tag_sets = saved_model_utils.get_saved_model_tag_sets(saved_model_dir)
    print('The given SavedModel contains the following tag-sets:')
    for tag_set in sorted(tag_sets):
        print('%r' % ', '.join(sorted(tag_set)))

def _get_ops_in_metagraph(meta_graph_def):
    if False:
        return 10
    'Returns a set of the ops in the MetaGraph.\n\n  Returns the set of all the ops used in the MetaGraphDef indicated by the\n  tag_set stored in SavedModel directory.\n\n  Args:\n    meta_graph_def: MetaGraphDef to list the ops of.\n\n  Returns:\n    A set of ops.\n  '
    return set(meta_graph_lib.ops_used_by_graph_def(meta_graph_def.graph_def))

def _show_ops_in_metagraph_mgd(meta_graph_def):
    if False:
        i = 10
        return i + 15
    all_ops_set = _get_ops_in_metagraph(meta_graph_def)
    print('The MetaGraph with tag set %s contains the following ops:' % meta_graph_def.meta_info_def.tags, all_ops_set)

def _show_ops_in_metagraph(saved_model_dir, tag_set):
    if False:
        while True:
            i = 10
    "Prints the ops in the MetaGraph.\n\n  Prints all the ops used in the MetaGraphDef indicated by the tag_set stored in\n  SavedModel directory.\n\n  Args:\n    saved_model_dir: Directory containing the SavedModel to inspect.\n    tag_set: Group of tag(s) of the MetaGraphDef in string format, separated by\n      ','. For tag-set contains multiple tags, all tags must be passed in.\n  "
    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)
    _show_ops_in_metagraph_mgd(meta_graph_def)

def _show_signature_def_map_keys(saved_model_dir, tag_set):
    if False:
        for i in range(10):
            print('nop')
    "Prints the keys for each SignatureDef in the SignatureDef map.\n\n  Prints the list of SignatureDef keys from the SignatureDef map specified by\n  the given tag-set and SavedModel directory.\n\n  Args:\n    saved_model_dir: Directory containing the SavedModel to inspect.\n    tag_set: Group of tag(s) of the MetaGraphDef to get SignatureDef map from,\n        in string format, separated by ','. For tag-set contains multiple tags,\n        all tags must be passed in.\n  "
    signature_def_map = get_signature_def_map(saved_model_dir, tag_set)
    print('The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:')
    for signature_def_key in sorted(signature_def_map.keys()):
        print('SignatureDef key: "%s"' % signature_def_key)

def _get_inputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key):
    if False:
        print('Hello World!')
    'Gets TensorInfo for all inputs of the SignatureDef.\n\n  Returns a dictionary that maps each input key to its TensorInfo for the given\n  signature_def_key in the meta_graph_def\n\n  Args:\n    meta_graph_def: MetaGraphDef protocol buffer with the SignatureDef map to\n        look up SignatureDef key.\n    signature_def_key: A SignatureDef key string.\n\n  Returns:\n    A dictionary that maps input tensor keys to TensorInfos.\n\n  Raises:\n    ValueError if `signature_def_key` is not found in the MetaGraphDef.\n  '
    if signature_def_key not in meta_graph_def.signature_def:
        raise ValueError(f'''Could not find signature "{signature_def_key}". Please choose from: {', '.join(meta_graph_def.signature_def.keys())}''')
    return meta_graph_def.signature_def[signature_def_key].inputs

def _get_outputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key):
    if False:
        print('Hello World!')
    'Gets TensorInfos for all outputs of the SignatureDef.\n\n  Returns a dictionary that maps each output key to its TensorInfo for the given\n  signature_def_key in the meta_graph_def.\n\n  Args:\n    meta_graph_def: MetaGraphDef protocol buffer with the SignatureDefmap to\n    look up signature_def_key.\n    signature_def_key: A SignatureDef key string.\n\n  Returns:\n    A dictionary that maps output tensor keys to TensorInfos.\n  '
    return meta_graph_def.signature_def[signature_def_key].outputs

def _show_inputs_outputs_mgd(meta_graph_def, signature_def_key, indent):
    if False:
        print('Hello World!')
    'Prints input and output TensorInfos.\n\n  Prints the details of input and output TensorInfos for the SignatureDef mapped\n  by the given signature_def_key.\n\n  Args:\n    meta_graph_def: MetaGraphDef to inspect.\n    signature_def_key: A SignatureDef key string.\n    indent: How far (in increments of 2 spaces) to indent each line of output.\n  '
    inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)
    outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)
    indent_str = '  ' * indent

    def in_print(s):
        if False:
            i = 10
            return i + 15
        print(indent_str + s)
    in_print('The given SavedModel SignatureDef contains the following input(s):')
    for (input_key, input_tensor) in sorted(inputs_tensor_info.items()):
        in_print("  inputs['%s'] tensor_info:" % input_key)
        _print_tensor_info(input_tensor, indent + 1)
    in_print('The given SavedModel SignatureDef contains the following output(s):')
    for (output_key, output_tensor) in sorted(outputs_tensor_info.items()):
        in_print("  outputs['%s'] tensor_info:" % output_key)
        _print_tensor_info(output_tensor, indent + 1)
    in_print('Method name is: %s' % meta_graph_def.signature_def[signature_def_key].method_name)

def _show_inputs_outputs(saved_model_dir, tag_set, signature_def_key, indent=0):
    if False:
        return 10
    "Prints input and output TensorInfos.\n\n  Prints the details of input and output TensorInfos for the SignatureDef mapped\n  by the given signature_def_key.\n\n  Args:\n    saved_model_dir: Directory containing the SavedModel to inspect.\n    tag_set: Group of tag(s) of the MetaGraphDef, in string format, separated by\n      ','. For tag-set contains multiple tags, all tags must be passed in.\n    signature_def_key: A SignatureDef key string.\n    indent: How far (in increments of 2 spaces) to indent each line of output.\n  "
    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)
    _show_inputs_outputs_mgd(meta_graph_def, signature_def_key, indent)

def _show_defined_functions(saved_model_dir, meta_graphs):
    if False:
        return 10
    'Prints the callable concrete and polymorphic functions of the Saved Model.\n\n  Args:\n    saved_model_dir: Directory containing the SavedModel to inspect.\n    meta_graphs: Already-extracted MetaGraphDef of the SavedModel.\n  '
    has_object_graph_def = False
    for meta_graph_def in meta_graphs:
        has_object_graph_def |= meta_graph_def.HasField('object_graph_def')
    if not has_object_graph_def:
        return
    with ops_lib.Graph().as_default():
        trackable_object = load.load(saved_model_dir)
    print('\nConcrete Functions:', end='')
    children = list(save._AugmentedGraphView(trackable_object).list_children(trackable_object))
    children = sorted(children, key=lambda x: x.name)
    for (name, child) in children:
        concrete_functions = []
        if isinstance(child, defun.ConcreteFunction):
            concrete_functions.append(child)
        elif isinstance(child, def_function.Function):
            concrete_functions.extend(child._list_all_concrete_functions_for_serialization())
        else:
            continue
        print("\n  Function Name: '%s'" % name)
        concrete_functions = sorted(concrete_functions, key=lambda x: x.name)
        for (index, concrete_function) in enumerate(concrete_functions, 1):
            (args, kwargs) = (None, None)
            if concrete_function.structured_input_signature:
                (args, kwargs) = concrete_function.structured_input_signature
            elif concrete_function._arg_keywords:
                args = concrete_function._arg_keywords
            if args:
                print('    Option #%d' % index)
                print('      Callable with:')
                _print_args(args, indent=4)
            if kwargs:
                _print_args(kwargs, 'Named Argument', indent=4)

def _print_args(arguments, argument_type='Argument', indent=0):
    if False:
        print('Hello World!')
    'Formats and prints the argument of the concrete functions defined in the model.\n\n  Args:\n    arguments: Arguments to format print.\n    argument_type: Type of arguments.\n    indent: How far (in increments of 2 spaces) to indent each line of\n     output.\n  '
    indent_str = '  ' * indent

    def _maybe_add_quotes(value):
        if False:
            print('Hello World!')
        is_quotes = "'" * isinstance(value, str)
        return is_quotes + str(value) + is_quotes

    def in_print(s, end='\n'):
        if False:
            i = 10
            return i + 15
        print(indent_str + s, end=end)
    for (index, element) in enumerate(arguments, 1):
        if indent == 4:
            in_print('%s #%d' % (argument_type, index))
        if isinstance(element, str):
            in_print('  %s' % element)
        elif isinstance(element, tensor_spec.TensorSpec):
            print((indent + 1) * '  ' + '%s: %s' % (element.name, repr(element)))
        elif isinstance(element, collections_abc.Iterable) and (not isinstance(element, dict)):
            in_print('  DType: %s' % type(element).__name__)
            in_print('  Value: [', end='')
            for value in element:
                print('%s' % _maybe_add_quotes(value), end=', ')
            print('\x08\x08]')
        elif isinstance(element, dict):
            in_print('  DType: %s' % type(element).__name__)
            in_print('  Value: {', end='')
            for (key, value) in element.items():
                print("'%s': %s" % (str(key), _maybe_add_quotes(value)), end=', ')
            print('\x08\x08}')
        else:
            in_print('  DType: %s' % type(element).__name__)
            in_print('  Value: %s' % str(element))

def _print_tensor_info(tensor_info, indent=0):
    if False:
        i = 10
        return i + 15
    'Prints details of the given tensor_info.\n\n  Args:\n    tensor_info: TensorInfo object to be printed.\n    indent: How far (in increments of 2 spaces) to indent each line output\n  '
    indent_str = '  ' * indent

    def in_print(s):
        if False:
            for i in range(10):
                print('nop')
        print(indent_str + s)
    in_print('    dtype: ' + {value: key for (key, value) in types_pb2.DataType.items()}[tensor_info.dtype])
    if tensor_info.tensor_shape.unknown_rank:
        shape = 'unknown_rank'
    else:
        dims = [str(dim.size) for dim in tensor_info.tensor_shape.dim]
        shape = ', '.join(dims)
        shape = '(' + shape + ')'
    in_print('    shape: ' + shape)
    in_print('    name: ' + tensor_info.name)

def _show_all(saved_model_dir):
    if False:
        i = 10
        return i + 15
    'Prints tag-set, ops, SignatureDef, and Inputs/Outputs of SavedModel.\n\n  Prints all tag-set, ops, SignatureDef and Inputs/Outputs information stored in\n  SavedModel directory.\n\n  Args:\n    saved_model_dir: Directory containing the SavedModel to inspect.\n  '
    saved_model = saved_model_utils.read_saved_model(saved_model_dir)
    for meta_graph_def in sorted(saved_model.meta_graphs, key=lambda meta_graph_def: list(meta_graph_def.meta_info_def.tags)):
        tag_set = meta_graph_def.meta_info_def.tags
        print("\nMetaGraphDef with tag-set: '%s' contains the following SignatureDefs:" % ', '.join(tag_set))
        tag_set = ','.join(tag_set)
        signature_def_map = meta_graph_def.signature_def
        for signature_def_key in sorted(signature_def_map.keys()):
            print("\nsignature_def['" + signature_def_key + "']:")
            _show_inputs_outputs_mgd(meta_graph_def, signature_def_key, indent=1)
        _show_ops_in_metagraph_mgd(meta_graph_def)
    _show_defined_functions(saved_model_dir, saved_model.meta_graphs)

def get_meta_graph_def(saved_model_dir, tag_set):
    if False:
        print('Hello World!')
    "DEPRECATED: Use saved_model_utils.get_meta_graph_def instead.\n\n  Gets MetaGraphDef from SavedModel. Returns the MetaGraphDef for the given\n  tag-set and SavedModel directory.\n\n  Args:\n    saved_model_dir: Directory containing the SavedModel to inspect or execute.\n    tag_set: Group of tag(s) of the MetaGraphDef to load, in string format,\n        separated by ','. For tag-set contains multiple tags, all tags must be\n        passed in.\n\n  Raises:\n    RuntimeError: An error when the given tag-set does not exist in the\n        SavedModel.\n\n  Returns:\n    A MetaGraphDef corresponding to the tag-set.\n  "
    return saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)

def get_signature_def_map(saved_model_dir, tag_set):
    if False:
        for i in range(10):
            print('nop')
    "Gets SignatureDef map from a MetaGraphDef in a SavedModel.\n\n  Returns the SignatureDef map for the given tag-set in the SavedModel\n  directory.\n\n  Args:\n    saved_model_dir: Directory containing the SavedModel to inspect or execute.\n    tag_set: Group of tag(s) of the MetaGraphDef with the SignatureDef map, in\n        string format, separated by ','. For tag-set contains multiple tags, all\n        tags must be passed in.\n\n  Returns:\n    A SignatureDef map that maps from string keys to SignatureDefs.\n  "
    meta_graph = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)
    return meta_graph.signature_def

def _get_op_denylist_set(op_denylist):
    if False:
        while True:
            i = 10
    set_of_denylisted_ops = set([op for op in op_denylist.split(',') if op])
    return set_of_denylisted_ops

def scan_meta_graph_def(meta_graph_def, op_denylist):
    if False:
        while True:
            i = 10
    'Scans meta_graph_def and reports if there are ops on denylist.\n\n  Print ops if they are on denylist, or print success if no denylisted ops\n  found.\n\n  Args:\n    meta_graph_def: MetaGraphDef protocol buffer.\n    op_denylist: set of ops to scan for.\n  '
    ops_in_metagraph = set(meta_graph_lib.ops_used_by_graph_def(meta_graph_def.graph_def))
    denylisted_ops = op_denylist & ops_in_metagraph
    if denylisted_ops:
        print('MetaGraph with tag set %s contains the following denylisted ops:' % meta_graph_def.meta_info_def.tags, denylisted_ops)
    else:
        print('MetaGraph with tag set %s does not contain the default denylisted ops:' % meta_graph_def.meta_info_def.tags, op_denylist)

def run_saved_model_with_feed_dict(saved_model_dir, tag_set, signature_def_key, input_tensor_key_feed_dict, outdir, overwrite_flag, worker=None, init_tpu=False, use_tfrt=False, tf_debug=False):
    if False:
        i = 10
        return i + 15
    "Runs SavedModel and fetch all outputs.\n\n  Runs the input dictionary through the MetaGraphDef within a SavedModel\n  specified by the given tag_set and SignatureDef. Also save the outputs to file\n  if outdir is not None.\n\n  Args:\n    saved_model_dir: Directory containing the SavedModel to execute.\n    tag_set: Group of tag(s) of the MetaGraphDef with the SignatureDef map, in\n        string format, separated by ','. For tag-set contains multiple tags, all\n        tags must be passed in.\n    signature_def_key: A SignatureDef key string.\n    input_tensor_key_feed_dict: A dictionary maps input keys to numpy ndarrays.\n    outdir: A directory to save the outputs to. If the directory doesn't exist,\n        it will be created.\n    overwrite_flag: A boolean flag to allow overwrite output file if file with\n        the same name exists.\n    worker: If provided, the session will be run on the worker.  Valid worker\n        specification is a bns or gRPC path.\n    init_tpu: If true, the TPU system will be initialized after the session\n        is created.\n    use_tfrt: If true, TFRT session will be used.\n    tf_debug: A boolean flag to use TensorFlow Debugger (TFDBG) to observe the\n        intermediate Tensor values and runtime GraphDefs while running the\n        SavedModel.\n\n  Raises:\n    ValueError: When any of the input tensor keys is not valid.\n    RuntimeError: An error when output file already exists and overwrite is not\n    enabled.\n  "
    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)
    inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)
    for input_key_name in input_tensor_key_feed_dict.keys():
        if input_key_name not in inputs_tensor_info:
            raise ValueError('"%s" is not a valid input key. Please choose from %s, or use --show option.' % (input_key_name, '"' + '", "'.join(inputs_tensor_info.keys()) + '"'))
    inputs_feed_dict = {inputs_tensor_info[key].name: tensor for (key, tensor) in input_tensor_key_feed_dict.items()}
    outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)
    output_tensor_keys_sorted = sorted(outputs_tensor_info.keys())
    output_tensor_names_sorted = [outputs_tensor_info[tensor_key].name for tensor_key in output_tensor_keys_sorted]
    config = None
    if use_tfrt:
        logging.info('Using TFRT session.')
        config = config_pb2.ConfigProto(experimental=config_pb2.ConfigProto.Experimental(use_tfrt=True))
    with session.Session(worker, graph=ops_lib.Graph(), config=config) as sess:
        if init_tpu:
            print('Initializing TPU System ...')
            sess.run(tpu.initialize_system())
        loader.load(sess, tag_set.split(','), saved_model_dir)
        if tf_debug:
            sess = local_cli_wrapper.LocalCLIDebugWrapperSession(sess)
        outputs = sess.run(output_tensor_names_sorted, feed_dict=inputs_feed_dict)
        for (i, output) in enumerate(outputs):
            output_tensor_key = output_tensor_keys_sorted[i]
            print('Result for output key %s:\n%s' % (output_tensor_key, output))
            if outdir:
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                output_full_path = os.path.join(outdir, output_tensor_key + '.npy')
                if not overwrite_flag and os.path.exists(output_full_path):
                    raise RuntimeError('Output file %s already exists. Add "--overwrite" to overwrite the existing output files.' % output_full_path)
                np.save(output_full_path, output)
                print('Output %s is saved to %s' % (output_tensor_key, output_full_path))

def preprocess_inputs_arg_string(inputs_str):
    if False:
        i = 10
        return i + 15
    'Parses input arg into dictionary that maps input to file/variable tuple.\n\n  Parses input string in the format of, for example,\n  "input1=filename1[variable_name1],input2=filename2" into a\n  dictionary looks like\n  {\'input_key1\': (filename1, variable_name1),\n   \'input_key2\': (file2, None)}\n  , which maps input keys to a tuple of file name and variable name(None if\n  empty).\n\n  Args:\n    inputs_str: A string that specified where to load inputs. Inputs are\n    separated by semicolons.\n        * For each input key:\n            \'<input_key>=<filename>\' or\n            \'<input_key>=<filename>[<variable_name>]\'\n        * The optional \'variable_name\' key will be set to None if not specified.\n\n  Returns:\n    A dictionary that maps input keys to a tuple of file name and variable name.\n\n  Raises:\n    RuntimeError: An error when the given input string is in a bad format.\n  '
    input_dict = {}
    inputs_raw = inputs_str.split(';')
    for input_raw in filter(bool, inputs_raw):
        match = re.match('([^=]+)=([^\\[\\]]+)\\[([^\\[\\]]+)\\]$', input_raw)
        if match:
            input_dict[match.group(1)] = (match.group(2), match.group(3))
        else:
            match = re.match('([^=]+)=([^\\[\\]]+)$', input_raw)
            if match:
                input_dict[match.group(1)] = (match.group(2), None)
            else:
                raise RuntimeError('--inputs "%s" format is incorrect. Please follow"<input_key>=<filename>", or"<input_key>=<filename>[<variable_name>]"' % input_raw)
    return input_dict

def preprocess_input_exprs_arg_string(input_exprs_str, safe=True):
    if False:
        i = 10
        return i + 15
    "Parses input arg into dictionary that maps input key to python expression.\n\n  Parses input string in the format of 'input_key=<python expression>' into a\n  dictionary that maps each input_key to its python expression.\n\n  Args:\n    input_exprs_str: A string that specifies python expression for input keys.\n      Each input is separated by semicolon. For each input key:\n        'input_key=<python expression>'\n    safe: Whether to evaluate the python expression as literals or allow\n      arbitrary calls (e.g. numpy usage).\n\n  Returns:\n    A dictionary that maps input keys to their values.\n\n  Raises:\n    RuntimeError: An error when the given input string is in a bad format.\n  "
    input_dict = {}
    for input_raw in filter(bool, input_exprs_str.split(';')):
        if '=' not in input_exprs_str:
            raise RuntimeError('--input_exprs "%s" format is incorrect. Please follow"<input_key>=<python expression>"' % input_exprs_str)
        (input_key, expr) = input_raw.split('=', 1)
        if safe:
            try:
                input_dict[input_key] = ast.literal_eval(expr)
            except Exception as exc:
                raise RuntimeError(f'Expression "{expr}" is not a valid python literal.') from exc
        else:
            input_dict[input_key] = eval(expr)
    return input_dict

def preprocess_input_examples_arg_string(input_examples_str):
    if False:
        while True:
            i = 10
    "Parses input into dict that maps input keys to lists of tf.Example.\n\n  Parses input string in the format of 'input_key1=[{feature_name:\n  feature_list}];input_key2=[{feature_name:feature_list}];' into a dictionary\n  that maps each input_key to its list of serialized tf.Example.\n\n  Args:\n    input_examples_str: A string that specifies a list of dictionaries of\n    feature_names and their feature_lists for each input.\n    Each input is separated by semicolon. For each input key:\n      'input=[{feature_name1: feature_list1, feature_name2:feature_list2}]'\n      items in feature_list can be the type of float, int, long or str.\n\n  Returns:\n    A dictionary that maps input keys to lists of serialized tf.Example.\n\n  Raises:\n    ValueError: An error when the given tf.Example is not a list.\n  "
    input_dict = preprocess_input_exprs_arg_string(input_examples_str)
    for (input_key, example_list) in input_dict.items():
        if not isinstance(example_list, list):
            raise ValueError('tf.Example input must be a list of dictionaries, but "%s" is %s' % (example_list, type(example_list)))
        input_dict[input_key] = [_create_example_string(example) for example in example_list]
    return input_dict

def _create_example_string(example_dict):
    if False:
        print('Hello World!')
    'Create a serialized tf.example from feature dictionary.'
    example = example_pb2.Example()
    for (feature_name, feature_list) in example_dict.items():
        if not isinstance(feature_list, list):
            raise ValueError('feature value must be a list, but %s: "%s" is %s' % (feature_name, feature_list, type(feature_list)))
        if isinstance(feature_list[0], float):
            example.features.feature[feature_name].float_list.value.extend(feature_list)
        elif isinstance(feature_list[0], str):
            example.features.feature[feature_name].bytes_list.value.extend([f.encode('utf8') for f in feature_list])
        elif isinstance(feature_list[0], bytes):
            example.features.feature[feature_name].bytes_list.value.extend(feature_list)
        elif isinstance(feature_list[0], int):
            example.features.feature[feature_name].int64_list.value.extend(feature_list)
        else:
            raise ValueError('Type %s for value %s is not supported for tf.train.Feature.' % (type(feature_list[0]), feature_list[0]))
    return example.SerializeToString()

def load_inputs_from_input_arg_string(inputs_str, input_exprs_str, input_examples_str):
    if False:
        i = 10
        return i + 15
    'Parses input arg strings and create inputs feed_dict.\n\n  Parses \'--inputs\' string for inputs to be loaded from file, and parses\n  \'--input_exprs\' string for inputs to be evaluated from python expression.\n  \'--input_examples\' string for inputs to be created from tf.example feature\n  dictionary list.\n\n  Args:\n    inputs_str: A string that specified where to load inputs. Each input is\n        separated by semicolon.\n        * For each input key:\n            \'<input_key>=<filename>\' or\n            \'<input_key>=<filename>[<variable_name>]\'\n        * The optional \'variable_name\' key will be set to None if not specified.\n        * File specified by \'filename\' will be loaded using numpy.load. Inputs\n            can be loaded from only .npy, .npz or pickle files.\n        * The "[variable_name]" key is optional depending on the input file type\n            as descripted in more details below.\n        When loading from a npy file, which always contains a numpy ndarray, the\n        content will be directly assigned to the specified input tensor. If a\n        variable_name is specified, it will be ignored and a warning will be\n        issued.\n        When loading from a npz zip file, user can specify which variable within\n        the zip file to load for the input tensor inside the square brackets. If\n        nothing is specified, this function will check that only one file is\n        included in the zip and load it for the specified input tensor.\n        When loading from a pickle file, if no variable_name is specified in the\n        square brackets, whatever that is inside the pickle file will be passed\n        to the specified input tensor, else SavedModel CLI will assume a\n        dictionary is stored in the pickle file and the value corresponding to\n        the variable_name will be used.\n    input_exprs_str: A string that specifies python expressions for inputs.\n        * In the format of: \'<input_key>=<python expression>\'.\n        * numpy module is available as np.\n    input_examples_str: A string that specifies tf.Example with dictionary.\n        * In the format of: \'<input_key>=<[{feature:value list}]>\'\n\n  Returns:\n    A dictionary that maps input tensor keys to numpy ndarrays.\n\n  Raises:\n    RuntimeError: An error when a key is specified, but the input file contains\n        multiple numpy ndarrays, none of which matches the given key.\n    RuntimeError: An error when no key is specified, but the input file contains\n        more than one numpy ndarrays.\n  '
    tensor_key_feed_dict = {}
    inputs = preprocess_inputs_arg_string(inputs_str)
    input_exprs = preprocess_input_exprs_arg_string(input_exprs_str)
    input_examples = preprocess_input_examples_arg_string(input_examples_str)
    for (input_tensor_key, (filename, variable_name)) in inputs.items():
        data = np.load(file_io.FileIO(filename, mode='rb'), allow_pickle=True)
        if variable_name:
            if isinstance(data, np.ndarray):
                logging.warn('Input file %s contains a single ndarray. Name key "%s" ignored.' % (filename, variable_name))
                tensor_key_feed_dict[input_tensor_key] = data
            elif variable_name in data:
                tensor_key_feed_dict[input_tensor_key] = data[variable_name]
            else:
                raise RuntimeError('Input file %s does not contain variable with name "%s".' % (filename, variable_name))
        elif isinstance(data, np.lib.npyio.NpzFile):
            variable_name_list = data.files
            if len(variable_name_list) != 1:
                raise RuntimeError('Input file %s contains more than one ndarrays. Please specify the name of ndarray to use.' % filename)
            tensor_key_feed_dict[input_tensor_key] = data[variable_name_list[0]]
        else:
            tensor_key_feed_dict[input_tensor_key] = data
    for (input_tensor_key, py_expr_evaluated) in input_exprs.items():
        if input_tensor_key in tensor_key_feed_dict:
            logging.warn('input_key %s has been specified with both --inputs and --input_exprs options. Value in --input_exprs will be used.' % input_tensor_key)
        tensor_key_feed_dict[input_tensor_key] = py_expr_evaluated
    for (input_tensor_key, example) in input_examples.items():
        if input_tensor_key in tensor_key_feed_dict:
            logging.warn('input_key %s has been specified in multiple options. Value in --input_examples will be used.' % input_tensor_key)
        tensor_key_feed_dict[input_tensor_key] = example
    return tensor_key_feed_dict

def show():
    if False:
        i = 10
        return i + 15
    'Function triggered by show command.'
    if _SMCLI_ALL.value:
        _show_all(_SMCLI_DIR.value)
    elif _SMCLI_TAG_SET.value is None:
        if _SMCLI_LIST_OPS.value:
            print('--list_ops must be paired with a tag-set or with --all.')
        _show_tag_sets(_SMCLI_DIR.value)
    else:
        if _SMCLI_LIST_OPS.value:
            _show_ops_in_metagraph(_SMCLI_DIR.value, _SMCLI_TAG_SET.value)
        if _SMCLI_SIGNATURE_DEF.value is None:
            _show_signature_def_map_keys(_SMCLI_DIR.value, _SMCLI_TAG_SET.value)
        else:
            _show_inputs_outputs(_SMCLI_DIR.value, _SMCLI_TAG_SET.value, _SMCLI_SIGNATURE_DEF.value)

def run():
    if False:
        for i in range(10):
            print('nop')
    'Function triggered by run command.\n\n  Raises:\n    AttributeError: An error when neither --inputs nor --input_exprs is passed\n    to run command.\n  '
    if not _SMCLI_INPUTS.value and (not _SMCLI_INPUT_EXPRS.value) and (not _SMCLI_INPUT_EXAMPLES.value):
        raise AttributeError('At least one of --inputs, --input_exprs or --input_examples must be required')
    tensor_key_feed_dict = load_inputs_from_input_arg_string(_SMCLI_INPUTS.value, _SMCLI_INPUT_EXPRS.value, _SMCLI_INPUT_EXAMPLES.value)
    run_saved_model_with_feed_dict(_SMCLI_DIR.value, _SMCLI_TAG_SET.value, _SMCLI_SIGNATURE_DEF.value, tensor_key_feed_dict, _SMCLI_OUTDIR.value, _SMCLI_OVERWRITE.value, worker=_SMCLI_WORKER.value, init_tpu=_SMCLI_INIT_TPU.value, use_tfrt=_SMCLI_USE_TFRT.value, tf_debug=_SMCLI_TF_DEBUG.value)

def scan():
    if False:
        print('Hello World!')
    'Function triggered by scan command.'
    if _SMCLI_TAG_SET.value and _SMCLI_OP_DENYLIST.value:
        scan_meta_graph_def(saved_model_utils.get_meta_graph_def(_SMCLI_DIR.value, _SMCLI_TAG_SET.value), _get_op_denylist_set(_SMCLI_OP_DENYLIST.value))
    elif _SMCLI_TAG_SET.value:
        scan_meta_graph_def(saved_model_utils.get_meta_graph_def(_SMCLI_DIR.value, _SMCLI_TAG_SET.value), _OP_DENYLIST)
    else:
        saved_model = saved_model_utils.read_saved_model(_SMCLI_DIR.value)
        if _SMCLI_OP_DENYLIST.value:
            for meta_graph_def in saved_model.meta_graphs:
                scan_meta_graph_def(meta_graph_def, _get_op_denylist_set(_SMCLI_OP_DENYLIST.value))
        else:
            for meta_graph_def in saved_model.meta_graphs:
                scan_meta_graph_def(meta_graph_def, _OP_DENYLIST)

def convert_with_tensorrt():
    if False:
        i = 10
        return i + 15
    "Function triggered by 'convert tensorrt' command."
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    if not _SMCLI_CONVERT_TF1_MODEL.value:
        params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(max_workspace_size_bytes=_SMCLI_MAX_WORKSPACE_SIZE_BYTES.value, precision_mode=_SMCLI_PRECISION_MODE.value, minimum_segment_size=_SMCLI_MINIMUM_SEGMENT_SIZE.value)
        try:
            converter = trt.TrtGraphConverterV2(input_saved_model_dir=_SMCLI_DIR.value, input_saved_model_tags=_SMCLI_TAG_SET.value.split(','), **params._asdict())
            converter.convert()
        except Exception as exc:
            raise RuntimeError('{}. Try passing "--convert_tf1_model=True".'.format(exc)) from exc
        converter.save(output_saved_model_dir=_SMCLI_OUTPUT_DIR.value)
    else:
        trt.create_inference_graph(None, None, max_batch_size=1, max_workspace_size_bytes=_SMCLI_MAX_WORKSPACE_SIZE_BYTES.value, precision_mode=_SMCLI_PRECISION_MODE.value, minimum_segment_size=_SMCLI_MINIMUM_SEGMENT_SIZE.value, is_dynamic_op=True, input_saved_model_dir=_SMCLI_DIR.value, input_saved_model_tags=_SMCLI_TAG_SET.value.split(','), output_saved_model_dir=_SMCLI_OUTPUT_DIR.value)

def freeze_model():
    if False:
        print('Hello World!')
    'Function triggered by freeze_model command.'
    checkpoint_path = _SMCLI_CHECKPOINT_PATH.value or os.path.join(_SMCLI_DIR.value, 'variables/variables')
    if not _SMCLI_VARIABLES_TO_FEED.value:
        variables_to_feed = []
    elif _SMCLI_VARIABLES_TO_FEED.value.lower() == 'all':
        variables_to_feed = None
    else:
        variables_to_feed = _SMCLI_VARIABLES_TO_FEED.value.split(',')
    saved_model_aot_compile.freeze_model(checkpoint_path=checkpoint_path, meta_graph_def=saved_model_utils.get_meta_graph_def(_SMCLI_DIR.value, _SMCLI_TAG_SET.value), signature_def_key=_SMCLI_SIGNATURE_DEF_KEY.value, variables_to_feed=variables_to_feed, output_prefix=_SMCLI_OUTPUT_PREFIX.value)

def aot_compile_cpu():
    if False:
        i = 10
        return i + 15
    'Function triggered by aot_compile_cpu command.'
    checkpoint_path = _SMCLI_CHECKPOINT_PATH.value or os.path.join(_SMCLI_DIR.value, 'variables/variables')
    if not _SMCLI_VARIABLES_TO_FEED.value:
        variables_to_feed = []
    elif _SMCLI_VARIABLES_TO_FEED.value.lower() == 'all':
        variables_to_feed = None
    else:
        variables_to_feed = _SMCLI_VARIABLES_TO_FEED.value.split(',')
    saved_model_aot_compile.aot_compile_cpu_meta_graph_def(checkpoint_path=checkpoint_path, meta_graph_def=saved_model_utils.get_meta_graph_def(_SMCLI_DIR.value, _SMCLI_TAG_SET.value), signature_def_key=_SMCLI_SIGNATURE_DEF_KEY.value, variables_to_feed=variables_to_feed, output_prefix=_SMCLI_OUTPUT_PREFIX.value, target_triple=_SMCLI_TARGET_TRIPLE.value, target_cpu=_SMCLI_TARGET_CPU.value, cpp_class=_SMCLI_CPP_CLASS.value, multithreading=_SMCLI_MULTITHREADING.value.lower() not in ('f', 'false', '0'))

def add_show_subparser(subparsers):
    if False:
        print('Hello World!')
    'Add parser for `show`.'
    show_msg = "Usage examples:\nTo show all tag-sets in a SavedModel:\n$saved_model_cli show --dir /tmp/saved_model\n\nTo show all available SignatureDef keys in a MetaGraphDef specified by its tag-set:\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve\n\nFor a MetaGraphDef with multiple tags in the tag-set, all tags must be passed in, separated by ';':\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve,gpu\n\nTo show all inputs and outputs TensorInfo for a specific SignatureDef specified by the SignatureDef key in a MetaGraph.\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve --signature_def serving_default\n\nTo show all ops in a MetaGraph.\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve --list_ops\n\nTo show all available information in the SavedModel:\n$saved_model_cli show --dir /tmp/saved_model --all"
    parser_show = subparsers.add_parser('show', description=show_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser_show.set_defaults(func=show)

def add_run_subparser(subparsers):
    if False:
        while True:
            i = 10
    'Add parser for `run`.'
    run_msg = 'Usage example:\nTo run input tensors from files through a MetaGraphDef and save the output tensors to files:\n$saved_model_cli show --dir /tmp/saved_model --tag_set serve \\\n   --signature_def serving_default \\\n   --inputs input1_key=/tmp/124.npz[x],input2_key=/tmp/123.npy \\\n   --input_exprs \'input3_key=np.ones(2)\' \\\n   --input_examples \'input4_key=[{"id":[26],"weights":[0.5, 0.5]}]\' \\\n   --outdir=/out\n\nFor more information about input file format, please see:\nhttps://www.tensorflow.org/guide/saved_model_cli\n'
    parser_run = subparsers.add_parser('run', description=run_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser_run.set_defaults(func=run)

def add_scan_subparser(subparsers):
    if False:
        return 10
    'Add parser for `scan`.'
    scan_msg = 'Usage example:\nTo scan for default denylisted ops in SavedModel:\n$saved_model_cli scan --dir /tmp/saved_model\nTo scan for a specific set of ops in SavedModel:\n$saved_model_cli scan --dir /tmp/saved_model --op_denylist OpName,OpName,OpName\nTo scan a specific MetaGraph, pass in --tag_set\n'
    parser_scan = subparsers.add_parser('scan', description=scan_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser_scan.set_defaults(func=scan)

def add_convert_subparser(subparsers):
    if False:
        while True:
            i = 10
    'Add parser for `convert`.'
    convert_msg = 'Usage example:\nTo convert the SavedModel to one that have TensorRT ops:\n$saved_model_cli convert \\\n   --dir /tmp/saved_model \\\n   --tag_set serve \\\n   --output_dir /tmp/saved_model_trt \\\n   tensorrt \n'
    parser_convert = subparsers.add_parser('convert', description=convert_msg, formatter_class=argparse.RawTextHelpFormatter)
    convert_subparsers = parser_convert.add_subparsers(title='conversion methods', description='valid conversion methods', help='the conversion to run with the SavedModel')
    parser_convert_with_tensorrt = convert_subparsers.add_parser('tensorrt', description='Convert the SavedModel with Tensorflow-TensorRT integration', formatter_class=argparse.RawTextHelpFormatter)
    parser_convert_with_tensorrt.set_defaults(func=convert_with_tensorrt)

def add_freeze_model_subparser(subparsers):
    if False:
        for i in range(10):
            print('nop')
    'Add parser for `freeze_model`.'
    compile_msg = '\n'.join(['Usage example:', 'To freeze a SavedModel in preparation for tfcompile:', '$saved_model_cli freeze_model \\', '   --dir /tmp/saved_model \\', '   --tag_set serve \\', '   --output_prefix /tmp/saved_model_xla_aot'])
    parser_compile = subparsers.add_parser('freeze_model', description=compile_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser_compile.set_defaults(func=freeze_model)

def add_aot_compile_cpu_subparser(subparsers):
    if False:
        return 10
    'Add parser for `aot_compile_cpu`.'
    compile_msg = '\n'.join(['Usage example:', 'To compile a SavedModel signature via (CPU) XLA AOT:', '$saved_model_cli aot_compile_cpu \\', '   --dir /tmp/saved_model \\', '   --tag_set serve \\', '   --output_dir /tmp/saved_model_xla_aot', '', '', 'Note: Additional XLA compilation options are available by setting the ', 'XLA_FLAGS environment variable.  See the XLA debug options flags for ', 'all the options: ', '  {}'.format(_XLA_DEBUG_OPTIONS_URL), '', 'For example, to disable XLA fast math when compiling:', '', 'XLA_FLAGS="--xla_cpu_enable_fast_math=false" $saved_model_cli ', 'aot_compile_cpu ...', '', 'Some possibly useful flags:', '  --xla_cpu_enable_fast_math=false', '  --xla_force_host_platform_device_count=<num threads>', '    (useful in conjunction with disabling multi threading)'])
    parser_compile = subparsers.add_parser('aot_compile_cpu', description=compile_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser_compile.set_defaults(func=aot_compile_cpu)

def create_parser():
    if False:
        print('Hello World!')
    'Creates a parser that parse the command line arguments.\n\n  Returns:\n    A namespace parsed from command line arguments.\n  '
    parser = argparse_flags.ArgumentParser(description='saved_model_cli: Command-line interface for SavedModel', conflict_handler='resolve')
    parser.add_argument('-v', '--version', action='version', version='0.1.0')
    subparsers = parser.add_subparsers(title='commands', description='valid commands', help='additional help')
    add_show_subparser(subparsers)
    add_run_subparser(subparsers)
    add_scan_subparser(subparsers)
    add_convert_subparser(subparsers)
    add_aot_compile_cpu_subparser(subparsers)
    add_freeze_model_subparser(subparsers)
    return parser

def main():
    if False:
        for i in range(10):
            print('nop')
    logging.set_verbosity(logging.INFO)

    def smcli_main(argv):
        if False:
            print('Hello World!')
        parser = create_parser()
        if len(argv) < 2:
            parser.error('Too few arguments.')
        flags.mark_flags_as_required(command_required_flags[argv[1]])
        args = parser.parse_args()
        args.func()
    app.run(smcli_main)
if __name__ == '__main__':
    main()
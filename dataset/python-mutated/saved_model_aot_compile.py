"""Helper utilities for AOT compilation."""
import collections
import copy
import os
import re
import shlex
from typing import List, Tuple
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import versions
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
try:
    from tensorflow.python import _pywrap_tfcompile
except ImportError as e:
    _pywrap_tfcompile_import_error = ImportError('Unable to import _pywrap_tfcompile; you must build TensorFlow with XLA.  You may need to build tensorflow with flag --define=with_xla_support=true.  Original error: {}'.format(str(e)))
else:
    _pywrap_tfcompile_import_error = None
_READ_ONLY_VARIABLE_OPS = ('ReadVariableOp', 'IsVariableInitializedOp', 'ResourceGather', 'ResourceGatherNd', 'VariableShape')
_PASS_THROUGH_VARIABLE_OPS = ('Identity', 'IdentityN')

def _shlex_quote(s):
    if False:
        for i in range(10):
            print('nop')
    return shlex.quote(s)

def _sysconfig_module():
    if False:
        while True:
            i = 10
    'Load tf.sysconfig if available and working (i.e., inside a pip package).'
    try:
        _ = sysconfig_lib.get_include()
    except (ImportError, ValueError):
        return None
    return sysconfig_lib

def _parse_tensor_name(name):
    if False:
        return 10
    "Convert a tensor name like 'tensor:0' into a tuple ('tensor', 0)."
    if ':' in name and (not name.endswith(':')):
        node_name = name[:name.rfind(':')]
        output_slot = int(name[name.rfind(':') + 1:])
        return (node_name, output_slot)
    else:
        return (name, None)
_XLA_MAKEFILE_TEMPLATE = '\nINC = -I{tensorflow_includes}\nLIB = -L{compiled_dir}\nCXXFLAGS = {cxx_flags}\n'

def _xla_makefile_string(output_prefix):
    if False:
        i = 10
        return i + 15
    'Returns a Makefile string with variables for using XLA binary object files.\n\n  Attempts to identify the right include header paths when run from either\n  an installed TensorFlow pip package, or from bazel run.\n\n  Args:\n    output_prefix: A string containing the output prefix for the XLA AOT\n      compiled header + object files.\n\n  Returns:\n    A string containing a filled out `_XLA_MAKEFILE_TEMPLATE`.\n  '
    sysconfig = _sysconfig_module()
    (output_dir, _) = os.path.split(output_prefix)
    if sysconfig:
        tensorflow_includes = _shlex_quote(sysconfig.get_include())
    else:
        if os.path.islink(__file__):
            this_file = __file__
            while os.path.islink(this_file):
                this_file = os.readlink(this_file)
            base = os.path.realpath(os.path.join(os.path.dirname(this_file), *[os.path.pardir] * 3))
        else:
            try:
                base = test.test_src_dir_path('')
            except KeyError:
                base = os.path.realpath(os.path.join(os.path.dirname(__file__), *[os.path.pardir] * 3))
        expected_header = os.path.join(base, 'tensorflow', 'compiler', 'tf2xla', 'xla_compiled_cpu_function.h')
        if not os.path.exists(expected_header):
            logging.error('Could not find includes path.  Missing file: {}'.format(expected_header))
        tensorflow_includes = base
    return _XLA_MAKEFILE_TEMPLATE.format(tensorflow_includes=tensorflow_includes, compiled_dir=_shlex_quote(output_dir), cxx_flags='-D_GLIBCXX_USE_CXX11_ABI={}'.format(versions.CXX11_ABI_FLAG))

def _get_variable_nodes_from_graph_def(graph_def):
    if False:
        return 10
    'Get the list of Variable nodes from `graph_def`.\n\n  Args:\n    graph_def: An instance of `GraphDef`.  This GraphDef *must*\n      have already been optimized by Grappler.  In particular, function\n      inlining must have already happened.\n\n  Returns:\n    A dict mapping string names of variables to tuples `(node_def, modified)`,\n    where `node_def` is the `NodeDef` corresponding to variable, and `modified`\n    is a python bool describing whether the variable is modified during runtime.\n  '
    variables = [n for n in graph_def.node if n.op == 'VarHandleOp']
    variable_name_map = dict(((n.name, n) for n in variables))
    child_map = collections.defaultdict(lambda : [])
    for n in graph_def.node:
        for inp in n.input:
            if not inp.startswith('^'):
                child_map[inp].append(n)
    variables = {}
    for (v_name, v_node) in variable_name_map.items():
        queue = list(child_map[v_name])
        processed = set([])
        while queue:
            n_current = queue.pop()
            if n_current.name in processed:
                continue
            processed.add(n_current.name)
            if n_current.op in _PASS_THROUGH_VARIABLE_OPS:
                children = child_map.get(n_current.name, [])
                queue.extend(children)
            elif n_current.op not in _READ_ONLY_VARIABLE_OPS:
                variables[v_name] = (v_node, True)
                queue = []
        if v_name not in variables:
            variables[v_name] = (v_node, False)
    return variables

def _prune_removed_feed_nodes(signature_def, graph_def):
    if False:
        return 10
    'Identify the inputs in the signature no longer in graph_def, prune them.\n\n  Args:\n    signature_def: A `SignatureDef` instance.\n    graph_def: A `GraphDef` instance.\n\n  Returns:\n    A new pruned `SignatureDef`.\n  '
    node_names = set([n.name for n in graph_def.node])
    new_signature_def = meta_graph_pb2.SignatureDef()
    new_signature_def.CopyFrom(signature_def)
    for (k, v) in signature_def.inputs.items():
        (tensor_name, _) = _parse_tensor_name(v.name)
        if tensor_name not in node_names:
            logging.warn("Signature input key '{}', tensor name '{}', has been pruned while freezing the graph.  Removing it from the compiled signatures.".format(k, tensor_name))
            del new_signature_def.inputs[k]
    return new_signature_def

def freeze_model(checkpoint_path: str, meta_graph_def: meta_graph_pb2.MetaGraphDef, output_prefix: str, signature_def_key: str, variables_to_feed: List[str]) -> Tuple[str, str]:
    if False:
        i = 10
        return i + 15
    "Freeze a `MetaGraphDef` in preparation for tfcompile`.\n\n  The graph is always optimized with grappler, and optionally (by default)\n  variables are frozen as constants, before compilation happens.\n\n  Args:\n    checkpoint_path: Python string.  Path to checkpoints/variables.\n    meta_graph_def: Instance of `MetaGraphDef`.\n    output_prefix: Python string.  Path prefix for outputs.\n    signature_def_key: String, the signature_def to use in the SavedModel.\n    variables_to_feed: A list of strings, the variables that will be fed by the\n      user; these won't be frozen.  If `None`, then we will extract all the\n      variables in the graph and mark them as to-feed.  The default behavior is\n      an empty tuple: all variables must be frozen.\n  Returns:\n    a pair containing the path to the frozen model and the path to the config.\n  Raises:\n    RuntimeError: If tensorflow was not built with XLA.\n    ImportError: If tensorflow was built with XLA but there was another\n      issue importing the tfcompile python wrapper.\n    ValueError: If `meta_graph_def.signature_def[signature_def_key]` is\n      missing or has empty outputs.\n  "
    if _pywrap_tfcompile_import_error:
        raise _pywrap_tfcompile_import_error
    signature_def_map = meta_graph_def.signature_def
    if signature_def_key not in signature_def_map:
        raise ValueError(f"Unable to find signature_def_key '{signature_def_key}' in signature def map of `meta_graph_def`. Available keys: {list(signature_def_map.keys())}")
    signature_def = signature_def_map[signature_def_key]
    if not signature_def.outputs:
        raise ValueError(f'Signature key {signature_def_key} must have outputs, but saw none:\n{str(signature_def)}')
    file_io.recursive_create_dir(output_prefix)
    if logging.get_verbosity() >= logging.INFO:
        original_graph_def_location = os.path.join(output_prefix, 'original_graph.pb')
        with file_io.FileIO(original_graph_def_location, 'wb') as graph_writer:
            graph_writer.write(meta_graph_def.graph_def.SerializeToString())
    _replace_input_placeholders_with_default_values(meta_graph_def.graph_def, signature_def)
    graph_def = _optimize_graph(meta_graph_def, signature_def)
    all_variables = _get_variable_nodes_from_graph_def(graph_def)
    if variables_to_feed is None:
        variable_nodes_to_feed = list(all_variables.values())
    else:
        not_in_graph = set(variables_to_feed).difference(list(all_variables))
        if not_in_graph:
            raise ValueError(f'Asked to feed variables that were not found in graph: {not_in_graph}. Variables contained in the graph: {list(all_variables)}')
        variable_nodes_to_feed = [all_variables[name] for name in variables_to_feed]
    if logging.get_verbosity() >= logging.INFO:
        prefrozen_graph_def_location = os.path.join(output_prefix, 'prefrozen_graph.pb')
        with file_io.FileIO(prefrozen_graph_def_location, 'wb') as graph_writer:
            graph_writer.write(graph_def.SerializeToString())
    with session.Session(graph=ops_lib.Graph()) as sess:
        restorer = saver_lib.import_meta_graph(meta_graph_def, clear_devices=True)
        if restorer is not None:
            restorer.restore(sess, checkpoint_path)
        graph_def.CopyFrom(convert_to_constants.convert_variables_to_constants(sess, graph_def, output_node_names=[_parse_tensor_name(n.name)[0] for n in signature_def.outputs.values()], variable_names_blacklist=[n.name for (n, _) in variable_nodes_to_feed]))
    signature_def = _prune_removed_feed_nodes(signature_def, graph_def)
    frozen_graph_def_location = os.path.join(output_prefix, 'frozen_graph.pb')
    config_pbtxt_location = os.path.join(output_prefix, 'config.pbtxt')
    logging.info('Writing graph def to: {}'.format(frozen_graph_def_location))
    with file_io.FileIO(frozen_graph_def_location, 'wb') as graph_writer:
        graph_writer.write(graph_def.SerializeToString())
    config = _signature_to_tf2xla_config(signature_def, variable_nodes_to_feed=variable_nodes_to_feed)
    logging.info('Writing config_pbtxt to: {}'.format(config_pbtxt_location))
    with file_io.FileIO(config_pbtxt_location, mode='w') as config_writer:
        config_writer.write(str(config))
    return (frozen_graph_def_location, config_pbtxt_location)

def aot_compile_cpu_meta_graph_def(checkpoint_path, meta_graph_def, output_prefix, signature_def_key, cpp_class, target_triple, target_cpu, variables_to_feed=(), multithreading=False):
    if False:
        for i in range(10):
            print('nop')
    "Compile a `MetaGraphDef` to header+object files in `output_prefix`.\n\n  Use XLA AOT (`tfcompile`) to convert the given meta graph and\n  signature into a header + object files.  Also create an include makefile\n  that helps identify the appropriate necessary include and library paths\n  to incorporate these files into your C++ program.\n\n  Freezing a graph entails restoring the checkpoint and replacing any inputs and\n  variables with constants. If values are feed, those are used, else inputs are\n  replaced with default all-zero constants. Finally, the graph is pruned and\n  then optimized with grappler.\n\n  If the `freeze_graph` is `True`, all variables are embedded as constants\n  into the graph and binary objects.  If it is `False`, then the variable\n  values become inputs and outputs of the compiled class and the C++\n  caller must set these values manually.\n\n  Args:\n    checkpoint_path: Python string.  Path to checkpoints/variables.\n    meta_graph_def: Instance of `MetaGraphDef`.\n    output_prefix: Python string.  Path prefix for outputs.\n    signature_def_key: String, the signature_def to use in the SavedModel.\n    cpp_class: String, Name of output C++ class.\n    target_triple: String, LLVM target triple.\n    target_cpu: String, LLVM target cpu name.\n    variables_to_feed: A list of strings, the variables that will be fed by the\n      user; these won't be frozen.  If `None`, then we will extract all the\n      variables in the graph and mark them as to-feed.  The default behavior is\n      an empty tuple: all variables must be frozen.\n    multithreading: Whether to enable multithreading in the compiled\n      computation.  Note that if using this option, the resulting object files\n      may have external dependencies on multithreading libraries like nsync.\n\n  Raises:\n    RuntimeError: If tensorflow was not built with XLA.\n    ImportError: If tensorflow was built with XLA but there was another\n      issue importing the tfcompile python wrapper.\n    ValueError: If `meta_graph_def.signature_def[signature_def_key]` is\n      missing or has empty outputs.\n  "
    if _pywrap_tfcompile_import_error:
        raise _pywrap_tfcompile_import_error
    else:
        xla_flags = os.environ.get('XLA_FLAGS')
        if not xla_flags:
            xla_flags = '--xla_cpu_multi_thread_eigen={}'.format('true' if multithreading else 'false')
        else:
            xla_flags += ' --xla_cpu_multi_thread_eigen={}'.format('true' if multithreading else 'false')
        os.environ['XLA_FLAGS'] = xla_flags
    temp_dir = test.get_temp_dir()
    file_io.recursive_create_dir(temp_dir)
    (frozen_graph_def_location, config_pbtxt_location) = freeze_model(checkpoint_path=checkpoint_path, meta_graph_def=meta_graph_def, output_prefix=temp_dir, signature_def_key=signature_def_key, variables_to_feed=variables_to_feed)
    output_dir = os.path.dirname(output_prefix)
    file_io.recursive_create_dir(output_dir)
    entry_point = re.sub('[^0-9a-zA-Z]+', '_', '__xla_' + output_prefix + '__' + cpp_class)
    logging.info('Generating XLA AOT artifacts in: {}'.format(output_dir))
    makefile_inc_location = '{}_makefile.inc'.format(output_prefix)
    with file_io.FileIO(makefile_inc_location, mode='w') as makefile_writer:
        makefile_writer.write(_xla_makefile_string(output_prefix))
    output_prefix = _shlex_quote(output_prefix)
    _pywrap_tfcompile.Compile(graph=frozen_graph_def_location, config=config_pbtxt_location, cpp_class=cpp_class, target_triple=target_triple, target_cpu=target_cpu, entry_point=entry_point, out_function_object='{}.o'.format(output_prefix), out_header='{}.h'.format(output_prefix), out_metadata_object='{}_metadata.o'.format(output_prefix), gen_name_to_index=True, gen_program_shape=False)

def _optimize_graph(meta_graph_def, signature_def):
    if False:
        while True:
            i = 10
    'Optimize `meta_graph_def` using grappler.  Returns a `GraphDef`.'
    new_meta_graph_def = copy.deepcopy(meta_graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for tensor_info in list(signature_def.inputs.values()) + list(signature_def.outputs.values()):
        fetch_collection.node_list.value.append(tensor_info.name)
    new_meta_graph_def.collection_def['train_op'].CopyFrom(fetch_collection)
    new_meta_graph_def.ClearField('saver_def')
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.min_graph_nodes = -1
    return tf_optimizer.OptimizeGraph(config, new_meta_graph_def)

def _replace_input_placeholders_with_default_values(graph_def, signature_def):
    if False:
        i = 10
        return i + 15
    "Replace graphdef's `tf.placeholder` input ops with all-zero constants."
    name_to_node_map = dict(((n.name, n) for n in graph_def.node))
    processed_nodes = set([])
    for (name, input_) in signature_def.inputs.items():
        (tensor_name, _) = _parse_tensor_name(input_.name)
        if tensor_name in processed_nodes:
            continue
        processed_nodes.add(tensor_name)
        if tensor_name not in name_to_node_map:
            raise RuntimeError(f"Unable to find input signature tensor '{tensor_name}' in optimized GraphDef. Graph nodes are: {list(name_to_node_map.keys())}")
        node = name_to_node_map[tensor_name]
        if node.op not in ('Placeholder', 'PlaceholderV2'):
            logging.info("Tried to convert SavedModel input node '{}' from a placeholder, but it doesn't look like a placeholder: {}".format(tensor_name, node))
            continue
        shape = tensor_shape.TensorShape(input_.tensor_shape)
        if not shape.is_fully_defined():
            raise ValueError(f"Expected fully defined input shape for signature_def '{name}', tensor name: '{tensor_name}'; but shape is: {shape}.")
        temp_graph = ops_lib.Graph()
        with temp_graph.as_default():
            const = array_ops.zeros(shape, dtype=input_.dtype, name=tensor_name)
        node.CopyFrom(const.op.node_def)
        for op in temp_graph.get_operations():
            if op.name == const.op.name:
                continue
            graph_def.node.append(op.node_def)
            name_to_node_map[op.node_def.name] = op.node_def

def _signature_to_tf2xla_config(signature_def, variable_nodes_to_feed):
    if False:
        print('Hello World!')
    'Convert `signature_def` to tf2xla config.  Returns a `tf2xla.Config` proto.\n\n  Args:\n    signature_def: Instance of `SignatureDef`.\n    variable_nodes_to_feed: List of tuples of form `(node_def, modified)`\n      corresponding to VarHandleOp, and a boolean `modified` that describes\n      whether the variable was modified during execution.\n\n  Returns:\n    An instance of `tf2xla.Config` proto.\n\n  Raises:\n    RuntimeError: If TensorFlow was not compiled with XLA.\n  '
    from tensorflow.compiler.tf2xla import tf2xla_pb2
    config = tf2xla_pb2.Config()
    tensor_id = tf2xla_pb2.TensorId
    for (name, input_) in signature_def.inputs.items():
        name = name.replace('/', '_')
        name = 'feed_{}'.format(name)
        (node_name, output_index) = _parse_tensor_name(input_.name)
        output_index = int(output_index)
        config.feed.append(tf2xla_pb2.Feed(id=tensor_id(node_name=node_name, output_index=output_index), name=name, type=input_.dtype, shape=input_.tensor_shape))
    for (name, output_) in signature_def.outputs.items():
        name = name.replace('/', '_')
        name = 'fetch_{}'.format(name)
        (node_name, output_index) = _parse_tensor_name(output_.name)
        output_index = int(output_index)
        config.fetch.append(tf2xla_pb2.Fetch(id=tensor_id(node_name=node_name, output_index=output_index), name=name, type=output_.dtype, shape=output_.tensor_shape))
    for (node, modified) in variable_nodes_to_feed:
        name = node.name.replace('/', '_')
        name = 'param_{}'.format(name)
        config.variable.append(tf2xla_pb2.Variable(node_name=node.name, name=name, type=node.attr['dtype'].type, shape=node.attr['shape'].shape, readonly=not modified))
    return config
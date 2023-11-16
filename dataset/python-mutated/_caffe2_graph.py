import copy
import logging
import os
import re
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from typing import Set, Dict, Tuple, List
log = logging.getLogger(__name__)

def _make_unique_name(seen: Set[str], name: str, min_version: int=0):
    if False:
        for i in range(10):
            print('nop')
    "\n    Make the name unique by appending a unique number to the name. Used for SSA.\n\n    Args:\n        seen (set): Set of names that have already been used (with respect to\n            some context).\n        name (str): The name to make unique\n        min_version (number): Starting index. Is incremented continually until\n            it can make the resulting name unique relative to 'seen'.\n\n    Returns:\n        x (str): A version of name that is not in seen.\n    "
    assert name is not None
    i = min_version
    x = '%s_%d' % (name, i) if i else name
    while x in seen:
        i += 1
        x = '%s_%d' % (name, i)
    seen.add(x)
    return x

def _rename_tensorflow_style(shapes, blob_name_tracker, ops):
    if False:
        return 10
    '\n    Convert some of the common names in Caffe2 to tensorflow.\n\n    NOTE: The common names in both Caffe2 and Tensorflow are currently\n        hardcoded, if either side changes at some point, then this code should\n        change as well.\n\n    Args:\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n        blob_name_tracker: Dictionary of all unique blob names (with respect to\n            some context).\n        ops: List of Caffe2 operators\n\n    Returns:\n        None. The _rename_all() call modifies blob_name_tracker and ops in-place.\n    '
    WEIGHT = re.compile('(_w)$')
    WEIGHT_ = re.compile('(_w_)')
    BN = re.compile('(_bn)$')
    BN_ = re.compile('(_bn_)')
    BIAS = re.compile('(_b)$')
    BIAS_ = re.compile('(_b_)')
    SCALE = re.compile('(_s)$')
    SCALE_ = re.compile('(_s_)')
    SUM = re.compile('(_sum)$')
    SUM_ = re.compile('(_sum_)')
    BRANCH = re.compile('(_branch)')

    def f(name):
        if False:
            while True:
                i = 10
        inter_name = WEIGHT_.sub('/weight_', WEIGHT.sub('/weight', name))
        inter_name = BN_.sub('/batchnorm_', BN.sub('/batchnorm', inter_name))
        inter_name = BIAS_.sub('/bias_', BIAS.sub('/bias', inter_name))
        inter_name = SCALE_.sub('/scale_', SCALE.sub('/scale', inter_name))
        inter_name = SUM_.sub('/sum_', SUM.sub('/sum', inter_name))
        new_name = BRANCH.sub('/branch', inter_name)
        return new_name
    _rename_all(shapes, blob_name_tracker, ops, f)

def _convert_to_ssa(shapes, blob_name_tracker, ops):
    if False:
        while True:
            i = 10
    '\n    Convert an operator graph to SSA (i.e. out-of-place).\n\n    i.e. blobs will be renamed so that each blob is produced only once.\n\n    Args:\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n        blob_name_tracker: Dictionary of all unique blob names (with respect to\n            some context).\n        ops: List of Caffe2 operators\n\n    Returns:\n        None. Modifies blob_name_tracker and ops in-place.\n    '
    ir = core.IR(ops)
    seen: Set[str] = set()
    versioned: Dict[Tuple[str, int], int] = {}
    new_shapes = {}
    new_blob_name_tracker = {}

    def ssa_name(name: str, versions: Dict[str, int]) -> int:
        if False:
            return 10
        assert name in versions
        version = versions[name]
        if (name, version) in versioned:
            return versioned[name, version]
        new_name = _make_unique_name(seen, name, min_version=version)
        versioned[name, version] = new_name
        if name in shapes:
            new_shapes[new_name] = shapes[name]
        if blob_name_tracker and name in blob_name_tracker:
            new_blob_name_tracker[new_name] = blob_name_tracker[name]
        return new_name
    for (op, ssa) in zip(ops, ir.ssa):
        assert op is ssa.op
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        op.input.extend((ssa_name(name, ssa.in_versions) for name in inputs))
        op.output.extend((ssa_name(name, ssa.out_versions) for name in outputs))
    shapes.clear()
    shapes.update(new_shapes)
    if blob_name_tracker:
        blob_name_tracker.clear()
        blob_name_tracker.update(new_blob_name_tracker)

def _get_blob_names(ops):
    if False:
        i = 10
        return i + 15
    "\n    Get all the operator input and output blobs and perform dedup on their names.\n\n    Args:\n        ops: List of Caffe2 operators to extract inputs and outputs from\n\n    Returns:\n        set containing distinct inputs and outputs from 'ops'\n    "
    names = set()
    for op in ops:
        names.update(op.input)
        names.update(op.output)
    return {name: name for name in names}

def _remap_keys(old_dict, rename_fn):
    if False:
        print('Hello World!')
    "\n    Rename keys of 'old_dict' according to 'rename_fn'.\n\n    Args:\n        old_dict: Dictionary (i.e. containing blob_name -> blob_name\n            relationships.)\n        rename_fn: Function string -> string for renaming.\n\n    Returns:\n        None. Modifies old_dict in-place.\n    "
    new_dict = {rename_fn(key): value for (key, value) in old_dict.items()}
    old_dict.clear()
    old_dict.update(new_dict)

def _rename_all(shapes, blob_name_tracker, ops, rename_fn):
    if False:
        return 10
    "\n    Rename all the names in the operators.\n\n    Args:\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n        blob_name_tracker: Dictionary of all unique blob names (with respect to\n            some context).\n        ops: List of Caffe2 operators\n        rename_fn: Function string -> string that specifies how to rename\n\n    Returns:\n        None. Modifies shapes, blob_name_tracker and ops in-place using the\n            specified 'rename_fn'.\n    "
    seen: Set[str] = set()
    renamed: Dict[Tuple[str, int], int] = {}

    def g(name):
        if False:
            return 10
        'Collision-free version of f.'
        if name is None:
            return None
        if name in renamed:
            return renamed[name]
        new_name = _make_unique_name(seen, rename_fn(name))
        renamed[name] = new_name
        return new_name
    for op in ops:
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        op.input.extend((g(name) for name in inputs))
        op.output.extend((g(name) for name in outputs))
    _remap_keys(shapes, g)
    if blob_name_tracker:
        _remap_keys(blob_name_tracker, g)
    seen.clear()
    renamed.clear()
    for op in ops:
        op.name = g(op.name)

def _add_gradient_scope(shapes, blob_name_tracker, ops):
    if False:
        i = 10
        return i + 15
    '\n    For all operators or blobs with name containing "_grad", add a "GRADIENTS/" scope.\n\n    Note: breaks graph execution since the blob -> gradient mapping is\n    hardcoded.\n\n    Args:\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n        blob_name_tracker: Dictionary of all unique blob names (with respect to\n            some context).\n        ops: List of Caffe2 operators\n\n    Returns:\n        None. Modifies shapes, blob_name_tracker and ops in-place by renaming.\n    '

    def f(name):
        if False:
            while True:
                i = 10
        if '_grad' in name:
            return f'GRADIENTS/{name}'
        else:
            return name
    _rename_all(shapes, blob_name_tracker, ops, f)

def _replace_colons(shapes, blob_name_tracker, ops, repl):
    if False:
        for i in range(10):
            print('nop')
    "\n    `:i` has a special meaning in Tensorflow. This function replaces all colons with $ to avoid any possible conflicts.\n\n    Args:\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n        blob_name_tracker: Dictionary of all unique blob names (with respect to\n            some context).\n        ops: List of Caffe2 operators\n        repl: String representing the text to replace ':' with. Usually this is\n            '$'.\n\n    Returns:\n        None. Modifies blob_name_tracker in-place.\n\n    "

    def f(name):
        if False:
            for i in range(10):
                print('nop')
        return name.replace(':', repl)
    _rename_all(shapes, blob_name_tracker, ops, f)

def _fill_missing_operator_names(ops):
    if False:
        print('Hello World!')
    "\n    Give missing operators a name.\n\n    We expect C2 operators to be generally unnamed. This gives them a scope\n    (inferred from their outputs) and a name after their type. Duplicates will\n    be postfixed by an index.\n\n    Args:\n        ops: List of Caffe2 operators to assign names to.\n\n    Returns:\n        None: Modifies 'ops' in-place.\n    "
    seen = set()
    for op in ops:
        seen.update(op.input)
        seen.update(op.output)
    for op in ops:
        if op.name:
            name = op.name
        elif op.output or op.input:
            name_list = [os.path.dirname(name) for name in op.output or op.input]
            scope = os.path.commonprefix(name_list)
            name = os.path.join(scope, op.type)
        else:
            name = op.type
        assert name
        op.name = _make_unique_name(seen, name)

def _tf_device(device_option):
    if False:
        for i in range(10):
            print('nop')
    '\n    Handle the devices.\n\n    Args:\n        device_option (caffe2_pb2.DeviceOption): DeviceOption protobuf,\n            associated to an operator, that contains information such as\n            device_type (optional), cuda_gpu_id (optional), node_name (optional,\n            tells which node the operator should execute on). See caffe2.proto\n            in caffe2/proto for the full list.\n\n    Returns:\n        Formatted string representing device information contained in\n            device_option.\n    '
    if not device_option.HasField('device_type'):
        return ''
    if device_option.device_type == caffe2_pb2.CPU or device_option.device_type == caffe2_pb2.MKLDNN:
        return '/cpu:*'
    if device_option.device_type == caffe2_pb2.CUDA:
        return f'/gpu:{device_option.device_id}'
    raise Exception('Unhandled device', device_option)

def _add_tf_shape(attr_dict, ints):
    if False:
        while True:
            i = 10
    '\n    Convert a list of ints to a TensorShapeProto representing the dimensions of a blob/object.\n\n    Args:\n        attr_dict: Dictionary to update (usually attributes of a Node)\n        ints: List of integers representing dimensions of some object.\n\n    Returns:\n        None. Modifies attr_dict in-place.\n    '
    shape_proto = TensorShapeProto()
    for i in ints:
        dim = TensorShapeProto.Dim()
        dim.size = i
        shape_proto.dim.extend([dim])
    attr_dict['_output_shapes'].list.shape.extend([shape_proto])

def _set_tf_attr(attr_dict, arg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add attributes to a node. Key is the arg.name, and values can be shape, floats, strings, ints or an empty list.\n\n    Args:\n        attr_dict: Dictionary to update (usually attributes of a Node)\n        arg: Object with name and data fields.\n\n    Returns:\n        None. Modifies attr_dict in-place.\n    '
    k = arg.name
    if k == 'shape' and arg.ints:
        _add_tf_shape(attr_dict, arg.ints)
        return
    if arg.HasField('f'):
        attr_dict[k].f = arg.f
        return
    if arg.HasField('i'):
        attr_dict[k].i = arg.i
        return
    if arg.HasField('s'):
        attr_dict[k].s = arg.s if isinstance(arg.s, bytes) else str(arg.s).encode('utf-8')
        return
    if arg.floats:
        attr_dict[k].list.f.extend(arg.floats)
        return
    if arg.ints:
        attr_dict[k].list.i.extend(arg.ints)
        return
    if arg.strings:
        attr_dict[k].list.s.extend((s if isinstance(s, bytes) else str(s).encode('utf-8') for s in arg.strings))
        return
    attr_dict[k].list.s.extend([])

def _operator_to_node(shapes, op):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert an operator to a node in a TF graph.\n\n    Args:\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n        op: The Caffe2 operator to convert to a TF graph node.\n\n    Returns:\n        n: The TF graph node created from op.\n    '
    assert op.name, op
    n = NodeDef()
    n.name = op.name
    n.input.extend(op.input)
    n.op = op.type
    n.device = _tf_device(op.device_option)
    if shapes:
        for output in op.output:
            if output not in shapes:
                break
            _add_tf_shape(n.attr, shapes[output])
    for arg in op.arg:
        _set_tf_attr(n.attr, arg)
    return n

def _operator_to_node_simp(op, inter_blobs, seen):
    if False:
        print('Hello World!')
    "\n    Convert the operators to nodes.\n\n    Args:\n        op: Caffe2 operator to convert to node\n        inter_blobs: Set of intermediate blobs\n        seen: Names that have already been used and are not unique\n\n    Returns:\n        nodes: Nodes representing 'op' and the outputs of 'op'\n    "
    assert op
    nodes = []
    outputs = [o for o in op.output if o not in inter_blobs]
    seen.update(outputs)
    len_outputs = len(outputs)
    if len_outputs == 1:
        n = NodeDef()
        n.name = outputs[0]
        n.input.extend(op.input)
        n.op = op.type
        n.device = _tf_device(op.device_option)
        for arg in op.arg:
            _set_tf_attr(n.attr, arg)
        nodes.append(n)
    elif len_outputs > 1:
        if op.name:
            name = op.name
        else:
            name_list = list(outputs)
            scope = os.path.commonprefix(name_list)
            name = os.path.join(scope, op.type)
        assert name
        op.name = _make_unique_name(seen, name)
        device = _tf_device(op.device_option)
        for output in outputs:
            n = NodeDef()
            n.name = output
            n.input.extend([op.name])
            n.op = 'Blob'
            n.device = device
            nodes.append(n)
        n = NodeDef()
        n.name = op.name
        n.input.extend(op.input)
        n.op = op.type
        n.device = device
        for arg in op.arg:
            _set_tf_attr(n.attr, arg)
        nodes.append(n)
    return nodes

def _blob_to_node(producing_ops, shapes, name):
    if False:
        return 10
    '\n    Convert a blob (operator input or output) to a node in a TF graph.\n\n    Args:\n        producing_ops: Dictionary of blob name to list of\n            (producing_op, blob_index within producing_op.output) mapping.\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n        name: String representing the name of this blob.\n\n    Returns:\n        n: The TF graph node created from this blob.\n    '
    assert name
    n = NodeDef()
    n.name = name
    produced_by = producing_ops.get(name, [])
    if len(produced_by) > 0:
        n.op = 'Blob'
    else:
        n.op = 'Placeholder'
    n.input.extend(('%s:%d' % (p_op.name, i) for (p_op, i) in produced_by))
    if produced_by:
        device = produced_by[0][0].device_option
        if all((producer[0].device_option == device for producer in produced_by)):
            n.device = _tf_device(device)
    if shapes and name in shapes:
        _add_tf_shape(n.attr, shapes[name])
    return n

def _clear_debug_info(ops, perform_clear):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove debug information from operators, they are copious.\n\n    Args:\n        ops: List of Caffe2 operators\n        perform_clear: Boolean passed from _operators_to_graph_def specifying\n            whether to remove the debug information. This boolean is passed into\n            this function to reduce the complexity of _operators_to_graph_def.\n\n    Returns:\n        None. Modifies the list of Caffe2 operators in-place and removes the\n        'debug_info' field.\n\n    "
    if not perform_clear:
        return
    for op in ops:
        if op.HasField('debug_info'):
            op.ClearField('debug_info')

def _check_if_forward(blob):
    if False:
        while True:
            i = 10
    "\n    Blobs with names containing '_m' or 'grad' are part of the backward pass.\n\n        This function references facebookresearch/Detectron/detectron/utils/net.py.\n\n    Args:\n        blob: The blob to inspect\n\n    Returns:\n        Boolean representing whether this blob is part of the forward pass\n    "
    return blob.find('__m') < 0 or blob.find('grad') < 0

def _check_if_cpu(blob):
    if False:
        print('Hello World!')
    "\n    Check if the blob's name starts with '_gpu'.\n\n    Args:\n        blob: The blob to inspect\n\n    Returns:\n        Boolean representing whether this blob is associated with a gpu\n    "
    return not blob.startswith('_gpu')

def _compute_in_out(ops):
    if False:
        while True:
            i = 10
    '\n    Find the input, intermediate and output nodes of a set of operators.\n\n    Args:\n        ops: List of Caffe2 operators to look through\n\n    Returns:\n        input_blobs: The input nodes of the set of operators\n        inter_blobs: The intermediate nodes of the set of operators\n        output_blobs: The output nodes of the set of operators\n    '
    in_blobs = set()
    out_blobs = set()
    for op in ops:
        for input_blob in op.input:
            in_blobs.add(input_blob)
        for output_blob in op.output:
            out_blobs.add(output_blob)
    input_blobs = list(in_blobs.difference(out_blobs))
    output_blobs = list(out_blobs.difference(in_blobs))
    inter_blobs = {b for b in output_blobs if b.startswith('_')}
    output_blobs = [b for b in output_blobs if b not in inter_blobs]
    return (input_blobs, inter_blobs, output_blobs)

def _filter_ops(ops, filter_fn, perform_filter):
    if False:
        while True:
            i = 10
    "\n    Filter unwanted operators based on criteria in 'filter_fn'.\n\n    Args:\n        ops: List of Caffe2 operators to filter\n        filter_fn: Criteria function for whether inputs/outputs in an operator\n            should be filtered.\n        perform_filter: Boolean passed from _operators_to_graph_def specifying\n            whether to filter operators\n\n    Returns:\n        new_ops: Subset of ops containing a subset of their inputs and outputs.\n    "
    if not perform_filter:
        return ops
    new_ops = []
    for op in ops:
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        new_inputs = [i for i in inputs if filter_fn(i)]
        new_outputs = [o for o in outputs if filter_fn(o)]
        if new_outputs:
            op.input.extend(new_inputs)
            op.output.extend(new_outputs)
            new_ops.append(op)
    return new_ops

def _operators_to_graph_def(shapes, ops, colon_replacement='$', with_ssa=True, with_gradient_scope=True, blob_name_tracker=None, show_simplified=False, custom_rename=None):
    if False:
        while True:
            i = 10
    "\n    Convert a set of operators to a graph using the main function.\n\n    Args:\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n        ops: List of Caffe2 operators, representing some computation graph\n        ### **kwargs (model_to_graph_def, nets_to_graph_def, protos_to_graph_def) ###\n        colon_replacement: Symbol to replace ':' with. ':i' in TF has a special\n            meaning, so we need to replace it with a non-conflicting symbol.\n        with_ssa: Boolean\n        with_gradient_scope: Boolean\n        blob_name_tracker: Dictionary tracking names of blobs (inputs/outputs\n            from operators)\n        show_simplified: Whether to show a simplified version of the model graph\n            Sets all of the following values:\n                clear_debug_info: Boolean representing whether to silence debug\n                    info (which can be very verbose)\n                show_forward_only: Boolean representing whether to only show\n                    blobs involved in the forward pass\n                show_cpu_only: Boolean representing whether to only show blobs\n                    that are not associated with a gpu\n                use_tensorflow_naming: Boolean representing whether to convert\n                    some common Caffe2 naming conventions to their Tensorflow\n                    counterparts\n        custom_rename: Function string -> string that defines a custom\n            renaming function to use.\n\n    Returns:\n        current_graph: GraphDef representing the computation graph formed by the\n            set of operators.\n    "
    if blob_name_tracker is not None:
        blob_name_tracker.clear()
    else:
        blob_name_tracker = {}
    blob_name_tracker.update(_get_blob_names(ops))
    _clear_debug_info(ops, show_simplified)
    ops = _filter_ops(ops, _check_if_forward, show_simplified)
    ops = _filter_ops(ops, _check_if_cpu, show_simplified)
    if custom_rename:
        _rename_all(shapes, blob_name_tracker, ops, custom_rename)
    if colon_replacement:
        _replace_colons(shapes, blob_name_tracker, ops, colon_replacement)
    if with_ssa:
        _convert_to_ssa(shapes, blob_name_tracker, ops)
    if with_gradient_scope:
        _add_gradient_scope(shapes, blob_name_tracker, ops)
    _fill_missing_operator_names(ops)
    if show_simplified:
        _rename_tensorflow_style(shapes, blob_name_tracker, ops)
    producing_ops: Dict[caffe2_pb2.OperatorDef, List] = {}
    blobs = set()
    (input_blobs, inter_blobs, _) = _compute_in_out(ops)
    current_graph = GraphDef()
    seen = set(input_blobs)
    for op in ops:
        nodes_from_op = _operator_to_node_simp(op, inter_blobs, seen) if show_simplified else [_operator_to_node(shapes, op)]
        current_graph.node.extend(nodes_from_op)
        for input_blob in op.input:
            blobs.add(input_blob)
        for (i, output_blob) in enumerate(op.output):
            blobs.add(output_blob)
            producing_ops.setdefault(output_blob, []).append((op, i))
    if show_simplified:
        blobs = input_blobs
    for blob in sorted(blobs):
        current_graph.node.extend([_blob_to_node(producing_ops, {}, blob)])
    return current_graph

def _propagate_device_option(net_def):
    if False:
        for i in range(10):
            print('nop')
    '\n    Propagate the device options from net to operators.\n\n    Args:\n        net_def: A caffe2_pb2.NetDef representing a computation graph. The graph\n            consists of Caffe2 operators.\n\n    Returns:\n        None. Iterates through all ops contained within the net. For each op,\n            modifies the op device_option in-place to be the net device_option\n            if the op has no pre-existing device_option, and leaves the op as-is\n            if it already has a device_option.\n    '
    if not net_def.HasField('device_option'):
        return
    for op in net_def.op:
        if not op.HasField('device_option'):
            op.device_option.CopyFrom(net_def.device_option)

def _try_get_shapes(nets):
    if False:
        i = 10
        return i + 15
    '\n    Get missing shapes for all blobs contained in the nets.\n\n    Args:\n        nets: List of core.Net to extract blob shape information from.\n\n    Returns:\n        Dictionary containing blob name to shape/dimensions mapping. The net\n            is a computation graph that is composed of operators, and the\n            operators have input and output blobs, each with their own dims.\n    '
    try:
        (shapes, _) = workspace.InferShapesAndTypes(nets)
        return shapes
    except Exception as e:
        log.warning('Failed to compute shapes: %s', e)
        return {}

def model_to_graph_def(model, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert a Caffe2 model to a Tensorflow graph.\n\n    This function extracts 'param_init_net' and 'net' from the model and passes it to nets_to_graph()\n    for further processing.\n\n    Args:\n        model (cnn.CNNModelHelper, model_helper.ModelHelper): The model to\n            extract the nets (instances of core.Net) from.\n\n    Returns:\n        Call to nets_to_graph_def() with extracted 'param_init_net', 'net' and\n            **kwargs. See _operators_to_graph_def for detailed **kwargs.\n    "
    nets = [model.param_init_net, model.net]
    return nets_to_graph_def(nets, **kwargs)

def nets_to_graph_def(nets, shapes=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a set of Caffe2 nets to a Tensorflow graph.\n\n    Args:\n        nets: List of core.Nets. core.Net is a wrapper around a NetDef protobuf.\n            The corresponding protobuf can be extracted using .Proto().\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n\n    Returns:\n        Call to protos_to_graph_def() with the extracted NetDef protobufs and\n            **kwargs. See _operators_to_graph_def for detailed **kwargs.\n    '
    shapes = {}
    nets = [copy.deepcopy(net.Proto()) for net in nets]
    shapes = copy.deepcopy(shapes)
    return protos_to_graph_def(nets, shapes, **kwargs)

def protos_to_graph_def(net_defs, shapes=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a set of Caffe2 net definitions to a Tensorflow graph.\n\n    Args:\n        net_defs: List of caffe2_pb2.NetDef protobufs representing computation\n            graphs.\n        shapes: Dictionary mapping blob names to their shapes/dimensions.\n\n    Returns:\n        Call to _operators_to_graph_def() with the extracted operators from the\n            NetDefs and **kwargs. See _operators_to_graph_def for detailed\n            **kwargs.\n    '
    for net in net_defs:
        _propagate_device_option(net)
    shapes = copy.deepcopy(shapes or {})
    ops = [op for net_def in net_defs for op in net_def.op]
    return _operators_to_graph_def(shapes, ops, **kwargs)
from caffe2.proto import caffe2_pb2

def gen_do_gradient(op, g_output):
    if False:
        return 10
    "\n    Generates gradient Do operator, given forward Do op and a list\n    of gradient blobs corresponding to forward op's outputs\n    Returns a gradient op and a list of blobs corresponding to input gradients\n    "
    from caffe2.python.core import BlobReference
    (subnet, outer_to_inner_map, inner_to_outer_map, workspace_blob_name) = _do_op_sanity_check_and_process(op)
    assert len(g_output) == len(op.output), 'Different number of gradient blobs and Do op outputs'
    (grad_ops, deduped_g_output) = dedupe_g_output(op, g_output)
    g_output = deduped_g_output
    op_output = [str(o) for o in op.output]
    op_output = op_output[:-1]
    op_input = [str(i) for i in op.input]
    op_input = op_input[:-1]
    ordered_inner_output_blob_names = [outer_to_inner_map[o] for o in op_output]
    backward_pass_initial_grad_map = {}
    initial_grad_map = {}
    for (inner_output_name, outer_grad_output_name) in zip(ordered_inner_output_blob_names, g_output):
        if outer_grad_output_name:
            inner_grad_output_name = inner_output_name + '/_DO_OPERATOR_INNER_GRAD_'
            backward_pass_initial_grad_map[BlobReference(inner_output_name)] = BlobReference(inner_grad_output_name)
            initial_grad_map[inner_grad_output_name] = str(outer_grad_output_name)
    assert len(initial_grad_map) > 0, 'Empty initial gradient map for Do op'
    (inner_grad_ops, inner_grad_names_map) = _gen_subgradient_pass(subnet, backward_pass_initial_grad_map)
    if len(inner_grad_ops) == 0:
        return ([], [])
    grad_copy_ops = []
    g_input = []
    new_op_outputs = []
    new_blob_bindings = {}
    for outer_input_name in op_input:
        inner_input_name = outer_to_inner_map[outer_input_name]
        if inner_input_name in inner_grad_names_map:
            inner_grad_input_name = inner_grad_names_map[inner_input_name]
            outer_grad_input_name = outer_input_name + '_grad'
            new_inner_grad_input_name = inner_input_name + '/_DO_OPERATOR_INNER_GRAD_COPY_'
            grad_copy_ops.append(_prepare_blob_copy_op(inner_grad_input_name, new_inner_grad_input_name))
            new_blob_bindings[new_inner_grad_input_name] = outer_grad_input_name
            new_op_outputs.append(outer_grad_input_name)
            g_input.append(outer_grad_input_name)
        else:
            g_input.append(None)
    new_op_inputs = []
    overwritten_names = set()
    saved_local_blob_names = set()
    for grad_op in inner_grad_ops:
        grad_op_input = [str(i) for i in grad_op.input]
        grad_op_output = [str(o) for o in grad_op.output]
        for grad_op_input_name in grad_op_input:
            if grad_op_input_name in overwritten_names:
                continue
            outer_name = inner_to_outer_map.get(grad_op_input_name, None)
            if not outer_name:
                outer_name = initial_grad_map.get(grad_op_input_name, None)
            if outer_name:
                outer_name = str(outer_name)
                if outer_name not in new_op_inputs:
                    new_op_inputs.append(outer_name)
                new_blob_bindings[grad_op_input_name] = outer_name
            else:
                saved_local_blob_names.add(grad_op_input_name)
        overwritten_names.update(grad_op_output)
    inner_grad_ops += grad_copy_ops
    gradient_do_def = _prepare_gradient_do_op(fwd_op=op, fwd_net=subnet, grad_ops=inner_grad_ops, inputs=new_op_inputs, outputs=new_op_outputs, blob_bindings=new_blob_bindings, saved_fwd_blobs=saved_local_blob_names, workspace_blob_name=workspace_blob_name)
    grad_ops.append(gradient_do_def)
    _do_op_sanity_check_and_process(gradient_do_def)
    return (grad_ops, g_input)

def dedupe_g_output(op, g_output):
    if False:
        while True:
            i = 10
    grad_ops = []
    deduped_g_output = []
    init_grad_map = {}
    for (output_name, grad_name) in zip(op.output, g_output):
        if not grad_name:
            deduped_g_output.append(grad_name)
            continue
        if output_name in init_grad_map:
            deduped_g_output.append(init_grad_map[output_name])
        elif grad_name not in init_grad_map.values():
            init_grad_map[output_name] = grad_name
            deduped_g_output.append(grad_name)
        else:
            deduped_grad_name = output_name + '_' + grad_name + '_DEDUP'
            assert deduped_grad_name not in init_grad_map.values()
            grad_copy_op = caffe2_pb2.OperatorDef()
            grad_copy_op.type = 'Copy'
            grad_copy_op.input.extend([grad_name])
            grad_copy_op.output.extend([deduped_grad_name])
            grad_ops.append(grad_copy_op)
            deduped_g_output.append(deduped_grad_name)
            init_grad_map[output_name] = deduped_grad_name
    return (grad_ops, deduped_g_output)

def gen_while_gradient(op, g_output):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates gradient While operator\n    '
    from caffe2.python.core import BlobReference
    assert op.type == 'While', 'Expected While op'
    assert len(op.input) > 0, 'Expected at least one input in While op'
    assert len(op.output) == len(g_output), 'Different number of gradient blobs and While op outputs'
    (grad_ops, deduped_g_output) = dedupe_g_output(op, g_output)
    g_output = deduped_g_output
    init_grad_map = {}
    op_output = [str(o) for o in op.output]
    for (output_name, grad_output_name) in zip(op_output, g_output):
        if grad_output_name:
            init_grad_map[BlobReference(output_name)] = BlobReference(grad_output_name)
    assert len(init_grad_map) > 0, 'Empty initial gradient map for While op'
    loop_net = _get_net_argument(op, 'loop_net')
    assert loop_net, 'Expected loop subnet in While op'
    assert len(loop_net.op) == 1 and loop_net.op[0].type == 'Do', 'Gradient While op requires single Do op as a loop body'
    do_op = loop_net.op[0]
    do_args = _get_do_arguments(do_op)
    assert 'reuse_workspace' not in do_args or not do_args['reuse_workspace'], 'Gradient While op requires Do loop body op without reuse_workspace set'
    assert len(do_op.output) > 0, 'Expected Do op with at least one output'
    workspace_blob = do_op.output[-1]
    (loop_grad_net, loop_grad_map, loop_input_names, loop_output_names) = _gen_subnet_gradient(loop_net, init_grad_map)
    assert loop_grad_net, 'Failed to get gradient net for loop body in While op'
    grad_ops += _prepare_gradient_while_ops(fwd_op=op, input_names=loop_input_names, output_names=loop_output_names, loop_grad_net=loop_grad_net, workspace_blob=workspace_blob, init_grad_map=init_grad_map, loop_grad_map=loop_grad_map)
    op_input = [str(i) for i in op.input]
    g_input = [loop_grad_map.get(i, None) for i in op_input]
    return (grad_ops, g_input)

def _prepare_gradient_while_ops(fwd_op, input_names, output_names, loop_grad_net, workspace_blob, init_grad_map, loop_grad_map):
    if False:
        while True:
            i = 10
    gradient_while_def = caffe2_pb2.OperatorDef()
    gradient_while_def.CopyFrom(fwd_op)
    if gradient_while_def.name:
        gradient_while_def.name += '_grad'
    loop_net_arg = caffe2_pb2.Argument()
    loop_net_arg.name = 'loop_net'
    loop_net_arg.n.CopyFrom(loop_grad_net)
    cond_net_arg = caffe2_pb2.Argument()
    cond_net_arg.name = 'cond_net'
    from caffe2.python.core import Net, BlobReference
    cond_net = Net('gradient_loop_cond_net')
    cond_init_net = Net('gradient_loop_cond_net_init')
    cond_blob = cond_net.NextScopedBlob(cond_net.Name() + '/cond')
    cond_init_net.HasScope(workspace_blob, cond_blob)
    cond_net.HasScope(workspace_blob, cond_blob)
    for (blob, init_grad_blob) in init_grad_map.items():
        blob_name = str(blob)
        init_grad_blob_name = str(init_grad_blob)
        if blob_name in loop_grad_map and loop_grad_map[blob_name] != init_grad_blob_name:
            cond_net.Copy(BlobReference(loop_grad_map[blob_name]), init_grad_blob)
            cond_init_net.Copy(init_grad_blob, BlobReference(loop_grad_map[blob_name]))
    cond_net_arg.n.CopyFrom(cond_net.Proto())
    del gradient_while_def.arg[:]
    gradient_while_def.arg.extend([loop_net_arg, cond_net_arg])
    del gradient_while_def.control_input[:]
    del gradient_while_def.input[:]
    gradient_while_def.input.extend([str(cond_blob).encode('utf-8')] + list(input_names))
    del gradient_while_def.output[:]
    gradient_while_def.output.extend(output_names)
    gradient_while_def.is_gradient_op = True
    return [o for o in cond_init_net.Proto().op] + [gradient_while_def]

def _get_do_arguments(do_op):
    if False:
        return 10
    assert do_op.type == 'Do', 'Expected Do op'
    args = {}
    for arg in do_op.arg:
        if not arg.name:
            continue
        if arg.name == 'net':
            assert arg.n, 'Expected non empty net argument'
            args['net'] = arg.n
        elif arg.name == 'reuse_workspace':
            assert arg.i, 'Expected non empty reuse_workspace argument'
            args['reuse_workspace'] = bool(arg.i)
        elif arg.name == 'inner_blobs':
            assert arg.strings, 'Expected non empty inner_blobs argument'
            args['inner_blobs'] = arg.strings
        elif arg.name == 'outer_blobs_idx':
            assert arg.ints, 'Expected non empty outer_blobs_idx argument'
            args['outer_blobs_idx'] = arg.ints
    return args

def gen_if_gradient(op, g_output):
    if False:
        return 10
    "\n    Generates gradient If operator, given forward If op and a list\n    of gradient blobs corresponding to forward op's outputs\n    Returns a gradient op and a list of blobs corresponding to input gradients\n    "
    from caffe2.python.core import BlobReference
    assert op.type == 'If', 'Expected If op'
    assert len(op.input) > 0, 'Expected at least one input in If op'
    assert len(op.output) == len(g_output), 'Different number of gradient blobs and If op outputs'
    (grad_ops, deduped_g_output) = dedupe_g_output(op, g_output)
    g_output = deduped_g_output
    init_grad_map = {}
    op_input = [str(i) for i in op.input]
    op_output = [str(o) for o in op.output]
    for (output_name, grad_output_name) in zip(op_output, g_output):
        if grad_output_name:
            init_grad_map[BlobReference(output_name)] = BlobReference(grad_output_name)
    assert len(init_grad_map) > 0, 'Empty initial gradient map for If op'
    grad_map = {}
    then_net = _get_net_argument(op, 'then_net')
    assert then_net, 'Expected then subnet in If op'
    (then_grad_net, then_grad_map, then_input_names, then_output_names) = _gen_subnet_gradient(then_net, init_grad_map)
    assert then_grad_net, 'Failed to get gradient net for then in If op'
    grad_map.update(then_grad_map)
    else_input_names = set()
    else_output_names = set()
    else_grad_map = {}
    else_grad_net = None
    else_net = _get_net_argument(op, 'else_net')
    if else_net:
        (else_grad_net, else_grad_map, else_input_names, else_output_names) = _gen_subnet_gradient(else_net, init_grad_map)
        assert else_grad_net, 'Failed to get gradient net for else in If op'
        for (else_blob, else_grad_blob) in else_grad_map.items():
            if else_blob in then_grad_map:
                then_grad_blob = then_grad_map[else_blob]
                if then_grad_blob != else_grad_blob:
                    init_grad_name = init_grad_map[else_blob] if else_blob in init_grad_map else None
                    if then_grad_blob == init_grad_name:
                        grad_map[else_blob] = else_grad_blob
                    elif else_grad_blob == init_grad_name:
                        grad_map[else_blob] = then_grad_blob
                    else:
                        raise 'Unexpected grad blob name ' + else_blob + ', ' + else_grad_blob + ', ' + then_grad_blob
            else:
                grad_map[else_blob] = else_grad_blob
    then_other_output_names = then_output_names - (then_output_names & else_output_names)
    then_other_grad_output_names = set([o for o in then_other_output_names if o in then_grad_map.values()])
    zero_then = _gen_grad_zero_init_ops(init_grad_map, then_grad_map, then_other_grad_output_names)
    if else_grad_net:
        else_grad_net.op.extend(zero_then)
    elif len(zero_then) > 0:
        else_grad_net = caffe2_pb2.NetDef()
        else_grad_net.CopyFrom(then_grad_net)
        if else_grad_net.name:
            else_grad_net.name += '_auto_else_zero_blobs_'
        del else_grad_net.op[:]
        else_grad_net.op.extend(zero_then)
        del else_grad_net.external_input[:]
        del else_grad_net.external_output[:]
    else_other_output_names = else_output_names - (then_output_names & else_output_names)
    else_other_grad_output_names = set([o for o in else_other_output_names if o in else_grad_map.values()])
    zero_else = _gen_grad_zero_init_ops(init_grad_map, else_grad_map, else_other_grad_output_names)
    then_grad_net.op.extend(zero_else)
    output_names = list(then_output_names | else_output_names)
    input_names = then_input_names | else_input_names
    input_names = [op_input[0]] + list(input_names - set(op_input[0]))
    gradient_if_def = _prepare_gradient_if_op(fwd_op=op, input_names=input_names, output_names=output_names, then_grad_net=then_grad_net, else_grad_net=else_grad_net)
    g_input = [grad_map.get(i, None) for i in op_input]
    return (grad_ops + [gradient_if_def], g_input)

def _gen_subnet_gradient(subnet, init_grad):
    if False:
        return 10
    (grad_ops, grad_names_map) = _gen_subgradient_pass(subnet, init_grad)
    output_names = set()
    input_names = set()
    for grad_op in grad_ops:
        for grad_op_input in grad_op.input:
            if str(grad_op_input) not in output_names:
                input_names.add(str(grad_op_input))
        for grad_op_output in grad_op.output:
            output_names.add(str(grad_op_output))
    gradient_net_def = caffe2_pb2.NetDef()
    gradient_net_def.CopyFrom(subnet)
    if gradient_net_def.name:
        gradient_net_def.name += '_grad'
    del gradient_net_def.op[:]
    gradient_net_def.op.extend(grad_ops)
    del gradient_net_def.external_input[:]
    del gradient_net_def.external_output[:]
    return (gradient_net_def, grad_names_map, input_names, output_names)

def _get_net_argument(op, net_name):
    if False:
        while True:
            i = 10
    for arg in op.arg:
        if arg.name and arg.name == net_name:
            assert arg.n, 'Expected non empty net argument ' + net_name
            return arg.n
    return None

def getNetArgument(op, net_name):
    if False:
        for i in range(10):
            print('nop')
    'A wrapper for external call'
    return _get_net_argument(op, net_name)

def _gen_subgradient_pass(subnet, init_grad):
    if False:
        i = 10
        return i + 15
    from caffe2.python.core import IR
    subnet_ir = IR(subnet.op)
    (grad_ops, grad_blob_map) = subnet_ir.GetBackwardPass(init_grad)
    grad_names_map = {}
    for (b, g) in grad_blob_map.items():
        grad_names_map[str(b)] = str(g)
    return (grad_ops, grad_names_map)

def _do_op_sanity_check_and_process(op):
    if False:
        return 10
    assert op.type == 'Do', 'Expected Do op'
    subnet = _get_net_argument(op, 'net')
    assert subnet, 'No net argument found in Do op'
    inner_blobs = None
    outer_blobs_idx = None
    for arg in op.arg:
        if arg.name and arg.name == 'inner_blobs':
            assert not inner_blobs, 'inner_blobs redefinition'
            assert arg.strings and len(arg.strings) > 0, 'Empty inner_blobs argument in Do op'
            inner_blobs = [s.decode('utf-8') for s in arg.strings]
        if arg.name and arg.name == 'outer_blobs_idx':
            assert not outer_blobs_idx, 'outer_blobs_idx redefinition'
            assert arg.ints and len(arg.ints) > 0, 'Empty outer_blobs_idx argument in Do op'
            outer_blobs_idx = arg.ints
        if inner_blobs and outer_blobs_idx:
            break
    assert inner_blobs, 'No inner_blobs argument found in Do op'
    assert outer_blobs_idx, 'No outer_blobs_idx argument found in Do op'
    assert len(inner_blobs) == len(outer_blobs_idx), 'Arguments inner_blobs and outer_blobs_idx of different length in Do op'
    all_inner_blobs = set(inner_blobs)
    assert len(all_inner_blobs) == len(inner_blobs), 'Found duplicates in inner_blobs in Do op'
    op_input = [str(i) for i in op.input]
    assert len(op_input) > 0, 'Expected at least one input blob'
    input_workspace_blob_name = op_input[-1]
    op_input = op_input[:-1]
    op_output = [str(o) for o in op.output]
    assert len(op_output) > 0, 'Expected at least one output blob'
    workspace_blob_name = op_output[-1]
    assert input_workspace_blob_name == workspace_blob_name, 'Expected same input/output workspace blob'
    op_output = op_output[:-1]
    all_op_input_blob_names = set(op_input)
    assert len(all_op_input_blob_names) == len(op_input), 'Found duplicates in Do op inputs'
    all_op_output_blob_names = set(op_output)
    assert len(all_op_output_blob_names) == len(op_output), 'Found duplicates in Do op outputs'
    ordered_outer_blob_names = op_input + op_output
    all_outer_blob_names = set(ordered_outer_blob_names)
    used_outer_blob_names = set()
    outer_to_inner_map = {}
    inner_to_outer_map = {}
    for (inner_name, outer_blob_idx) in zip(inner_blobs, outer_blobs_idx):
        assert outer_blob_idx >= 0 and outer_blob_idx < len(ordered_outer_blob_names), 'Outer blob index is out of bounds in Do op'
        outer_name = ordered_outer_blob_names[outer_blob_idx]
        assert outer_name not in used_outer_blob_names, 'Reusage of outer blob name ' + outer_name + ' in Do op'
        used_outer_blob_names.add(outer_name)
        outer_to_inner_map[outer_name] = inner_name
        inner_to_outer_map[inner_name] = outer_name
    assert len(used_outer_blob_names) == len(all_outer_blob_names), 'Not all outer blob names are used in blob bindings in Do op'
    return (subnet, outer_to_inner_map, inner_to_outer_map, workspace_blob_name)

def _prepare_blob_copy_op(from_name, to_name):
    if False:
        while True:
            i = 10
    copy_op_def = caffe2_pb2.OperatorDef()
    copy_op_def.type = 'Copy'
    copy_op_def.input.extend([from_name])
    copy_op_def.output.extend([to_name])
    return copy_op_def

def _prepare_gradient_do_op(fwd_op, fwd_net, grad_ops, inputs, outputs, blob_bindings, saved_fwd_blobs, workspace_blob_name):
    if False:
        return 10
    gradient_net_def = caffe2_pb2.NetDef()
    gradient_net_def.CopyFrom(fwd_net)
    if gradient_net_def.name:
        gradient_net_def.name += '_grad'
    del gradient_net_def.op[:]
    gradient_net_def.op.extend(grad_ops)
    del gradient_net_def.external_input[:]
    del gradient_net_def.external_output[:]
    gradient_do_def = caffe2_pb2.OperatorDef()
    gradient_do_def.CopyFrom(fwd_op)
    if gradient_do_def.name and len(gradient_do_def.name) > 0:
        gradient_do_def.name += '_grad'
    del gradient_do_def.input[:]
    gradient_do_def.input.extend(inputs)
    gradient_do_def.input.append(workspace_blob_name)
    del gradient_do_def.output[:]
    gradient_do_def.output.extend(outputs)
    gradient_do_def.output.append(workspace_blob_name)
    net_arg = caffe2_pb2.Argument()
    net_arg.name = 'net'
    net_arg.n.CopyFrom(gradient_net_def)
    ordered_new_outer_names = inputs + outputs
    inner_blobs = blob_bindings.keys()
    new_outer_blobs_idx = [ordered_new_outer_names.index(blob_bindings[b]) for b in inner_blobs]
    inner_blobs_arg = caffe2_pb2.Argument()
    inner_blobs_arg.name = 'inner_blobs'
    inner_blobs_arg.strings.extend([b.encode('utf-8') for b in inner_blobs])
    outer_blobs_idx_arg = caffe2_pb2.Argument()
    outer_blobs_idx_arg.name = 'outer_blobs_idx'
    outer_blobs_idx_arg.ints.extend(new_outer_blobs_idx)
    saved_blobs_arg = caffe2_pb2.Argument()
    saved_blobs_arg.name = 'saved_fwd_blobs'
    saved_blobs_arg.strings.extend([b.encode('utf-8') for b in saved_fwd_blobs])
    del gradient_do_def.arg[:]
    gradient_do_def.arg.extend([net_arg, inner_blobs_arg, outer_blobs_idx_arg, saved_blobs_arg])
    del gradient_do_def.control_input[:]
    gradient_do_def.is_gradient_op = True
    return gradient_do_def

def _gen_grad_zero_init_ops(init_grad_map, grad_map, grad_output_names):
    if False:
        while True:
            i = 10
    grad_init_ops = []
    for grad_output in grad_output_names:
        output_name = None
        for (o, g) in grad_map.items():
            if g == grad_output:
                output_name = o
                break
        assert output_name, 'Unknown gradient output ' + grad_output
        grad_init_op = None
        if output_name in init_grad_map:
            init_grad_name = init_grad_map[output_name]
            if init_grad_name != grad_output:
                grad_init_op = caffe2_pb2.OperatorDef()
                grad_init_op.type = 'Copy'
                grad_init_op.input.extend([str(init_grad_name)])
                grad_init_op.output.extend([str(grad_output)])
        else:
            grad_init_op = caffe2_pb2.OperatorDef()
            grad_init_op.type = 'ConstantFill'
            grad_init_op.input.extend([output_name])
            grad_init_op.output.extend([grad_output])
            value_arg = caffe2_pb2.Argument()
            value_arg.name = 'value'
            value_arg.f = 0.0
            grad_init_op.arg.extend([value_arg])
        if grad_init_op:
            grad_init_ops.append(grad_init_op)
    return grad_init_ops

def _prepare_gradient_if_op(fwd_op, input_names, output_names, then_grad_net, else_grad_net):
    if False:
        for i in range(10):
            print('nop')
    gradient_if_def = caffe2_pb2.OperatorDef()
    gradient_if_def.CopyFrom(fwd_op)
    del gradient_if_def.input[:]
    gradient_if_def.input.extend(input_names)
    del gradient_if_def.output[:]
    gradient_if_def.output.extend(output_names)
    then_net_arg = caffe2_pb2.Argument()
    then_net_arg.name = 'then_net'
    then_net_arg.n.CopyFrom(then_grad_net)
    gradient_args = [then_net_arg]
    if else_grad_net:
        else_net_arg = caffe2_pb2.Argument()
        else_net_arg.name = 'else_net'
        else_net_arg.n.CopyFrom(else_grad_net)
        gradient_args.append(else_net_arg)
    del gradient_if_def.arg[:]
    gradient_if_def.arg.extend(gradient_args)
    if gradient_if_def.name:
        gradient_if_def.name += '_grad'
    del gradient_if_def.control_input[:]
    gradient_if_def.is_gradient_op = True
    return gradient_if_def

def disambiguate_grad_if_op_output(grad_op, idx, new_grad_output):
    if False:
        i = 10
        return i + 15
    then_net = _get_net_argument(grad_op, 'then_net')
    old_grad_out_match = grad_op.output[idx]
    for op in then_net.op:
        for (i, out) in enumerate(op.output):
            if out == old_grad_out_match:
                op.output[i] = new_grad_output
    else_net = _get_net_argument(grad_op, 'else_net')
    if else_net:
        for op in else_net.op:
            for (i, out) in enumerate(op.output):
                if out == old_grad_out_match:
                    op.output[i] = new_grad_output
    grad_op.output[idx] = new_grad_output
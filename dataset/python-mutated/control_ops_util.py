from caffe2.python import core

def get_external_blob_names(net, lexical_scope):
    if False:
        while True:
            i = 10
    '\n    Returns a set of blobs a given net depends on and a set of\n    output blobs that are written by the net\n    Inputs:\n        net - net to return input/output blobs for;\n        lexical_scope - all external blob names visible to the net\n    '
    net_proto = net.Proto()
    (net_ssa, _) = core.get_ssa(net_proto)
    input_names = core.get_undefined_blobs(net_ssa)
    for input_name in input_names:
        assert str(input_name) in lexical_scope, 'Input blob ' + input_name + ' is undefined'
    output_names = set()
    for op in net_proto.op:
        for output in op.output:
            if output in lexical_scope:
                output_names.add(output)
    return (input_names, output_names)

def add_if_op(if_net, cond_blob, lexical_scope, then_net, else_net=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    A helper function to add an If op to the net.\n    Automatically determines whether blobs in the then/else subnets are external\n    (from the outer workspace) or local (visible only inside subnet's workspace)\n    based on lexical scope - set of all outer blob names visible to the 'If'\n    operator. All the blobs in then/else subnets with names matching a name in lexical\n    scope and all the blobs that are first used as the operators' inputs are\n    considered outer blobs - these blobs must exist in the outer workspace,\n    then/else subnets can read their values and new values written into these blobs\n    will be visible outside of the 'If' operator. All other blobs are local - exist\n    only within inner workspaces for then/else.\n    Inputs:\n        if_net - net to add an If op to;\n        cond_blob - scalar bool blob reference, used as If condition;\n        lexical_scope - a set of outer blob names visible to then/else branches;\n        then_net/else_net - nets (core.Net) for then/else branches\n    "
    (then_input_blob_names, then_output_blob_names) = get_external_blob_names(then_net, lexical_scope)
    else_input_blob_names = set()
    else_output_blob_names = set()
    if else_net:
        (else_input_blob_names, else_output_blob_names) = get_external_blob_names(else_net, lexical_scope)
    input_blob_names = then_input_blob_names | else_input_blob_names
    output_blob_names = then_output_blob_names | else_output_blob_names
    if_inputs = [cond_blob]
    if_inputs += [core.BlobReference(name=b, net=None) for b in input_blob_names]
    if_outputs = [core.BlobReference(name=b, net=None) for b in output_blob_names]
    do_then_net = core.Net('do_then_net')
    then_input_blobs = [core.BlobReference(name=b, net=None) for b in then_input_blob_names]
    then_output_blobs = [core.BlobReference(name=b, net=None) for b in then_output_blob_names]
    then_input_output_names_ordered = [str(b) for b in then_input_blobs + then_output_blobs]
    then_outer_blob_names = list(then_input_blob_names | then_output_blob_names)
    then_outer_blob_names_idx = [then_input_output_names_ordered.index(b) for b in then_outer_blob_names]
    do_then_workspace_blob = if_net.NextScopedBlob(if_net.Name() + '/workspace_if_then')
    then_input_blobs.append(do_then_workspace_blob)
    then_output_blobs.append(do_then_workspace_blob)
    if_inputs.append(do_then_workspace_blob)
    if_outputs.append(do_then_workspace_blob)
    do_then_net.Do(then_input_blobs, then_output_blobs, net=then_net.Proto(), inner_blobs=then_outer_blob_names, outer_blobs_idx=then_outer_blob_names_idx)
    do_then_net.AddExternalOutput(*then_output_blobs)
    if_args = {}
    if_args['then_net'] = do_then_net.Proto()
    do_else_workspace_blob = None
    if else_net:
        do_else_net = core.Net('do_else_net')
        else_input_blobs = [core.BlobReference(name=b, net=None) for b in else_input_blob_names]
        else_output_blobs = [core.BlobReference(name=b, net=None) for b in else_output_blob_names]
        else_input_output_names_ordered = [str(b) for b in else_input_blobs + else_output_blobs]
        else_outer_blob_names = list(else_input_blob_names | else_output_blob_names)
        else_outer_blob_names_idx = [else_input_output_names_ordered.index(b) for b in else_outer_blob_names]
        do_else_workspace_blob = if_net.NextScopedBlob(if_net.Name() + '/workspace_if_else')
        else_input_blobs.append(do_else_workspace_blob)
        else_output_blobs.append(do_else_workspace_blob)
        if_inputs.append(do_else_workspace_blob)
        if_outputs.append(do_else_workspace_blob)
        do_else_net.Do(else_input_blobs, else_output_blobs, net=else_net.Proto(), inner_blobs=else_outer_blob_names, outer_blobs_idx=else_outer_blob_names_idx)
        do_else_net.AddExternalOutput(*else_output_blobs)
        if_args['else_net'] = do_else_net.Proto()
    if_net.CreateScope([], [do_then_workspace_blob])
    if do_else_workspace_blob:
        if_net.CreateScope([], [do_else_workspace_blob])
    if_net.If(if_inputs, if_outputs, **if_args)
    if_net.AddExternalOutput(*if_outputs)

def add_while_op(while_net, cond_blob, lexical_scope, loop_body_net, condition_body_net=None):
    if False:
        print('Hello World!')
    "\n    A helper function to add a While op to the net. Same rules for determining\n    outer and inner blobs as for the 'If' operator apply for the 'While' operator\n    loop and condition subnets. If specified, condition net is executed in a separate\n    workspace before the first and after each iteration, the last operator must have\n    a single scalar boolean output that is written into the condition blob.\n    Inputs:\n        while_net - net to add a While op to;\n        cond_blob - scalar bool blob reference, used as a stop condition;\n        lexical_scope - a set of outer blob names visible to the loop's body;\n        loop_body_net - net to execute on each iteration;\n        condition_body_net - net to compute condition value\n    "
    (input_blob_names, output_blob_names) = get_external_blob_names(loop_body_net, lexical_scope)
    input_blob_names |= output_blob_names
    loop_inputs = [core.BlobReference(name=b, net=None) for b in input_blob_names]
    loop_outputs = [core.BlobReference(name=b, net=None) for b in output_blob_names]
    while_inputs = [cond_blob] + loop_inputs
    while_outputs = [] + loop_outputs
    do_loop_body_net = core.Net('do_loop_body_net')
    loop_input_output_names_ordered = [str(b) for b in loop_inputs + loop_outputs]
    loop_body_outer_blob_names = list(input_blob_names | output_blob_names)
    loop_body_outer_blob_names_idx = [loop_input_output_names_ordered.index(b) for b in loop_body_outer_blob_names]
    do_loop_body_workspace_blob = while_net.NextScopedBlob(while_net.Name() + '/workspace_loop_body')
    loop_inputs.append(do_loop_body_workspace_blob)
    loop_outputs.append(do_loop_body_workspace_blob)
    while_inputs.append(do_loop_body_workspace_blob)
    while_outputs.append(do_loop_body_workspace_blob)
    do_loop_body_net.Do(loop_inputs, loop_outputs, net=loop_body_net.Proto(), inner_blobs=loop_body_outer_blob_names, outer_blobs_idx=loop_body_outer_blob_names_idx, copy_external_blobs=True)
    do_loop_body_net.AddExternalOutput(*loop_outputs)
    while_args = {}
    while_args['loop_net'] = do_loop_body_net.Proto()
    cond_workspace_blob = None
    if condition_body_net:
        (cond_input_blob_names, cond_output_blob_names) = get_external_blob_names(condition_body_net, lexical_scope)
        found_condition_output = False
        for op in condition_body_net.Proto().op:
            if str(cond_blob) in op.output:
                found_condition_output = True
                break
        assert found_condition_output, 'Condition net does not write into condition blob'
        if str(cond_blob) not in cond_output_blob_names:
            cond_output_blob_names.add(str(cond_blob))
        cond_inputs = [core.BlobReference(name=b, net=None) for b in cond_input_blob_names]
        assert str(cond_blob) in cond_output_blob_names, 'Condition blob expected in condition net output'
        cond_outputs = [core.BlobReference(name=b, net=None) for b in cond_output_blob_names]
        condition_net = core.Net('do_loop_condition_net')
        cond_input_output_names_ordered = [str(b) for b in cond_inputs + cond_outputs]
        cond_body_outer_blob_names = list(cond_input_blob_names | cond_output_blob_names)
        cond_body_outer_blob_names_idx = [cond_input_output_names_ordered.index(b) for b in cond_body_outer_blob_names]
        cond_workspace_blob = while_net.NextScopedBlob(while_net.Name() + '/workspace_loop_cond')
        cond_inputs.append(cond_workspace_blob)
        cond_outputs.append(cond_workspace_blob)
        condition_net.Do(cond_inputs, cond_outputs, net=condition_body_net.Proto(), inner_blobs=cond_body_outer_blob_names, outer_blobs_idx=cond_body_outer_blob_names_idx)
        condition_net.AddExternalOutput(*cond_outputs)
        while_args['cond_net'] = condition_net.Proto()
        while_inputs += [b for b in cond_inputs if str(b) not in input_blob_names]
        while_outputs += [b for b in cond_outputs if str(b) not in output_blob_names]
        if str(cond_blob) not in lexical_scope:
            while_net.ConstantFill([], cond_blob, dtype=core.DataType.BOOL, value=False)
    while_net.CreateScope([], [do_loop_body_workspace_blob])
    if cond_workspace_blob:
        while_net.CreateScope([], [cond_workspace_blob])
    while_net.While(while_inputs, while_outputs, **while_args)
    while_net.AddExternalOutput(*while_outputs)
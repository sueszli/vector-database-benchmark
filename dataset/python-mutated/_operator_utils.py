def _repack_list(sets, fn):
    if False:
        i = 10
        return i + 15
    "Repack list from [[a, b, c], [a', b', c'], ....]\n    to [fn(a, a', ...), fn(b, b', ...), fn(c, c', ...)]\n    where fn can be `tuple` or `list`\n    Assume that all elements of input have the same length\n    "
    output_list = []
    arg_list_len = len(sets[0])
    for i in range(arg_list_len):
        output_list.append(fn((input_set[i] for input_set in sets)))
    return output_list

def _repack_output_sets(outputs):
    if False:
        return 10
    'Repack and "transpose" the output sets, from groups of outputs of individual operators\n    to interleaved groups of consecutive outputs, that is from:\n    [[out0, out1, out2], [out0\', out1\', out2\'], ...] produce:\n    [[out0, out0\', ...], [out1, out1\', ...], [out2, out2\', ...]]\n\n\n    Assume that all elements of input have the same length\n    If the inputs were 1-elem lists, it is flattened, that is:\n    [[out0], [out0\'], [out0\'\'], ...] -> [out0, out0\', out0\'\', ...]\n    '
    if len(outputs) > 1 and len(outputs[0]) == 1:
        output = []
        for elem in outputs:
            output.append(elem[0])
        return output
    return _repack_list(outputs, list)

def _build_input_sets(inputs, op_name):
    if False:
        return 10
    'Detect if the list of positional inputs [Inp_0, Inp_1, Inp_2, ...], represents Multiple\n    Input Sets (MIS) to operator and prepare lists of regular DataNode-only positional inputs to\n    individual operator instances.\n\n    If all Inp_i are DataNodes there are no MIS involved.\n    If any of Inp_i is a list of DataNodes, this is considered a MIS. In that case, non-list\n    Inp_i is repeated to match the length of the one that is a list, and those lists are regrouped,\n    for example:\n\n    inputs = [a, b, [x, y, z], [u, v, w]]\n\n    # "a" and "b" are repeated to match the length of [x, y, z]:\n    -> [[a, a, a], [b, b, b], [x, y, z], [u, v, w]]\n\n    # input sets are rearranged, so they form a regular tuples of DataNodes suitable to being passed\n    # to one Operator Instance.\n    -> [(a, b, x, u), (a, b, y, v), (a, b, z, w)]\n\n    This allows to create 3 operator instances, each with 4 positional inputs.\n\n    Parameters\n    ----------\n    inputs : List of positional inputs\n        The inputs are either DataNodes or lists of DataNodes indicating MIS.\n    op_name : str\n        Name of the invoked operator, for error reporting purposes.\n    '

    def _detect_multiple_input_sets(inputs):
        if False:
            while True:
                i = 10
        'Check if any of inputs is a list, indicating a usage of MIS.'
        return any((isinstance(input, list) for input in inputs))

    def _safe_len(input):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(input, list):
            return len(input)
        else:
            return 1

    def _check_common_length(inputs):
        if False:
            for i in range(10):
                print('nop')
        'Check if all list representing multiple input sets have the same length and return it'
        arg_list_len = max((_safe_len(input) for input in inputs))
        for input in inputs:
            if isinstance(input, list):
                if len(input) != arg_list_len:
                    raise ValueError(f'All argument lists for Multiple Input Sets used with operator {op_name} must have the same length')
        return arg_list_len

    def _unify_lists(inputs, arg_list_len):
        if False:
            print('Hello World!')
        'Pack single _DataNodes into lists, so they are treated as Multiple Input Sets\n        consistently with the ones already present\n\n        Parameters\n        ----------\n        arg_list_len : int\n            Number of MIS.\n        '
        result = ()
        for input in inputs:
            if isinstance(input, list):
                result = result + (input,)
            else:
                result = result + ([input] * arg_list_len,)
        return result

    def _repack_input_sets(inputs):
        if False:
            while True:
                i = 10
        "Zip the list from [[arg0, arg0', arg0''], [arg1', arg1'', arg1''], ...]\n        to [(arg0, arg1, ...), (arg0', arg1', ...), (arg0'', arg1'', ...)]\n        "
        return _repack_list(inputs, tuple)
    input_sets = []
    if _detect_multiple_input_sets(inputs):
        arg_list_len = _check_common_length(inputs)
        packed_inputs = _unify_lists(inputs, arg_list_len)
        input_sets = _repack_input_sets(packed_inputs)
    else:
        input_sets = [inputs]
    return input_sets
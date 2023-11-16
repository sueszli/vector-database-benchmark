from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from itertools import permutations

def _get_nn_spec(spec):
    if False:
        return 10
    if spec.WhichOneof('Type') == 'neuralNetwork':
        nn_spec = spec.neuralNetwork
    elif spec.WhichOneof('Type') == 'neuralNetworkClassifier':
        nn_spec = spec.neuralNetworkClassifier
    elif spec.WhichOneof('Type') == 'neuralNetworkRegressor':
        nn_spec = spec.neuralNetworkRegressor
    else:
        raise ValueError('Specification must contain a neural network')
    return nn_spec

def _get_blob_out_degree(spec):
    if False:
        print('Hello World!')
    "\n    Computes use count of every tensor/node in NN graph\n    i.e. How many layers are using it as an input\n\n    :param nn_spec : NeuralNetworkSpecification\n    :returns use_count_dict : str -> int, a dictionary with node name as a key and it's use count as a value\n    "

    def _get_blob_out_degree_rec(nn_spec, out_degree):
        if False:
            while True:
                i = 10
        nn_layers = nn_spec.layers
        for layer in nn_layers:
            layer_type = layer.WhichOneof('layer')
            for inp in layer.input:
                out_degree[inp] = out_degree.get(inp, 0) + 1
            if layer_type == 'loop':
                out_degree[layer.loop.conditionVar] = out_degree.get(layer.loop.conditionVar, 0) + 1
                _get_blob_out_degree_rec(layer.loop.conditionNetwork, out_degree)
                _get_blob_out_degree_rec(layer.loop.bodyNetwork, out_degree)
            elif layer_type == 'branch':
                _get_blob_out_degree_rec(layer.branch.ifBranch, out_degree)
                _get_blob_out_degree_rec(layer.branch.elseBranch, out_degree)
    use_count_dict = {}
    nn_spec = _get_nn_spec(spec)
    _get_blob_out_degree_rec(nn_spec, use_count_dict)
    network_outputs = _get_network_output(spec)
    for _output in network_outputs:
        use_count_dict[_output] = use_count_dict.get(_output, 0) + 1
    return use_count_dict

def _is_layer(nn_layer, layer_type):
    if False:
        return 10
    '\n    :param nn_layer : NN layer proto message\n    :param layer_type : str Layer type to check against\n    :returns True if nn_layer is of type `layer_type` otherwise False\n    '
    return nn_layer.WhichOneof('layer') == layer_type

def _get_input(layer, index=0):
    if False:
        return 10
    '\n    :param layer : NN Layer Proto message\n    :param index : Layer input index (Default 0)\n    :returns name of input at provided index if present, otherwise None\n    '
    if len(layer.input) <= index:
        return None
    return layer.input[index]

def _get_output(layer, index=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param layer : NN Layer Proto message\n    :param index : Layer output index (Default 0)\n    :returns name of output at provided index if present, otherwise None\n    '
    if len(layer.output) <= index:
        return None
    return layer.output[index]

def _get_network_output(spec):
    if False:
        return 10
    '\n    :param spec : CoreML Specification\n    :returns network output names\n    '
    network_output_names = []
    for _out in spec.description.output:
        network_output_names.append(_out.name)
    return network_output_names

def transform_conv_crop(spec):
    if False:
        return 10
    '\n    Transforms Conv -> Crop -> BN (if present) -> Activation (if present) into\n               Conv -> BN (if present) -> Activation (if present) -> Crop\n    This transformation will allow Conv -> BN -> Activation fusion by changing\n    the position of the crop layer, which does not affect the computation\n    '
    out_degree = _get_blob_out_degree(spec)
    network_output_names = _get_network_output(spec)
    nn_spec = _get_nn_spec(spec)
    nn_layers = nn_spec.layers
    for i in range(0, len(nn_layers) - 2):
        if not _is_layer(nn_layers[i], 'convolution'):
            continue
        if not (_is_layer(nn_layers[i + 1], 'crop') and _get_input(nn_layers[i + 1]) not in network_output_names and (out_degree[_get_output(nn_layers[i + 1])] == 1)):
            continue
        layer_to_shuffle_with = -1
        if _is_layer(nn_layers[i + 2], 'batchnorm') and out_degree[_get_output(nn_layers[i + 2])] == 1:
            layer_to_shuffle_with = i + 2
        if i + 3 < len(nn_layers) and _is_layer(nn_layers[i + 3], 'activation') and (out_degree[_get_output(nn_layers[i + 3])] == 1):
            layer_to_shuffle_with = i + 3
        if layer_to_shuffle_with == -1:
            continue
        nn_layers[i].output[0] = nn_layers[i + 1].output[0]
        nn_layers[i + 1].output[0] = nn_layers[layer_to_shuffle_with].output[0]
        nn_layers[layer_to_shuffle_with].output[0] = nn_layers[i + 1].input[0]
        crop_layer = nn_layers[i + 1]
        nn_layers.remove(crop_layer)
        nn_layers.insert(layer_to_shuffle_with, crop_layer)

def remove_disconnected_layers(spec):
    if False:
        i = 10
        return i + 15
    "\n    Removes layers from model specification if it's output is not\n    connected or on path to the network output.\n    "

    def _remove_layers_from_spec(nn_spec, layers_to_delete):
        if False:
            for i in range(10):
                print('nop')
        nn_layers = nn_spec.layers
        for _layer in layers_to_delete:
            nn_layers.remove(_layer)

    def _get_disconnected_layers_rec(nn_spec):
        if False:
            for i in range(10):
                print('nop')
        "\n        - Iteraters over layers in bottom-up fashion\n        - Collect layers if it's output is not being used (marks and does lazy deletion)\n        - Recursively iterates over NN Spec if layer is Loop or Branch\n        "

        def _decrease_input_degree(layer):
            if False:
                i = 10
                return i + 15
            '\n            Helper routine to reduce degree input nodes for given layer\n            '
            for _input in layer.input:
                out_degree[_input] -= 1
                if out_degree[_input] == 0:
                    del out_degree[_input]
        nn_layers = nn_spec.layers
        layers_to_delete = []
        for _layer in reversed(nn_layers):
            layer_type = _layer.WhichOneof('layer')
            if layer_type == 'loop':
                condition_net_layers_to_delete = _get_disconnected_layers_rec(_layer.loop.conditionNetwork)
                body_net_layers_to_delete = _get_disconnected_layers_rec(_layer.loop.bodyNetwork)
                _remove_layers_from_spec(_layer.loop.conditionNetwork, condition_net_layers_to_delete)
                _remove_layers_from_spec(_layer.loop.bodyNetwork, body_net_layers_to_delete)
                if len(_layer.loop.conditionNetwork.layers) == 0 or len(_layer.loop.bodyNetwork.layers) == 0:
                    layers_to_delete.append(_layer)
                    _decrease_input_degree(_layer)
                continue
            if layer_type == 'branch':
                if_layers_to_delete = _get_disconnected_layers_rec(_layer.branch.ifBranch)
                else_layers_to_delete = _get_disconnected_layers_rec(_layer.branch.elseBranch)
                total_if_layers = len(_layer.branch.ifBranch.layers)
                total_else_layers = len(_layer.branch.elseBranch.layers)
                if len(if_layers_to_delete) != total_if_layers and len(else_layers_to_delete) != total_else_layers:
                    _remove_layers_from_spec(_layer.branch.ifBranch, if_layers_to_delete)
                    _remove_layers_from_spec(_layer.branch.elseBranch, else_layers_to_delete)
                elif len(if_layers_to_delete) == total_if_layers and len(else_layers_to_delete) == total_else_layers:
                    layers_to_delete.append(_layer)
                    _decrease_input_degree(_layer)
                continue
            output_is_used = False
            for _output in _layer.output:
                if _output in out_degree:
                    output_is_used = True
                    break
            if not output_is_used:
                layers_to_delete.append(_layer)
                _decrease_input_degree(_layer)
        return layers_to_delete

    def _remove_disconnected_layers_rec(nn_spec):
        if False:
            return 10
        '\n        Entry point for removing disconnected layers\n        '
        layers_to_delete = _get_disconnected_layers_rec(nn_spec)
        _remove_layers_from_spec(nn_spec, layers_to_delete)
    out_degree = _get_blob_out_degree(spec)
    nn_spec = _get_nn_spec(spec)
    _remove_disconnected_layers_rec(nn_spec)

def remove_redundant_transposes(spec):
    if False:
        i = 10
        return i + 15
    '\n    Removes layers from model specification that are back to back transposes\n    that compose to the identity.\n    '

    def blob_name_to_layers(nn_layers):
        if False:
            while True:
                i = 10
        '\n        output_to_layers: {str: layer_proto_message} : {blob name: layers that it feeds into}\n        input_to_parent_layers: {str: layer_proto_message} : {blob name: parent layers that feed in}\n        '
        output_to_layers = {}
        for layer in nn_layers:
            for input in layer.input:
                if not input in output_to_layers:
                    output_to_layers[input] = [layer]
                else:
                    output_to_layers[input].append(layer)
        input_to_parent_layers = {}
        for layer in nn_layers:
            for output in layer.output:
                if not layer.WhichOneof('layer') == 'copy':
                    assert output not in input_to_parent_layers, "'{}' blob is generated by more than 1 layers".format(output)
                input_to_parent_layers[output] = layer
        return (input_to_parent_layers, output_to_layers)

    def _delete_layers(nn_spec, layers_to_delete):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a neural network spec and pairs of transposes to remove, rewire\n        the network to bypass those transposes and remove them from the spec.\n        '
        nn_layers = nn_spec.layers
        (_, output_to_layers) = blob_name_to_layers(nn_layers)
        for layers in layers_to_delete:
            start_layer = layers[0]
            end_layer = layers[-1]
            children = output_to_layers[end_layer.output[0]]
            for child in children:
                idx = [i for (i, input) in enumerate(child.input) if input == end_layer.output[0]]
                assert len(idx) == 1
                idx = idx[0]
                child.input[idx] = start_layer.input[0]
        for layers in layers_to_delete:
            for layer in layers:
                nn_layers.remove(layer)

    def _find_redundant_transposes(nn_spec):
        if False:
            return 10
        '\n        Search the neural network spec for sequence of transposes that together\n        are the identity, and return a list of those sequence.\n        '
        nn_layers = nn_spec.layers
        layers_to_delete = []
        (input_to_parent_layers, output_to_layers) = blob_name_to_layers(nn_layers)
        for layer in nn_layers:
            if not layer.WhichOneof('layer') == 'transpose':
                continue
            if layer.output[0] in output_to_layers and len(output_to_layers[layer.output[0]]) == 1 and (output_to_layers[layer.output[0]][0].WhichOneof('layer') == 'transpose'):
                continue
            layers = []
            cursor = layer
            while True:
                if cursor.output[0] in output_to_layers:
                    layers.append(cursor)
                if not cursor.input[0] in input_to_parent_layers:
                    break
                cursor = input_to_parent_layers[cursor.input[0]]
                if cursor.WhichOneof('layer') != 'transpose':
                    break
                if len(output_to_layers[cursor.output[0]]) != 1:
                    break
            layers = layers[::-1]
            if len(layers) == 0:
                continue

            def solve_dp(layers):
                if False:
                    return 10
                '\n                The resulting dp[i] means the maximum length of transpose sequence resulting\n                in identity starting at index i\n                For example, dp[0] = 0 means there is no sequence starting at 0 results in identity\n                dp[10] = 5 means the longest identity sequence starts at 10 is 5,\n                so [layers[10],layer[11],..,layer[14]] is the longest identity sequence start at 10.\n\n                # dic: {tuple:int}\n                # key is the net transpose axes pattern starting from the first layer\n                # value is the highest id of the layer which has this pattern\n                # e.g. if dic[(1,2,0)] = 34, it means that starting from the 1st layer,\n                # the net transpose pattern  `(1,2,0)` is last seen at layer id 34. No layer after 34-th\n                # layer will result in the net pattern `(1,2,0)`\n                '
                dim = len(layers[0].transpose.axes)
                dp = [0] * len(layers)
                dic = {}
                axes = list(range(dim))
                dic[tuple(axes)] = 0
                for i in range(len(layers)):
                    axes = [axes[k] for k in layers[i].transpose.axes]
                    key = tuple(axes)
                    if key in dic:
                        dp[dic[key]] = i - dic[key] + 1
                    dic[key] = i + 1
                for i in range(len(layers) - 1, -1, -1):
                    j = i + dp[i]
                    if j < len(layers):
                        dp[i] = dp[i] + dp[j]
                return dp
            dp = solve_dp(layers)
            '\n            Once we know the maximum identity sequence starts at each index, we solve\n            for the maximum total node we can remove.\n            I think there must be lots of different solution for this, but I use DP again.\n            sol_num[i] keeps track of the maximum number of nodes can be remove after index i\n            For example, if sol_num[10] = 5, this means after index 10, we can at most remove 5 nodes.\n            sol_bt[i] keeps the first starting point of identity sequence which results in the\n            optimal solution after index i.\n            For example, if sol_num[10] = 12, means that in order to get rid of the maxium number of\n            nodes after 10, the first starting point is index 12.\n            After construct sol_num and sol_bt by dynamic programming, we backtrack for the optimal\n            solution using sol_bt.\n            '
            sol_num = [0] * len(dp)
            sol_bt = [None] * len(dp)
            if dp[-1] != 0:
                sol_num[-1] = dp[-1]
                sol_bt[-1] = len(dp) - 1
            for i in range(len(sol_num) - 2, -1, -1):
                if dp[i] == 0:
                    sol_num[i] = sol_num[i + 1]
                    sol_bt[i] = sol_bt[i + 1]
                else:
                    num = dp[i]
                    j = i + dp[i]
                    if j < len(sol_num):
                        num += sol_num[j]
                    if num > sol_num[i + 1]:
                        sol_num[i] = num
                        sol_bt[i] = i
                    else:
                        sol_num[i] = sol_num[i + 1]
                        sol_bt[i] = sol_bt[i + 1]
            cursor = 0
            while cursor < len(dp):
                if sol_bt[cursor] == None:
                    break
                cursor = sol_bt[cursor]
                tmp = [layers[i] for i in range(cursor, cursor + dp[cursor])]
                layers_to_delete.append(tmp)
                cursor += dp[cursor]
        return layers_to_delete
    nn_spec = _get_nn_spec(spec)
    layers_to_delete = _find_redundant_transposes(nn_spec)
    if len(layers_to_delete) > 0:
        _delete_layers(nn_spec, layers_to_delete)
        print('{} transpose pairs deleted'.format(len(layers_to_delete)))
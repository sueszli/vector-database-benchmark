import keras as _keras
import numpy as _np
_KERAS_LAYERS_1D = [_keras.layers.Convolution1D, _keras.layers.AtrousConvolution1D, _keras.layers.UpSampling1D, _keras.layers.ZeroPadding1D, _keras.layers.Cropping1D, _keras.layers.MaxPooling1D, _keras.layers.AveragePooling1D, _keras.layers.GlobalMaxPooling1D, _keras.layers.GlobalAveragePooling1D]
_KERAS_ACTIVATION_LAYERS = [_keras.layers.Activation, _keras.layers.advanced_activations.LeakyReLU, _keras.layers.advanced_activations.PReLU, _keras.layers.advanced_activations.ELU, _keras.layers.advanced_activations.ParametricSoftplus, _keras.layers.advanced_activations.ThresholdedReLU, _keras.layers.advanced_activations.SReLU]
_KERAS_NORMALIZATION_LAYERS = [_keras.layers.BatchNormalization]
_KERAS_RECURRENT_LAYERS = [_keras.layers.recurrent.LSTM, _keras.layers.recurrent.SimpleRNN, _keras.layers.recurrent.GRU, _keras.layers.wrappers.Bidirectional]

def _to_list(x):
    if False:
        print('Hello World!')
    if type(x) is not list:
        return [x]
    else:
        return x

def _insert_to_dict(d, key, e):
    if False:
        print('Hello World!')
    if key not in d:
        d[key] = []
    if e not in d[key]:
        d[key].append(e)

class NetGraph(object):
    """
    Attributes:
    layer_list - a list of layer names in the Keras model
    connection_map - a map where the key is a layer, the value is a list of its successors
    reverse_connection_map - a map where the key is a layer, the value is a list of its predecessors
    keras_layer_map - a map where the key is a layer name, the value is Keras layer
    model - a reference of the keras model.
    blob_names - blob names for each one of the edge.
    """

    def __init__(self, model):
        if False:
            while True:
                i = 10
        self.layer_list = []
        self.edge_map = {}
        self.reverse_edge_map = {}
        self.keras_layer_map = {}
        self.input_layers = []
        self.output_layers = []
        self.layers_inputs = {}
        self.layers_outputs = {}
        self.optional_inputs = []
        self.optional_outputs = []
        self.layers_optional_inputs = {}
        self.layers_optional_outputs = {}
        self.model = model

    def _add_layer(self, keras_layer):
        if False:
            while True:
                i = 10
        layer = keras_layer.name
        if layer not in self.layer_list:
            self.layer_list.append(layer)
            self.keras_layer_map[layer] = keras_layer

    def get_predecessors(self, layer_name):
        if False:
            i = 10
            return i + 15
        if layer_name in self.reverse_edge_map:
            return self.reverse_edge_map[layer_name][:]
        else:
            return []

    def get_successors(self, layer_name):
        if False:
            for i in range(10):
                print('nop')
        if layer_name in self.edge_map:
            return self.edge_map[layer_name][:]
        else:
            return []

    def get_keras_layer(self, layer_name):
        if False:
            print('Hello World!')
        return self.keras_layer_map[layer_name]

    def make_input_layers(self):
        if False:
            print('Hello World!')
        '\n        Extract the ordering of the input layers.\n        '
        self.input_layers = []
        if hasattr(self.model, 'input_layers'):
            input_keras_layers = self.model.input_layers[:]
            self.input_layers = [None] * len(input_keras_layers)
            for layer in self.layer_list:
                keras_layer = self.keras_layer_map[layer]
                if isinstance(keras_layer, _keras.engine.topology.InputLayer):
                    if keras_layer in input_keras_layers:
                        idx = input_keras_layers.index(keras_layer)
                        self.input_layers[idx] = layer
        elif len(self.model.inbound_nodes) <= 1:
            for ts in _to_list(self.model.input):
                for l in self.layer_list:
                    kl = self.keras_layer_map[l]
                    if isinstance(kl, _keras.engine.topology.InputLayer) and kl.input == ts:
                        self.input_layers.append(l)
        else:
            raise ValueError('Input values cannot be identified.')

    def make_output_layers(self):
        if False:
            print('Hello World!')
        '\n        Extract the ordering of output layers.\n        '
        self.output_layers = []
        for layer in self.layer_list:
            if len(self.get_successors(layer)) == 0:
                self.output_layers.append(layer)

    def get_input_layers(self):
        if False:
            for i in range(10):
                print('nop')
        return self.input_layers

    def get_output_layers(self):
        if False:
            print('Hello World!')
        return self.output_layers

    def generate_blob_names(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate blob names for each one of the edge.  At this time, Keras does not\n        support "fork" operation (a layer with more than 1 blob output). So we just\n        use names of the src layer to identify a blob.  We also assume all neural\n        networks are singly-connected graphs - which should be the case.\n        '
        for layer in self.layer_list:
            keras_layer = self.keras_layer_map[layer]
            if not isinstance(keras_layer, _keras.engine.topology.InputLayer):
                preds = self.get_predecessors(layer)
                for pred in preds:
                    blob_name = pred + '_output'
                    _insert_to_dict(self.layers_inputs, layer, blob_name)
                blob_name = layer + '_output'
                _insert_to_dict(self.layers_outputs, layer, blob_name)

    def get_layer_blobs(self, layer):
        if False:
            for i in range(10):
                print('nop')
        keras_layer = self.keras_layer_map[layer]
        if isinstance(keras_layer, _keras.engine.topology.InputLayer):
            return (None, None)
        else:
            input_blobs = self.layers_inputs[layer]
            output_blobs = self.layers_outputs[layer]
            if layer in self.layers_optional_inputs:
                input_blobs += self.layers_optional_inputs[layer]
            if layer in self.layers_optional_outputs:
                output_blobs += self.layers_optional_outputs[layer]
            return (input_blobs, output_blobs)

    def reset_model_input_names(self, new_names):
        if False:
            i = 10
            return i + 15
        if new_names is None:
            return
        if len(new_names) != len(self.input_layers):
            print('Input name length mismatch')
            return
        for (i, in_layer) in enumerate(self.input_layers):
            old_blob_name = in_layer + '_output'
            new_blob_name = new_names[i]
            succs = self.get_successors(in_layer)
            for succ in succs:
                idx = self.layers_inputs[succ].index(old_blob_name)
                self.layers_inputs[succ][idx] = new_blob_name

    def reset_model_output_names(self, new_names):
        if False:
            return 10
        if new_names is None:
            return
        if len(new_names) != len(self.output_layers):
            print('Output name length mismatch')
            return
        for (i, out_layer) in enumerate(self.output_layers):
            new_blob_name = new_names[i]
            self.layers_outputs[out_layer][0] = new_blob_name

    def add_recurrent_optionals(self):
        if False:
            return 10
        for layer in self.layer_list:
            keras_layer = self.keras_layer_map[layer]
            if type(keras_layer) in _KERAS_RECURRENT_LAYERS:
                if not isinstance(keras_layer, _keras.layers.wrappers.Bidirectional):
                    hidden_size = keras_layer.output_dim
                else:
                    hidden_size = keras_layer.forward_layer.output_dim
                h_in_name = layer + '_h_in'
                h_out_name = layer + '_h_out'
                self.optional_inputs.append((h_in_name, hidden_size))
                self.optional_outputs.append((h_out_name, hidden_size))
                _insert_to_dict(self.layers_optional_inputs, layer, h_in_name)
                _insert_to_dict(self.layers_optional_outputs, layer, h_out_name)
                if isinstance(keras_layer, _keras.layers.recurrent.LSTM):
                    c_in_name = layer + '_c_in'
                    c_out_name = layer + '_c_out'
                    self.optional_inputs.append((c_in_name, hidden_size))
                    self.optional_outputs.append((c_out_name, hidden_size))
                    _insert_to_dict(self.layers_optional_inputs, layer, c_in_name)
                    _insert_to_dict(self.layers_optional_outputs, layer, c_out_name)
                elif isinstance(keras_layer, _keras.layers.wrappers.Bidirectional):
                    c_in_name = layer + '_c_in'
                    c_out_name = layer + '_c_out'
                    h_in_name_rev = layer + '_h_in_rev'
                    c_in_name_rev = layer + '_c_in_rev'
                    h_out_name_rev = layer + '_h_out_rev'
                    c_out_name_rev = layer + '_c_out_rev'
                    self.optional_inputs.append((c_in_name, hidden_size))
                    self.optional_outputs.append((c_out_name, hidden_size))
                    self.optional_inputs.append((h_in_name_rev, hidden_size))
                    self.optional_inputs.append((c_in_name_rev, hidden_size))
                    self.optional_outputs.append((h_out_name_rev, hidden_size))
                    self.optional_outputs.append((c_out_name_rev, hidden_size))
                    _insert_to_dict(self.layers_optional_inputs, layer, c_in_name)
                    _insert_to_dict(self.layers_optional_outputs, layer, c_out_name)
                    _insert_to_dict(self.layers_optional_inputs, layer, h_in_name_rev)
                    _insert_to_dict(self.layers_optional_inputs, layer, c_in_name_rev)
                    _insert_to_dict(self.layers_optional_outputs, layer, h_out_name_rev)
                    _insert_to_dict(self.layers_optional_outputs, layer, c_out_name_rev)

    def _get_first_embedded_model(self):
        if False:
            while True:
                i = 10
        for (idx, layer) in enumerate(self.layer_list):
            keras_layer = self.keras_layer_map[layer]
            if isinstance(keras_layer, _keras.models.Sequential) or isinstance(keras_layer, _keras.models.Model):
                return idx
        return -1

    def _get_first_shared_layer(self):
        if False:
            while True:
                i = 10
        for (idx, layer) in enumerate(self.layer_list):
            if not isinstance(self.keras_layer_map[layer], _keras.layers.Merge) and len(self.get_predecessors(layer)) > 1:
                return idx
        return -1

    def _get_first_layer_of_type(self, layer_type):
        if False:
            while True:
                i = 10
        for (idx, layer) in enumerate(self.layer_list):
            keras_layer = self.keras_layer_map[layer]
            if isinstance(keras_layer, layer_type):
                return idx
        return -1

    def _add_edge(self, src, snk):
        if False:
            i = 10
            return i + 15
        if src not in self.edge_map:
            self.edge_map[src] = []
        if snk not in self.edge_map[src]:
            self.edge_map[src].append(snk)
        if snk not in self.reverse_edge_map:
            self.reverse_edge_map[snk] = []
        if src not in self.reverse_edge_map[snk]:
            self.reverse_edge_map[snk].append(src)

    def _remove_edge(self, src, snk):
        if False:
            while True:
                i = 10
        self.edge_map[src].remove(snk)
        if len(self.edge_map[src]) == 0:
            self.edge_map.pop(src)
        self.reverse_edge_map[snk].remove(src)
        if len(self.reverse_edge_map[snk]) == 0:
            self.reverse_edge_map.pop(snk)

    def _remove_layer(self, layer):
        if False:
            i = 10
            return i + 15
        '\n        remove the layer and its input/output edges\n        '
        successors = self.get_successors(layer)
        predecessors = self.get_predecessors(layer)
        for succ in successors:
            self._remove_edge(layer, succ)
        for pred in predecessors:
            self._remove_edge(pred, layer)
        self.keras_layer_map.pop(layer)
        self.layer_list.remove(layer)

    def _remove_layer_and_reconnect(self, layer):
        if False:
            print('Hello World!')
        '\n        remove the layer, and reconnect each of its predecessor to each of its successor\n        '
        successors = self.get_successors(layer)
        predecessors = self.get_predecessors(layer)
        for succ in successors:
            self._remove_edge(layer, succ)
        for pred in predecessors:
            self._remove_edge(pred, layer)
        for pred in predecessors:
            for succ in successors:
                self._add_edge(pred, succ)
        self.layer_list.remove(layer)
        self.keras_layer_map.pop(layer)

    def _remove_old_edges(self, layer):
        if False:
            for i in range(10):
                print('nop')
        predecessors = self.get_predecessors(layer)
        successors = self.get_successors(layer)
        for pred in predecessors:
            self._remove_edge(pred, layer)
        for succ in successors:
            self._remove_edge(layer, succ)

    def _remove_layers_of_type(self, layer_type):
        if False:
            while True:
                i = 10
        idx = self._get_first_layer_of_type(layer_type)
        while idx >= 0:
            layer = self.layer_list[idx]
            self._remove_layer_and_reconnect(layer)
            idx = self._get_first_layer_of_type(layer_type)

    def remove_skip_layers(self, skip_layers):
        if False:
            return 10
        for skip_layer in skip_layers:
            self._remove_layers_of_type(skip_layer)

    def remove_internal_input_layers(self):
        if False:
            for i in range(10):
                print('nop')
        (idx, nb_layers) = (0, len(self.layer_list))
        while idx < nb_layers:
            layer = self.layer_list[idx]
            keras_layer = self.keras_layer_map[layer]
            if isinstance(keras_layer, _keras.engine.topology.InputLayer) and len(self.get_predecessors(layer)) > 0:
                self._remove_layer_and_reconnect(layer)
                idx -= 1
                nb_layers -= 1
            idx += 1

    def _insert_layer_after(self, layer_idx, new_layer, new_keras_layer):
        if False:
            print('Hello World!')
        "\n        Insert the new_layer after layer, whose position is layer_idx. The new layer's\n        parameter is stored in a Keras layer called new_keras_layer\n        "
        layer = self.layer_list[layer_idx]
        self.layer_list.insert(layer_idx + 1, new_layer)
        self.keras_layer_map[new_layer] = new_keras_layer
        successors = self.get_successors(layer)
        self._add_edge(layer, new_layer)
        for succ in successors:
            self._add_edge(new_layer, succ)
            self._remove_edge(layer, succ)

    def _insert_layer_between(self, src, snk, new_layer, new_keras_layer):
        if False:
            return 10
        "\n        Insert the new_layer before layer, whose position is layer_idx. The new layer's\n        parameter is stored in a Keras layer called new_keras_layer\n        "
        if snk is None:
            insert_pos = self.layer_list.index(src) + 1
        else:
            insert_pos = self.layer_list.index(snk)
        self.layer_list.insert(insert_pos, new_layer)
        self.keras_layer_map[new_layer] = new_keras_layer
        if src is None:
            self._add_edge(new_layer, snk)
        elif snk is None:
            self._add_edge(src, new_layer)
        else:
            self._add_edge(src, new_layer)
            self._add_edge(new_layer, snk)
            self._remove_edge(src, snk)

    def defuse_activation(self):
        if False:
            return 10
        '\n        Defuse the fused activation layers in the network.\n        '
        (idx, nb_layers) = (0, len(self.layer_list))
        while idx < nb_layers:
            layer = self.layer_list[idx]
            k_layer = self.keras_layer_map[layer]
            if isinstance(k_layer, _keras.layers.TimeDistributed):
                k_layer = k_layer.layer
            if isinstance(k_layer, _keras.layers.convolutional.Convolution2D) or isinstance(k_layer, _keras.layers.convolutional.Convolution1D) or isinstance(k_layer, _keras.layers.core.Dense):
                import six
                if six.PY2:
                    func_name = k_layer.activation.func_name
                else:
                    func_name = k_layer.activation.__name__
                if func_name != 'linear':
                    new_layer = layer + '__activation__'
                    new_keras_layer = _keras.layers.core.Activation(func_name)
                    self._insert_layer_after(idx, new_layer, new_keras_layer)
                    idx += 1
                    nb_layers += 1
            idx += 1

    def is_activation(self, layer):
        if False:
            return 10
        keras_layer = self.keras_layer_map[layer]
        for activation_type in _KERAS_ACTIVATION_LAYERS:
            if isinstance(keras_layer, activation_type):
                return True
        return False

    def is_1d_layer(self, layer):
        if False:
            for i in range(10):
                print('nop')
        keras_layer = self.keras_layer_map[layer]
        for layer_type in _KERAS_LAYERS_1D:
            if isinstance(keras_layer, layer_type):
                return True
        return False

    def _get_1d_interface_edges(self):
        if False:
            i = 10
            return i + 15
        "\n        Get edges that represents transition from not 1D to 1D, and 1D to not 1D\n        A 'in_edge e(u,v)' means u operates on non-1D blobs, but v operates on 1D blobs.\n        An 'out_edge e(u,v)' means u operates on 1D blobs, but v operates on non-1D blobs.\n        "
        in_edges = []
        for layer in self.layer_list:
            if not self.is_1d_layer(layer):
                continue
            preds = self.get_predecessors(layer)
            if len(preds) == 0:
                in_edges.append((None, layer))
            else:
                (u, v) = (preds[0], layer)
                while u != None and (self.is_activation(u) or type(u) in _KERAS_NORMALIZATION_LAYERS):
                    preds = self.get_predecessors(u)
                    v = u
                    u = preds[0] if len(preds) > 0 else None
                if u is None or not self.is_1d_layer(u):
                    in_edges.append((u, v))
        out_edges = []
        for layer in self.layer_list:
            if not self.is_1d_layer(layer):
                continue
            succs = self.get_successors(layer)
            if len(succs) == 0:
                out_edges.append((layer, None))
            elif not self.is_activation(succs[0]):
                for succ in succs:
                    if not self.is_1d_layer(succ):
                        out_edges.append((layer, succ))
            else:
                act_layer = succs[0]
                succs = self.get_successors(act_layer)
                if len(succs) == 0:
                    out_edges.append((act_layer, None))
                else:
                    for succ in succs:
                        if not self.is_1d_layer(succ):
                            out_edges.append((act_layer, succ))
        return (in_edges, out_edges)

    def insert_1d_permute_layers(self):
        if False:
            i = 10
            return i + 15
        '\n        Insert permutation layers before a 1D start point or after 1D end point\n        '
        (idx, nb_layers) = (0, len(self.layer_list))
        (in_edges, out_edges) = self._get_1d_interface_edges()
        for in_edge in in_edges:
            (src, snk) = in_edge
            if src is None:
                permute_layer = '_permute_' + snk
            else:
                permute_layer = src + '_permute_' + snk
            keras_permute = _keras.layers.Permute(dims=(3, 1, 2, 0))
            self._insert_layer_between(src, snk, permute_layer, keras_permute)
        for out_edge in out_edges:
            (src, snk) = out_edge
            if snk is None:
                permute_layer = src + '_permute_'
            else:
                permute_layer = src + '_permute_' + snk
            keras_permute = _keras.layers.Permute(dims=(3, 1, 2, 0))
            self._insert_layer_between(src, snk, permute_layer, keras_permute)

    def insert_permute_for_spatial_bn(self):
        if False:
            for i in range(10):
                print('nop')
        spatial_bn_layers = []
        for layer in self.layer_list:
            keras_layer = self.keras_layer_map[layer]
            if isinstance(keras_layer, _keras.layers.BatchNormalization) and len(keras_layer.input_shape) == 4:
                if keras_layer.axis == 1 or keras_layer.axis == 2:
                    spatial_bn_layers.append(layer)
        for sbn in spatial_bn_layers:
            axis = self.keras_layer_map[sbn].axis
            dims = (0, 2, 1, 3) if axis == 1 else (0, 3, 2, 1)
            pred = self.get_predecessors(sbn)[0]
            permute_layer = pred + '_permute_' + sbn
            keras_permute = _keras.layers.Permute(dims=dims)
            self._insert_layer_between(pred, sbn, permute_layer, keras_permute)
            succs = self.get_successors(sbn)
            if len(succs) == 0:
                permute_layer = sbn + '_permute_'
                keras_permute = _keras.layers.Permute(dims=dims)
                self._insert_layer_between(sbn, None, permute_layer, keras_permute)
            else:
                for succ in succs:
                    permute_layer = sbn + '_permute_' + succ
                    keras_permute = _keras.layers.Permute(dims=dims)
                    self._insert_layer_between(sbn, succ, permute_layer, keras_permute)

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        model = self.model
        if not (type(model) == _keras.models.Sequential or type(model) == _keras.models.Model):
            raise TypeError('Keras layer of type %s is not supported.' % type(model))
            self = None
            return
        for (i, layer) in enumerate(model.layers):
            for node in layer.inbound_nodes:
                for pred in node.inbound_layers:
                    if pred.name not in self.layer_list:
                        self.layer_list.append(pred.name)
                        self.keras_layer_map[pred.name] = pred
                    self._add_edge(pred.name, layer.name)
            self.layer_list.append(layer.name)
            self.keras_layer_map[layer.name] = layer
        idx = self._get_first_shared_layer()
        while idx >= 0:
            layer = self.layer_list[idx]
            keras_layer = self.keras_layer_map[layer]
            predecessors = self.reverse_edge_map[layer]
            successors = self.edge_map[layer]
            new_layers = [layer + '_' + str(i) for i in range(len(predecessors))]
            self.layer_list[idx:idx + 1] = new_layers
            for (i, new_layer) in enumerate(new_layers):
                self.edge_map[new_layer] = []
                self.reverse_edge_map[new_layer] = []
                self.keras_layer_map[new_layer] = keras_layer
                pred = predecessors[i]
                self._add_edge(pred, new_layer)
                for succ in successors:
                    self._add_edge(new_layer, succ)
            self._remove_old_edges(layer)
            self.keras_layer_map.pop(layer)
            idx = self._get_first_shared_layer()
        idx = self._get_first_embedded_model()
        while idx >= 0:
            embedded_model = self.layer_list[idx]
            embedded_keras_model = self.keras_layer_map[embedded_model]
            embedded_graph = NetGraph(embedded_keras_model)
            embedded_graph.build()
            embedded_layer_list = embedded_graph.layer_list
            new_layer_list = []
            for embedded_layer_name in embedded_layer_list:
                new_layer_name = embedded_model + '_' + embedded_layer_name
                new_layer_list.append(new_layer_name)
                self.keras_layer_map[new_layer_name] = embedded_graph.keras_layer_map[embedded_layer_name]
                embedded_successors = embedded_graph.get_successors(embedded_layer_name)
                for embed_succ_name in embedded_successors:
                    new_embed_succ_name = embedded_model + '_' + embed_succ_name
                    self._add_edge(new_layer_name, new_embed_succ_name)
                embedded_predecessors = embedded_graph.get_predecessors(embedded_layer_name)
                for embed_pred_name in embedded_predecessors:
                    new_embed_pred_name = embedded_model + '_' + embed_pred_name
                    self._add_edge(new_embed_pred_name, new_layer_name)
            self.layer_list[idx + 1:idx + 1] = new_layer_list
            predecessors = self.get_predecessors(embedded_model)
            embedded_inputs = embedded_graph.get_input_layers()
            for (i, pred) in enumerate(predecessors):
                embed_input = embedded_inputs[i]
                new_embed_input = embedded_model + '_' + embed_input
                self._add_edge(pred, new_embed_input)
            embedded_outputs = embedded_graph.get_output_layers()
            successors = self.get_successors(embedded_model)
            for (i, succ) in enumerate(successors):
                embed_output = embedded_outputs[i]
                new_embed_output = embedded_model + '_' + embed_output
                self._add_edge(new_embed_output, succ)
            self._remove_layer(embedded_model)
            idx = self._get_first_embedded_model()
        self.make_input_layers()
        self.make_output_layers()

    def print_layer_list(self):
        if False:
            print('Hello World!')
        print('\n')
        print('layer_list')
        print(self.layer_list)

    def print_edge_map(self):
        if False:
            while True:
                i = 10
        print('\n')
        print('edge map:')
        for src in self.edge_map:
            for snk in self.edge_map[src]:
                print('  ', src, '-->', snk)

    def print_reverse_edge_map(self):
        if False:
            print('Hello World!')
        print('\n')
        print('reverse edge map: ')
        for snk in self.reverse_edge_map:
            for src in self.reverse_edge_map[snk]:
                print('  ', snk, '<--', src)

    def print_mapping(self):
        if False:
            return 10
        print('\nmapping:')
        for key in self.keras_layer_map:
            print(key, '-->', self.keras_layer_map[key], '(', self.keras_layer_map[key].name, ')')

    def print_all(self):
        if False:
            while True:
                i = 10
        print('=' * 80)
        self.print_layer_list()
        self.print_edge_map()
        self.print_reverse_edge_map()
        self.print_mapping()
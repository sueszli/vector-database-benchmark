from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .._tf_model import TensorFlowModel
import turicreate.toolkits._tf_utils as _utils
from turicreate._deps.minimal_package import _minimal_package_import_check

def _lazy_import_tensorflow():
    if False:
        i = 10
        return i + 15
    _utils.suppress_tensorflow_warnings()
    _tf = _minimal_package_import_check('tensorflow.compat.v1')
    return _tf

class SoundClassifierTensorFlowModel(TensorFlowModel):

    def __init__(self, num_inputs, num_classes, custom_layer_sizes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Defines the TensorFlow model, loss, optimisation and accuracy.\n        '
        _tf = _lazy_import_tensorflow()
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.custom_layer_sizes = custom_layer_sizes
        self.gpu_policy = _utils.TensorFlowGPUPolicy()
        self.gpu_policy.start()
        self.sc_graph = _tf.Graph()
        self.sess = _tf.Session(graph=self.sc_graph)
        self.is_initialized = False

    def __del__(self):
        if False:
            print('Hello World!')
        self.sess.close()
        self.gpu_policy.stop()

    @staticmethod
    def _build_network(x, weights, biases):
        if False:
            for i in range(10):
                print('nop')
        _tf = _lazy_import_tensorflow()
        for i in range(len(weights.keys())):
            weight_name = 'sound_dense{}_weight'.format(i)
            bias_name = 'sound_dense{}_bias'.format(i)
            if i == 0:
                curr_dense = _tf.nn.xw_plus_b(x, weights=weights[weight_name], biases=biases[bias_name])
            else:
                curr_dense = _tf.nn.xw_plus_b(curr_dense, weights=weights[weight_name], biases=biases[bias_name])
            if i == len(weights.keys()) - 1:
                out = _tf.nn.softmax(curr_dense)
            else:
                curr_dense = _tf.nn.relu(curr_dense)
        return (out, curr_dense)

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.is_initialized
        with self.sc_graph.as_default():
            self.init_sound_classifier_graph()
        self.is_initialized = True

    def init_sound_classifier_graph(self):
        if False:
            for i in range(10):
                print('nop')
        _tf = _lazy_import_tensorflow()
        self.x = _tf.placeholder('float', [None, self.num_inputs])
        self.y = _tf.placeholder('float', [None, self.num_classes])
        initializer = _tf.keras.initializers.glorot_uniform()
        weights = {}
        biases = {}
        self.names_of_layers = []
        for (i, cur_layer_size) in enumerate(self.custom_layer_sizes):
            weight_name = 'sound_dense{}_weight'.format(i)
            bias_name = 'sound_dense{}_bias'.format(i)
            self.names_of_layers.append('dense{}'.format(i))
            out_units = cur_layer_size
            if i == 0:
                in_units = self.num_inputs
            weights[weight_name] = _tf.Variable(initializer([in_units, out_units]), name=weight_name)
            biases[bias_name] = _tf.Variable(initializer([out_units]), name=bias_name)
            in_units = out_units
        i += 1
        weight_name = 'sound_dense{}_weight'.format(i)
        bias_name = 'sound_dense{}_bias'.format(i)
        self.names_of_layers.append('dense{}'.format(i))
        weights[weight_name] = _tf.Variable(initializer([in_units, self.num_classes]), name=weight_name)
        biases[bias_name] = _tf.Variable(initializer([self.num_classes]), name=bias_name)
        (self.predictions, curr_dense) = SoundClassifierTensorFlowModel._build_network(self.x, weights, biases)
        self.cost = _tf.reduce_mean(_tf.nn.softmax_cross_entropy_with_logits_v2(logits=curr_dense, labels=self.y))
        self.optimizer = _tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(self.cost)
        correct_prediction = _tf.equal(_tf.argmax(self.predictions, 1), _tf.argmax(self.y, 1))
        self.accuracy = _tf.reduce_mean(_tf.cast(correct_prediction, 'float'))
        self.sess.run(_tf.global_variables_initializer())

    def train(self, data, label):
        if False:
            print('Hello World!')
        _tf = _lazy_import_tensorflow()
        assert self.is_initialized
        data_shape = data.shape[0]
        (_, final_train_loss, final_train_accuracy) = self.sess.run([self.optimizer, self.cost, self.accuracy], feed_dict={self.x: data.reshape((data_shape, 12288)), self.y: _tf.keras.utils.to_categorical(label, self.num_classes).reshape((data_shape, self.num_classes))})
        result = {'accuracy': final_train_accuracy, 'loss': final_train_loss}
        return result

    def evaluate(self, data, label):
        if False:
            while True:
                i = 10
        assert self.is_initialized
        _tf = _lazy_import_tensorflow()
        data_shape = data.shape[0]
        (pred_probs, final_accuracy) = self.sess.run([self.predictions, self.accuracy], feed_dict={self.x: data.reshape((data_shape, 12288)), self.y: _tf.keras.utils.to_categorical(label, self.num_classes).reshape((data_shape, self.num_classes))})
        result = {'accuracy': final_accuracy, 'predictions': pred_probs}
        return result

    def predict(self, data):
        if False:
            i = 10
            return i + 15
        assert self.is_initialized
        data_shape = data.shape[0]
        pred_probs = self.sess.run(self.predictions, feed_dict={self.x: data.reshape((data_shape, 12288))})
        return pred_probs

    def export_weights(self):
        if False:
            i = 10
            return i + 15
        '\n        Retrieve weights from the TF model, converts to the CoreML format\n        and stores in a list of dictionaries.\n\n        Returns\n        -------\n        layers : list\n            List of dictionaries of weights and activations, where\n            the key, for each element of the list, is `weight`, `bias`\n            and `act` and the value is the respective weight of type\n            `numpy.ndarray` converted to the CoreML format and the\n            respective activation applied to the layer.\n        '
        assert self.is_initialized
        _tf = _lazy_import_tensorflow()
        with self.sc_graph.as_default():
            layer_names = _tf.trainable_variables()
            layer_weights = self.sess.run(layer_names)
        layer_dict = {}
        for (var, val) in zip(layer_names, layer_weights):
            layer_dict[var.name] = val
        layers = []
        for (i, name) in enumerate(self.names_of_layers):
            weight_name = 'sound_{}_weight:0'.format(name)
            bias_name = 'sound_{}_bias:0'.format(name)
            layer = {}
            layer['weight'] = layer_dict[weight_name].transpose(1, 0)
            layer['bias'] = layer_dict[bias_name]
            if i == len(self.names_of_layers) - 1:
                layer['act'] = None
            else:
                layer['act'] = 'relu'
            layers.append(layer)
        return layers

    def get_weights(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns\n        -------\n                : dict\n                Containing model weights and shapes\n                {'data': weight data dict, 'shapes': weight shapes dict}\n                Model is saved in CoreML format, hence dense weights and\n                shapes are transposed.\n\n        "
        assert self.is_initialized
        _tf = _lazy_import_tensorflow()
        with self.sc_graph.as_default():
            layer_names = _tf.trainable_variables()
            layer_weights = self.sess.run(layer_names)
        data = {}
        shapes = {}
        for (var, val) in zip(layer_names, layer_weights):
            layer_name = var.name[:-2]
            if 'bias' in layer_name:
                data[layer_name] = val
            else:
                data[layer_name] = val.transpose(1, 0)
            shapes[layer_name] = val.shape[::-1]
        return {'data': data, 'shapes': shapes}

    def load_weights(self, net_params):
        if False:
            return 10
        '\n        TensorFlow model is assigned weights from `net_params` dictionary.\n        `net_params` contains weights in CoreML format. The dense layers\n        need to be transposed to match TF format.\n\n        '
        _tf = _lazy_import_tensorflow()
        with self.sc_graph.as_default():
            (weights, biases) = ({}, {})
            for (cur_name, cur_layer) in net_params['data'].items():
                if 'bias' in cur_name:
                    biases[cur_name] = _tf.Variable(cur_layer.astype('float32'), name=cur_name)
                else:
                    assert 'weight' in cur_name
                    weights[cur_name] = _tf.Variable(cur_layer.transpose(1, 0).astype('float32'), name=cur_name)
            self.x = _tf.placeholder('float', [None, self.num_inputs])
            (self.predictions, _) = SoundClassifierTensorFlowModel._build_network(self.x, weights, biases)
            self.sess.run(_tf.global_variables_initializer())
        self.is_initialized = True
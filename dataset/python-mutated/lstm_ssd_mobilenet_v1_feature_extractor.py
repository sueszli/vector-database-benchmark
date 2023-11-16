"""LSTMSSDFeatureExtractor for MobilenetV1 features."""
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from lstm_object_detection.lstm import lstm_cells
from lstm_object_detection.lstm import rnn_decoder
from lstm_object_detection.meta_architectures import lstm_ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import mobilenet_v1
slim = tf.contrib.slim

class LSTMSSDMobileNetV1FeatureExtractor(lstm_ssd_meta_arch.LSTMSSDFeatureExtractor):
    """LSTM Feature Extractor using MobilenetV1 features."""

    def __init__(self, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams_fn, reuse_weights=None, use_explicit_padding=False, use_depthwise=True, override_base_feature_extractor_hyperparams=False, lstm_state_depth=256):
        if False:
            print('Hello World!')
        'Initializes instance of MobileNetV1 Feature Extractor for LSTMSSD Models.\n\n    Args:\n      is_training: A boolean whether the network is in training mode.\n      depth_multiplier: A float depth multiplier for feature extractor.\n      min_depth: A number representing minimum feature extractor depth.\n      pad_to_multiple: The nearest multiple to zero pad the input height and\n        width dimensions to.\n      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d\n        and separable_conv2d ops in the layers that are added on top of the\n        base feature extractor.\n      reuse_weights: Whether to reuse variables. Default is None.\n      use_explicit_padding: Whether to use explicit padding when extracting\n        features. Default is False.\n      use_depthwise: Whether to use depthwise convolutions. Default is True.\n      override_base_feature_extractor_hyperparams: Whether to override\n        hyperparameters of the base feature extractor with the one from\n        `conv_hyperparams_fn`.\n      lstm_state_depth: An integter of the depth of the lstm state.\n    '
        super(LSTMSSDMobileNetV1FeatureExtractor, self).__init__(is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams_fn, reuse_weights, use_explicit_padding, use_depthwise, override_base_feature_extractor_hyperparams)
        self._feature_map_layout = {'from_layer': ['Conv2d_13_pointwise_lstm', '', '', '', ''], 'layer_depth': [-1, 512, 256, 256, 128], 'use_explicit_padding': self._use_explicit_padding, 'use_depthwise': self._use_depthwise}
        self._base_network_scope = 'MobilenetV1'
        self._lstm_state_depth = lstm_state_depth

    def create_lstm_cell(self, batch_size, output_size, state_saver, state_name):
        if False:
            for i in range(10):
                print('nop')
        'Create the LSTM cell, and initialize state if necessary.\n\n    Args:\n      batch_size: input batch size.\n      output_size: output size of the lstm cell, [width, height].\n      state_saver: a state saver object with methods `state` and `save_state`.\n      state_name: string, the name to use with the state_saver.\n\n    Returns:\n      lstm_cell: the lstm cell unit.\n      init_state: initial state representations.\n      step: the step\n    '
        lstm_cell = lstm_cells.BottleneckConvLSTMCell(filter_size=(3, 3), output_size=output_size, num_units=max(self._min_depth, self._lstm_state_depth), activation=tf.nn.relu6, visualize_gates=False)
        if state_saver is None:
            init_state = lstm_cell.init_state(state_name, batch_size, tf.float32)
            step = None
        else:
            step = state_saver.state(state_name + '_step')
            c = state_saver.state(state_name + '_c')
            h = state_saver.state(state_name + '_h')
            init_state = (c, h)
        return (lstm_cell, init_state, step)

    def extract_features(self, preprocessed_inputs, state_saver=None, state_name='lstm_state', unroll_length=5, scope=None):
        if False:
            return 10
        'Extracts features from preprocessed inputs.\n\n    The features include the base network features, lstm features and SSD\n    features, organized in the following name scope:\n\n    <parent scope>/MobilenetV1/...\n    <parent scope>/LSTM/...\n    <parent scope>/FeatureMaps/...\n\n    Args:\n      preprocessed_inputs: A [batch, height, width, channels] float tensor\n        representing a batch of consecutive frames from video clips.\n      state_saver: A state saver object with methods `state` and `save_state`.\n      state_name: A python string for the name to use with the state_saver.\n      unroll_length: The number of steps to unroll the lstm.\n      scope: The scope for the base network of the feature extractor.\n\n    Returns:\n      A list of tensors where the ith tensor has shape [batch, height_i,\n      width_i, depth_i]\n    '
        preprocessed_inputs = shape_utils.check_min_image_dim(33, preprocessed_inputs)
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=self._is_training)):
            with slim.arg_scope(self._conv_hyperparams_fn()) if self._override_base_feature_extractor_hyperparams else context_manager.IdentityContextManager():
                with slim.arg_scope([slim.batch_norm], fused=False):
                    with tf.variable_scope(scope, self._base_network_scope, reuse=self._reuse_weights) as scope:
                        (net, image_features) = mobilenet_v1.mobilenet_v1_base(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple), final_endpoint='Conv2d_13_pointwise', min_depth=self._min_depth, depth_multiplier=self._depth_multiplier, scope=scope)
        with slim.arg_scope(self._conv_hyperparams_fn()):
            with slim.arg_scope([slim.batch_norm], fused=False, is_training=self._is_training):
                batch_size = net.shape[0].value / unroll_length
                with tf.variable_scope('LSTM', reuse=self._reuse_weights) as lstm_scope:
                    (lstm_cell, init_state, _) = self.create_lstm_cell(batch_size, (net.shape[1].value, net.shape[2].value), state_saver, state_name)
                    net_seq = list(tf.split(net, unroll_length))
                    c_ident = tf.identity(init_state[0], name='lstm_state_in_c')
                    h_ident = tf.identity(init_state[1], name='lstm_state_in_h')
                    init_state = (c_ident, h_ident)
                    (net_seq, states_out) = rnn_decoder.rnn_decoder(net_seq, init_state, lstm_cell, scope=lstm_scope)
                    batcher_ops = None
                    self._states_out = states_out
                    if state_saver is not None:
                        self._step = state_saver.state('%s_step' % state_name)
                        batcher_ops = [state_saver.save_state('%s_c' % state_name, states_out[-1][0]), state_saver.save_state('%s_h' % state_name, states_out[-1][1]), state_saver.save_state('%s_step' % state_name, self._step + 1)]
                    with tf_ops.control_dependencies(batcher_ops):
                        image_features['Conv2d_13_pointwise_lstm'] = tf.concat(net_seq, 0)
                    tf.identity(states_out[-1][0], name='lstm_state_out_c')
                    tf.identity(states_out[-1][1], name='lstm_state_out_h')
                with tf.variable_scope('FeatureMaps', reuse=self._reuse_weights):
                    feature_maps = feature_map_generators.multi_resolution_feature_maps(feature_map_layout=self._feature_map_layout, depth_multiplier=self._depth_multiplier, min_depth=self._min_depth, insert_1x1_conv=True, image_features=image_features)
        return feature_maps.values()
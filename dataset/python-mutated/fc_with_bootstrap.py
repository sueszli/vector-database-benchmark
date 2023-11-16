import math
import numpy as np
from caffe2.python import core, schema
from caffe2.python.helpers.arg_scope import get_current_scope
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.layers.sampling_trainable_mixin import SamplingTrainableMixin

def get_fc_predictor_version(fc_version):
    if False:
        for i in range(10):
            print('nop')
    assert fc_version in ['fp32'], 'Only support fp32 for the fully connected layer in the predictor net, the provided FC precision is {}'.format(fc_version)
    return fc_version

class FCWithBootstrap(SamplingTrainableMixin, ModelLayer):

    def __init__(self, model, input_record, output_dims, num_bootstrap, weight_init=None, bias_init=None, weight_optim=None, bias_optim=None, name='fc_with_bootstrap', weight_reg=None, bias_reg=None, clip_param=None, axis=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), 'Incorrect input type {}'.format(input_record)
        assert len(input_record.field_types()[0].shape) > 0, 'FC expects limited dimensions of the input tensor'
        assert axis >= 1, 'axis {} should >= 1.'.format(axis)
        self.axis = axis
        input_dims = np.prod(input_record.field_types()[0].shape[axis - 1:])
        assert input_dims > 0, 'FC expects input dimensions > 0, got {}'.format(input_dims)
        self.clip_args = None
        self.num_bootstrap = num_bootstrap
        self.input_dims = input_dims
        self.bootstrapped_FCs = []
        self.batch_size = None
        self.output_dim_vec = None
        self.lower_bound = None
        self.upper_bound = None
        if clip_param is not None:
            assert len(clip_param) == 2, 'clip_param must be a tuple / list of length 2 and in the form of (clip_min, clip max)'
            (clip_min, clip_max) = clip_param
            assert clip_min is not None or clip_max is not None, 'clip_min, and clip_max in clip_param cannot both be None'
            assert (clip_min is None or clip_max is None) or clip_min < clip_max, 'clip_param = [clip_min, clip_max] must have clip_min < clip_max'
            self.clip_args = {}
            if clip_min is not None:
                self.clip_args['min'] = clip_min
            if clip_max is not None:
                self.clip_args['max'] = clip_max
        scale = math.sqrt(1.0 / input_dims)
        weight_init = weight_init if weight_init else ('UniformFill', {'min': -scale, 'max': scale})
        bias_init = bias_init if bias_init else ('UniformFill', {'min': -scale, 'max': scale})
        '\n        bootstrapped FCs:\n            Ex: [\n                bootstrapped_weights_blob_1, bootstrapped_bias_blob_1,\n                ...,\n                ...,\n                bootstrapped_weights_blob_b, bootstrapped_bias_blob_b\n                ]\n\n        output_schema:\n            Note: indices will always be on even indices.\n            Ex: Struct(\n                    indices_0_blob,\n                    preds_0_blob,\n                    ...\n                    ...\n                    indices_b_blob,\n                    preds_b_blob\n                )\n        '
        bootstrapped_FCs = []
        output_schema = schema.Struct()
        for i in range(num_bootstrap):
            output_schema += schema.Struct(('bootstrap_iteration_{}/indices'.format(i), self.get_next_blob_reference('bootstrap_iteration_{}/indices'.format(i))), ('bootstrap_iteration_{}/preds'.format(i), self.get_next_blob_reference('bootstrap_iteration_{}/preds'.format(i))))
            self.bootstrapped_FCs.extend([self.create_param(param_name='bootstrap_iteration_{}/w'.format(i), shape=[output_dims, input_dims], initializer=weight_init, optimizer=weight_optim, regularizer=weight_reg), self.create_param(param_name='bootstrap_iteration_{}/b'.format(i), shape=[output_dims], initializer=bias_init, optimizer=bias_optim, regularizer=bias_reg)])
        self.output_schema = output_schema
        if axis == 1:
            output_shape = (output_dims,)
        else:
            output_shape = list(input_record.field_types()[0].shape)[0:axis - 1]
            output_shape = tuple(output_shape + [output_dims])

    def _generate_bootstrapped_indices(self, net, copied_cur_layer, iteration):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            net: the caffe2 net to insert operator\n\n            copied_cur_layer: blob of the bootstrapped features (make sure this\n            blob has a stop_gradient on)\n\n            iteration: the bootstrap interation to generate for. Used to correctly\n            populate the output_schema\n\n        Return:\n            A blob containing the generated indices of shape: (batch_size,)\n        '
        with core.NameScope('bootstrap_iteration_{}'.format(iteration)):
            if iteration == 0:
                input_shape = net.Shape(copied_cur_layer, 'input_shape')
                batch_size_index = net.Const(np.array([0]), 'batch_size_index')
                batch_size = net.Gather([input_shape, batch_size_index], 'batch_size')
                self.batch_size = batch_size
                lower_bound = net.Const(np.array([0]), 'lower_bound', dtype=np.int32)
                offset = net.Const(np.array([1]), 'offset', dtype=np.int32)
                int_batch_size = net.Cast([self.batch_size], 'int_batch_size', to=core.DataType.INT32)
                upper_bound = net.Sub([int_batch_size, offset], 'upper_bound')
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound
            indices = net.UniformIntFill([self.batch_size, self.lower_bound, self.upper_bound], self.output_schema[iteration * 2].field_blobs()[0], input_as_shape=1)
            return indices

    def _bootstrap_ops(self, net, copied_cur_layer, indices, iteration):
        if False:
            for i in range(10):
                print('nop')
        '\n            This method contains all the bootstrapping logic used to bootstrap\n            the features. Only used by the train_net.\n\n            Args:\n                net: the caffe2 net to insert bootstrapping operators\n\n                copied_cur_layer: the blob representing the current features.\n                    Note, this layer should have a stop_gradient on it.\n\n            Returns:\n                bootstrapped_features: blob of bootstrapped version of cur_layer\n                    with same dimensions\n        '
        bootstrapped_features = net.Gather([copied_cur_layer, indices], net.NextScopedBlob('bootstrapped_features_{}'.format(iteration)))
        bootstrapped_features = schema.Scalar((np.float32, self.input_dims), bootstrapped_features)
        return bootstrapped_features

    def _insert_fc_ops(self, net, features, params, outputs, version):
        if False:
            while True:
                i = 10
        '\n        Args:\n            net: the caffe2 net to insert operator\n\n            features: Scalar containing blob of the bootstrapped features or\n            actual cur_layer features\n\n            params: weight and bias for FC\n\n            outputs: the output blobs\n\n            version: support fp32 for now.\n        '
        if version == 'fp32':
            pred_blob = net.FC(features.field_blobs() + params, outputs, axis=self.axis, **self.kwargs)
            return pred_blob
        else:
            raise Exception('unsupported FC type version {}'.format(version))

    def _add_ops(self, net, features, iteration, params, version):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            params: the weight and bias, passed by either add_ops or\n            add_train_ops function\n\n            features: feature blobs to predict on. Can be the actual cur_layer\n            or the bootstrapped_feature blobs.\n\n            version: currently fp32 support only\n        '
        if self.clip_args is not None:
            clipped_params = [net.NextScopedBlob('clipped_%s' % str(p)) for p in params]
            for (p, cp) in zip(params, clipped_params):
                net.Clip([p], [cp], **self.clip_args)
            params = clipped_params
        if self.output_dim_vec is None or len(self.output_dim_vec) == 1:
            self._insert_fc_ops(net=net, features=features, params=params, outputs=[self.output_schema.field_blobs()[iteration * 2 + 1]], version=version)

    def add_ops(self, net):
        if False:
            i = 10
            return i + 15
        '\n            Both the predict net and the eval net will call this function.\n\n            For bootstrapping approach, the goal is to pass the cur_layer feature\n            inputs through all the bootstrapped FCs that are stored under\n            self.bootstrapped_FCs. Return the preds in the same output_schema\n            with dummy indices (because they are not needed).\n        '
        version_info = get_current_scope().get(get_fc_predictor_version.__name__, {'fc_version': 'fp32'})
        predictor_fc_fp_version = version_info['fc_version']
        for i in range(self.num_bootstrap):
            indices = self._generate_bootstrapped_indices(net=net, copied_cur_layer=self.input_record.field_blobs()[0], iteration=i)
            params = self.bootstrapped_FCs[i * 2:i * 2 + 2]
            self._add_ops(net=net, features=self.input_record, params=params, iteration=i, version=predictor_fc_fp_version)

    def add_train_ops(self, net):
        if False:
            print('Hello World!')
        for i in range(self.num_bootstrap):
            indices = self._generate_bootstrapped_indices(net=net, copied_cur_layer=self.input_record.field_blobs()[0], iteration=i)
            bootstrapped_features = self._bootstrap_ops(net=net, copied_cur_layer=self.input_record.field_blobs()[0], indices=indices, iteration=i)
            self._add_ops(net, features=bootstrapped_features, iteration=i, params=self.train_param_blobs[i * 2:i * 2 + 2], version='fp32')

    def get_fp16_compatible_parameters(self):
        if False:
            i = 10
            return i + 15
        if self.output_dim_vec is None or len(self.output_dim_vec) == 1:
            return [blob for (idx, blob) in enumerate(self.bootstrapped_FCs) if idx % 2 == 0]
        else:
            raise Exception('Currently only supports functionality for output_dim_vec == 1')

    @property
    def param_blobs(self):
        if False:
            for i in range(10):
                print('nop')
        if self.output_dim_vec is None or len(self.output_dim_vec) == 1:
            return self.bootstrapped_FCs
        else:
            raise Exception('FCWithBootstrap layer only supports output_dim_vec==1')
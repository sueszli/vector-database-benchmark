from caffe2.python import core, model_helper, schema, scope, utils, muji
from caffe2.python.modeling.parameter_info import ParameterInfo
from caffe2.python.modeling.parameter_sharing import parameter_sharing_context
from caffe2.python.modeling.net_modifier import NetModifier
from caffe2.python.optimizer import get_param_device, Optimizer
from caffe2.python.regularizer import Regularizer, RegularizationBy
from caffe2.python.layers import layers
import logging
import numpy as np
import copy
logger = logging.getLogger(__name__)

class LayerModelHelper(model_helper.ModelHelper):
    """
    Model helper for building models on top of layers abstractions.

    Each layer is the abstraction that is higher level than Operator. Layer
    is responsible for ownership of it's own parameters and can easily be
    instantiated in multiple nets possible with different sets of ops.
    As an example: one can easily instantiate predict and train nets from
    the same set of layers, where predict net will have subset of the
    operators from train net.
    """

    def __init__(self, name, input_feature_schema, trainer_extra_schema, keep_blobs=False, use_attribution=True):
        if False:
            i = 10
            return i + 15
        ' TODO(amalevich): more documnetation on input args\n\n        use_attribution:\n            if True, will generate the atrribution net for feature importance\n            calculation; Need to turn it to false when FC is quantized as FP16\n            This attribute access will be consistent with MTML model.\n        '
        super().__init__(name=name)
        self._layer_names = set()
        self._layers = []
        self._param_to_shape = {}
        self._seed = None
        self._sequence_seed = True
        self.param_to_optim = {}
        self.param_to_reg = {}
        self._default_optimizer = None
        self._loss = None
        self._prediction = []
        self._output_schema = None
        self._post_grad_net_modifiers = []
        self._final_net_modifiers = []
        self._breakdown_map = None
        self._input_feature_schema = schema.NewRecord(self.net, input_feature_schema) if not keep_blobs else input_feature_schema.clone()
        self._trainer_extra_schema = schema.NewRecord(self.net, trainer_extra_schema) if not keep_blobs else trainer_extra_schema.clone()
        self._metrics_schema = schema.Struct()
        self._preproc_output_schema = None
        self._init_global_constants()
        self.param_init_net = self.create_init_net('param_init_net')
        self._initialize_params = True
        self._transfer_learning_blob_name_mappings = None
        self.ad_hoc_diagnose_blobs_and_operations = []
        self.ad_hoc_plot_blobs = []
        self.use_attribution = use_attribution

    def clear_output_schema(self):
        if False:
            for i in range(10):
                print('nop')
        self._output_schema = None

    def set_initialize_params(self, initialize_params):
        if False:
            return 10
        self._initialize_params = initialize_params

    def add_metric_field(self, name, value):
        if False:
            print('Hello World!')
        assert name not in self._metrics_schema.fields, 'Try to add metric field twice: {}'.format(name)
        self._metrics_schema = self._metrics_schema + schema.Struct((name, value))

    def filter_metrics_schema(self, white_set):
        if False:
            print('Hello World!')
        logger.info('Filter metric schema with white_set {}'.format(white_set))
        field_names = self._metrics_schema.field_names()
        for name in field_names:
            if name not in white_set:
                self._metrics_schema = self._metrics_schema - schema.Struct((name, schema.Scalar()))

    def add_ad_hoc_plot_blob(self, blob, dtype=None):
        if False:
            return 10
        assert isinstance(blob, (str, core.BlobReference)), 'expect type str or BlobReference, but got {}'.format(type(blob))
        dtype = dtype or (np.float64, (1,))
        self.add_metric_field(str(blob), schema.Scalar(dtype, blob))
        self.ad_hoc_plot_blobs.append(blob)

    @staticmethod
    def _get_global_constant_initializer_op(blob_name, array=None, dtype=None, initializer=None):
        if False:
            while True:
                i = 10
        if array is not None:
            assert initializer is None, 'Only one from array and initializer should be specified'
            if dtype is None:
                array = np.array(array)
            else:
                array = np.array(array, dtype=dtype)
            op_name = None
            if array.dtype == np.int32:
                op_name = 'GivenTensorIntFill'
            elif array.dtype == np.int64:
                op_name = 'GivenTensorInt64Fill'
            elif array.dtype == str:
                op_name = 'GivenTensorStringFill'
            elif array.dtype == bool:
                op_name = 'GivenTensorBoolFill'
            else:
                op_name = 'GivenTensorFill'

            def initializer(blob_name):
                if False:
                    print('Hello World!')
                return core.CreateOperator(op_name, [], blob_name, shape=array.shape, values=array.flatten().tolist())
        else:
            assert initializer is not None
        initializer_op = initializer(blob_name)
        return initializer_op

    def add_global_constant(self, name, array=None, dtype=None, initializer=None):
        if False:
            i = 10
            return i + 15
        assert isinstance(name, str), 'name should be a string as we are using it as map key'
        assert name not in self.global_constants, '%s already added in global_constants' % name
        blob_name = self.net.NextBlob(name)
        self.global_constants[name] = blob_name
        initializer_op = LayerModelHelper._get_global_constant_initializer_op(blob_name, array, dtype, initializer)
        assert blob_name not in self.global_constant_initializers, 'there is already a initializer op associated with blob %s' % blob_name
        self.global_constant_initializers[blob_name] = initializer_op
        return blob_name

    def maybe_add_global_constant(self, name, *args, **kwargs):
        if False:
            print('Hello World!')
        if name in self.global_constants:
            blob_name = self.global_constants[name]
            initializer_op = LayerModelHelper._get_global_constant_initializer_op(blob_name, *args, **kwargs)
            assert utils.OpAlmostEqual(initializer_op, self.global_constant_initializers[blob_name], 'debug_info'), 'conflict initializers for global constant %s, previous %s, now %s' % (blob_name, str(initializer_op), str(self.global_constant_initializers[blob_name]))
            return blob_name
        return self.add_global_constant(name, *args, **kwargs)

    def _init_global_constants(self):
        if False:
            i = 10
            return i + 15
        self.global_constants = {}
        self.global_constant_initializers = {}
        self.add_global_constant('ONE', 1.0)
        self.add_global_constant('NAN', float('NaN'))
        self.add_global_constant('ZERO', 0.0)
        self.add_global_constant('ZERO_RANGE', [0, 0], dtype='int32')

    def _add_global_constants(self, init_net):
        if False:
            print('Hello World!')
        for initializer_op in self.global_constant_initializers.values():
            init_net._net.op.extend([initializer_op])

    def create_init_net(self, name):
        if False:
            return 10
        init_net = core.Net(name)
        self._add_global_constants(init_net)
        return init_net

    def _validate_param_shape(self, param_name, shape):
        if False:
            for i in range(10):
                print('nop')
        if param_name not in self._param_to_shape:
            return
        ref_shape = self._param_to_shape[param_name]
        if shape != ref_shape:
            raise ValueError('Got inconsistent shapes between shared parameters when trying to map a blob in scope {0} to {1}. ref_shape :  {2}, shape : {3}'.format(scope.CurrentNameScope(), param_name, ref_shape, shape))

    def _validate_param_optim(self, param_name, optim):
        if False:
            i = 10
            return i + 15
        if param_name not in self.param_to_optim:
            return
        logger.info('{} shares the same parameter with another parameter. Validating if the same optimizer has been specified for them.'.format(param_name))
        ref_optim = self.param_to_optim[param_name]
        if optim is None:
            assert ref_optim == self._default_optimizer, 'Optim for {} is None which will fall back to use default_optimizer. However, the optimizer that has been specified for this shared parameter is {} which is different from default_optimizer {}. Please check the optimizers specified for parameters shared with {} and the default_optimizer to ensure the consistency.'.format(param_name, ref_optim, self._default_optimizer, param_name)
        elif optim == self.NoOptim:
            assert ref_optim == self.NoOptim, 'Optim for {} is NoOptim. However, the optimizer for the parameters shared with {} is {} which is different from NoOptim. Please check the optimizer specified for other parameters in the shared group to ensure consistency.'.format(param_name, param_name, ref_optim)
        elif isinstance(optim, Optimizer):
            assert isinstance(ref_optim, Optimizer), 'Optim for {} is an instance of Optimizer. However, the optimizer for the parameters shared with {} is {} which is not an instance of Optimizer. Please check the optimizer specified for other  parameters in the shared group to ensure consistency.'.format(param_name, param_name, ref_optim, optim)
            assert type(optim) is type(ref_optim) and optim.attributes == ref_optim.attributes, "Optim for {} is an instance of Optimizer. However, the optimizer for the parameters shared with {} is {}. This optimizer either doesn't have the same type as the current optimizer: {} vs {}, or its attributes such as learning rate are different from that of current optimizer which is {} vs {}. Please check the optimizer specified for other parameters in the shared group to ensure consistency.".format(param_name, param_name, ref_optim, type(optim), type(ref_optim), optim.attributes, ref_optim.attributes)
        else:
            raise ValueError('optim should be either None, NoOptim, or an instance of Optimizer, Got {} '.format(optim))

    def create_param(self, param_name, shape, initializer, optimizer=None, ps_param=None, regularizer=None):
        if False:
            return 10
        if isinstance(param_name, core.BlobReference):
            param_name = str(param_name)
        elif isinstance(param_name, str):
            param_name = parameter_sharing_context.get_parameter_name(param_name)
        else:
            raise ValueError('Unsupported type for param_name')
        param_blob = core.BlobReference(param_name)
        if len(initializer) == 1:
            init_op_args = {}
        else:
            assert len(initializer) == 2
            init_op_args = copy.deepcopy(initializer[1])
        if shape is not None:
            assert 'shape' not in init_op_args
            init_op_args.update({'shape': shape})
        initializer_op = None
        if self._initialize_params:
            initializer_op = core.CreateOperator(initializer[0], [], param_blob, **init_op_args)
        param = layers.LayerParameter(parameter=param_blob, initializer=initializer_op, optimizer=optimizer, ps_param=ps_param, regularizer=regularizer)
        self._validate_param_shape(param_name, shape)
        self._validate_param_optim(param_name, optimizer)
        self._param_to_shape[param_name] = shape
        return param

    def next_layer_name(self, prefix):
        if False:
            return 10
        base_name = core.ScopedName(prefix)
        name = base_name
        index = 0
        while name in self._layer_names:
            name = base_name + '_auto_' + str(index)
            index += 1
        self._layer_names.add(name)
        return name

    def add_layer(self, layer):
        if False:
            i = 10
            return i + 15
        self._layers.append(layer)
        for param in layer.get_parameters():
            assert isinstance(param.parameter, core.BlobReference)
            self.param_to_optim[str(param.parameter)] = param.optimizer or self.default_optimizer
            self.params.append(param.parameter)
            if isinstance(param, layers.LayerParameter):
                logger.info('Add parameter regularizer {0}'.format(param.parameter))
                self.param_to_reg[param.parameter] = param.regularizer
            elif isinstance(param, ParameterInfo):
                logger.info('regularization is unsupported for ParameterInfo object')
            else:
                raise ValueError('unknown object type besides ParameterInfo and LayerParameter: {}'.format(param))
        layer.add_operators(self.net, self.param_init_net)
        return layer.output_schema

    def get_parameter_blobs(self):
        if False:
            print('Hello World!')
        param_blobs = []
        for layer in self._layers:
            for param in layer.get_parameters():
                param_blobs.append(param.parameter)
        return param_blobs

    def add_post_grad_net_modifiers(self, modifier):
        if False:
            print('Hello World!')
        assert modifier not in self._post_grad_net_modifiers, '{0} is already in {1}'.format(modifier, self._post_grad_net_modifiers)
        assert isinstance(modifier, NetModifier), '{} has to be a NetModifier instance'.format(modifier)
        self._post_grad_net_modifiers.append(modifier)

    def add_final_net_modifiers(self, modifier):
        if False:
            print('Hello World!')
        assert modifier not in self._final_net_modifiers, '{0} is already in {1}'.format(modifier, self._final_net_modifiers)
        assert isinstance(modifier, NetModifier), '{} has to be a NetModifier instance'.format(modifier)
        self._final_net_modifiers.append(modifier)

    @property
    def seed(self):
        if False:
            print('Hello World!')
        return self._seed

    @property
    def sequence_seed(self):
        if False:
            return 10
        return self._sequence_seed

    def store_seed(self, seed, sequence_seed=True):
        if False:
            while True:
                i = 10
        self._seed = seed
        self._sequence_seed = sequence_seed

    def apply_seed(self, net):
        if False:
            while True:
                i = 10
        if self._seed:
            net.set_rand_seed(self._seed, self._sequence_seed)

    @property
    def default_optimizer(self):
        if False:
            while True:
                i = 10
        return self._default_optimizer

    @default_optimizer.setter
    def default_optimizer(self, optimizer):
        if False:
            return 10
        self._default_optimizer = optimizer

    @property
    def input_feature_schema(self):
        if False:
            i = 10
            return i + 15
        return self._input_feature_schema

    @property
    def trainer_extra_schema(self):
        if False:
            print('Hello World!')
        return self._trainer_extra_schema

    @property
    def metrics_schema(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the schema that represents model output that should be used for\n        metric reporting.\n\n        During the training/evaluation this schema will be appended to the\n        schema that represents model output.\n        '
        return self._metrics_schema

    @property
    def output_schema(self):
        if False:
            i = 10
            return i + 15
        assert self._output_schema is not None
        return self._output_schema

    @output_schema.setter
    def output_schema(self, schema):
        if False:
            print('Hello World!')
        assert self._output_schema is None
        self._output_schema = schema

    @property
    def preproc_output_schema(self):
        if False:
            while True:
                i = 10
        assert self._preproc_output_schema is not None
        return self._preproc_output_schema

    @preproc_output_schema.setter
    def preproc_output_schema(self, schema):
        if False:
            while True:
                i = 10
        assert self._preproc_output_schema is None
        self._preproc_output_schema = schema

    @property
    def prediction(self):
        if False:
            i = 10
            return i + 15
        assert self._prediction, 'model prediction is empty'
        return self._prediction

    def add_prediction(self, prediction, weight=1.0):
        if False:
            while True:
                i = 10
        assert prediction is not None, 'Added prediction should not be None'
        self._prediction.append((prediction, weight))

    @property
    def transfer_learning_blob_name_mappings(self):
        if False:
            while True:
                i = 10
        return self._transfer_learning_blob_name_mappings

    @transfer_learning_blob_name_mappings.setter
    def transfer_learning_blob_name_mappings(self, blob_name_mappings):
        if False:
            return 10
        assert blob_name_mappings is not None, 'Transfer learning blob name mappings should not be None'
        self._transfer_learning_blob_name_mappings = blob_name_mappings

    @property
    def loss(self):
        if False:
            print('Hello World!')
        assert self._loss is not None
        return self._loss

    @loss.setter
    def loss(self, loss):
        if False:
            return 10
        assert self._loss is None
        self._loss = loss

    def has_loss(self):
        if False:
            while True:
                i = 10
        return self._loss is not None

    def add_loss(self, loss, name='unnamed'):
        if False:
            while True:
                i = 10
        assert loss is not None, 'Added loss should not be None'
        assert isinstance(loss, schema.Scalar) or isinstance(loss, schema.Struct), 'Added loss should be a scalar or a struct'
        if self._loss is None:
            self._loss = schema.Struct((name, loss))
        else:
            if isinstance(self._loss, schema.Scalar):
                self._loss = schema.Struct(('unnamed', self._loss))
            prefix_base = name + '_auto_'
            index = 0
            prefix = name
            while prefix in self._loss:
                prefix = prefix_base + str(index)
                index += 1
            loss_struct = schema.Struct((prefix, loss))
            self._loss = self._loss + loss_struct

    def add_output_schema(self, name, value):
        if False:
            return 10
        assert value is not None, 'Added output schema {} should not be None'.format(name)
        assert isinstance(value, schema.Scalar) or isinstance(value, schema.Struct), 'Added output schema {} should be a scalar or a struct.\n            Now it is {}.'.format(name, type(value))
        if self._output_schema is None:
            self._output_schema = schema.Struct((name, value))
        else:
            assert name not in self._output_schema.fields, 'Output Schema Field {} already exists'.format(name)
            self._output_schema = self._output_schema + schema.Struct((name, value))

    def add_trainer_extra_schema(self, trainer_extra_schema):
        if False:
            print('Hello World!')
        trainer_extra_record = schema.NewRecord(self.net, trainer_extra_schema)
        self._trainer_extra_schema += trainer_extra_record

    def __getattr__(self, layer):
        if False:
            while True:
                i = 10

        def is_functional_layer(layer):
            if False:
                while True:
                    i = 10
            if core.IsOperator(layer):
                return True
            elif layer.startswith('FunctionalLayer'):
                return True
            else:
                return False

        def resolve_functional_layer(layer):
            if False:
                print('Hello World!')
            if core.IsOperator(layer):
                return layer
            elif layer.startswith('FunctionalLayer'):
                return layer[len('FunctionalLayer'):]
            else:
                raise ValueError('%s cannot be resolved as functional layer' % layer)
        if layer.startswith('__'):
            raise AttributeError(layer)
        if layers.layer_exists(layer):

            def wrapper(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                new_layer = layers.create_layer(layer, self, *args, **kwargs)
                if kwargs.get('output_to_metrics', False):
                    new_layer.export_output_for_metrics()
                if kwargs.get('params_to_metrics', False):
                    new_layer.export_params_for_metrics()
                return self.add_layer(new_layer)
            return wrapper
        elif is_functional_layer(layer):
            layer = resolve_functional_layer(layer)

            def wrapper(*args, **kwargs):
                if False:
                    while True:
                        i = 10

                def apply_operator(net, in_record, out_record, **kwargs):
                    if False:
                        print('Hello World!')
                    net.__getattr__(layer)(in_record.field_blobs(), out_record.field_blobs(), **kwargs)
                if 'name' not in kwargs:
                    kwargs['name'] = layer
                new_layer = layers.create_layer('Functional', self, *args, function=apply_operator, **kwargs)
                if kwargs.get('output_to_metrics', False):
                    new_layer.export_output_for_metrics()
                if kwargs.get('params_to_metrics', False):
                    new_layer.export_params_for_metrics()
                return self.add_layer(new_layer)
            return wrapper
        else:
            raise AttributeError('Trying to create non-registered layer: {}'.format(layer))

    @property
    def layers(self):
        if False:
            for i in range(10):
                print('nop')
        return self._layers

    def apply_regularizers_on_loss(self, train_net, train_init_net, blob_to_device=None):
        if False:
            return 10
        logger.info('apply regularizer on loss')
        for (param, regularizer) in self.param_to_reg.items():
            if regularizer is None:
                continue
            logger.info('add regularizer {0} for param {1} to loss'.format(regularizer, param))
            assert isinstance(regularizer, Regularizer)
            added_loss_blob = regularizer(train_net, train_init_net, param, grad=None, by=RegularizationBy.ON_LOSS)
            logger.info(added_loss_blob)
            if added_loss_blob is not None:
                self.add_loss(schema.Scalar(blob=added_loss_blob), str(added_loss_blob))

    def apply_regularizers_after_optimizer(self, train_net, train_init_net, grad_map, blob_to_device=None):
        if False:
            i = 10
            return i + 15
        logger.info('apply regularizer after optimizer')
        CPU = muji.OnCPU()
        blob_to_device = blob_to_device or {}
        for (param, regularizer) in self.param_to_reg.items():
            if regularizer is None:
                continue
            assert isinstance(regularizer, Regularizer)
            logger.info('add regularizer {0} for param {1} to optimizer'.format(regularizer, param))
            device = get_param_device(param, grad_map.get(str(param)), param_to_device=blob_to_device, default_device=CPU)
            with core.DeviceScope(device):
                regularizer(train_net, train_init_net, param, grad=grad_map.get(str(param)), by=RegularizationBy.AFTER_OPTIMIZER)

    def apply_post_grad_net_modifiers(self, trainer_net, trainer_init_net, grad_map, blob_to_device=None, modify_output_record=False):
        if False:
            return 10
        param_grad_map = {param: grad_map[param] for param in self.param_to_optim.keys() if param in grad_map}
        for modifier in self._post_grad_net_modifiers:
            modifier(trainer_net, trainer_init_net, param_grad_map, blob_to_device=blob_to_device, modify_output_record=modify_output_record)

    def apply_final_net_modifiers(self, trainer_net, trainer_init_net, grad_map, blob_to_device=None, modify_output_record=False):
        if False:
            i = 10
            return i + 15
        for modifier in self._final_net_modifiers:
            modifier(trainer_net, trainer_init_net, grad_map, blob_to_device=blob_to_device, modify_output_record=modify_output_record)

    def apply_optimizers(self, train_net, train_init_net, grad_map, blob_to_device=None):
        if False:
            for i in range(10):
                print('nop')
        CPU = muji.OnCPU()
        blob_to_device = blob_to_device or {}
        for (param, optimizer) in self.param_to_optim.items():
            assert optimizer is not None, 'default optimizer must have been set in add_layer'
            device = get_param_device(param, grad_map.get(str(param)), param_to_device=blob_to_device, default_device=CPU)
            if device is not None:
                del device.extra_info[:]
            with core.DeviceScope(device):
                optimizer(train_net, train_init_net, param, grad_map.get(str(param)))

    def _GetOne(self):
        if False:
            return 10
        return self.global_constants['ONE']

    def NoOptim(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    @property
    def breakdown_map(self):
        if False:
            print('Hello World!')
        return self._breakdown_map

    @breakdown_map.setter
    def breakdown_map(self, breakdown_map):
        if False:
            i = 10
            return i + 15
        assert isinstance(breakdown_map, dict)
        assert all((isinstance(k, str) for k in breakdown_map))
        assert sorted(breakdown_map.values()) == list(range(len(breakdown_map)))
        self._breakdown_map = breakdown_map
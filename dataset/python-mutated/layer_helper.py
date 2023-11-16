import copy
import paddle
from . import unique_name
from .dygraph_utils import _append_activation_in_dygraph
from .framework import Parameter, _global_flags, dtype_is_floating, in_dygraph_mode
from .layer_helper_base import LayerHelperBase
from .param_attr import ParamAttr

class LayerHelper(LayerHelperBase):

    def __init__(self, layer_type, **kwargs):
        if False:
            while True:
                i = 10
        self.kwargs = kwargs
        name = self.kwargs.get('name', None)
        if name is None:
            self.kwargs['name'] = unique_name.generate(layer_type)
        super().__init__(self.kwargs['name'], layer_type=layer_type)

    def append_op(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.main_program.current_block().append_op(*args, **kwargs)

    def multiple_input(self, input_param_name='input'):
        if False:
            return 10
        inputs = self.kwargs.get(input_param_name, [])
        ret = []
        if isinstance(inputs, (list, tuple)):
            for inp in inputs:
                ret.append(self.to_variable(inp))
        else:
            ret.append(self.to_variable(inputs))
        return ret

    def input(self, input_param_name='input'):
        if False:
            print('Hello World!')
        inputs = self.multiple_input(input_param_name)
        if len(inputs) != 1:
            raise f'{self.layer_type} layer only takes one input'
        return inputs[0]

    @property
    def param_attr(self):
        if False:
            print('Hello World!')
        return ParamAttr._to_attr(self.kwargs.get('param_attr', None))

    @property
    def bias_attr(self):
        if False:
            print('Hello World!')
        return ParamAttr._to_attr(self.kwargs.get('bias_attr', None))

    def multiple_param_attr(self, length):
        if False:
            return 10
        param_attr = self.param_attr
        if isinstance(param_attr, ParamAttr):
            param_attr = [param_attr]
        if len(param_attr) != 1 and len(param_attr) != length:
            raise ValueError('parameter number mismatch')
        elif len(param_attr) == 1 and length != 1:
            tmp = [None] * length
            for i in range(length):
                tmp[i] = copy.deepcopy(param_attr[0])
            param_attr = tmp
        return param_attr

    def iter_inputs_and_params(self, input_param_name='input'):
        if False:
            i = 10
            return i + 15
        inputs = self.multiple_input(input_param_name)
        param_attrs = self.multiple_param_attr(len(inputs))
        yield from zip(inputs, param_attrs)

    def input_dtype(self, input_param_name='input'):
        if False:
            return 10
        inputs = self.multiple_input(input_param_name)
        dtype = None
        for each in inputs:
            if dtype is None:
                dtype = each.dtype
            elif dtype != each.dtype:
                raise ValueError('Data Type mismatch: %d to %d' % (dtype, each.dtype))
        return dtype

    def get_parameter(self, name):
        if False:
            for i in range(10):
                print('nop')
        param = self.main_program.global_block().var(name)
        if not isinstance(param, Parameter):
            raise ValueError('no Parameter name %s found' % name)
        return param

    def append_bias_op(self, input_var, dim_start=1, dim_end=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Append bias operator and return its output. If the user does not set\n        bias_attr, append_bias_op will return input_var\n\n        :param input_var: the input variable. The len(input_var.shape) is\n        larger or equal than 2.\n        :bias_initializer: an instance of a subclass of Initializer used to\n        initialize the bias\n        :param dim_start:\n        :param dim_end: the shape of the bias will be\n        input_var.shape[dim_start:dim_end]. The bias is broadcasted to other\n        dimensions and added to input_var to get the output\n        '
        size = list(input_var.shape[dim_start:dim_end])
        bias_attr = self.bias_attr
        if not bias_attr:
            return input_var
        b = self.create_parameter(attr=bias_attr, shape=size, dtype=input_var.dtype, is_bias=True)
        tmp = self.create_variable_for_type_inference(dtype=input_var.dtype)
        self.append_op(type='elementwise_add', inputs={'X': [input_var], 'Y': [b]}, outputs={'Out': [tmp]}, attrs={'axis': dim_start})
        return tmp

    def append_activation(self, input_var):
        if False:
            print('Hello World!')
        act = self.kwargs.get('act', None)
        if act is None:
            return input_var
        if isinstance(act, str):
            act = {'type': act}
        else:
            raise TypeError(str(act) + ' should be unicode or str')
        use_cudnn = None
        if 'use_cudnn' in self.kwargs and self.kwargs.get('use_cudnn'):
            use_cudnn = self.kwargs.get('use_cudnn')
            act['use_cudnn'] = use_cudnn
        use_mkldnn = self.kwargs.get('use_mkldnn', _global_flags().get('FLAGS_use_mkldnn', False))
        if use_mkldnn:
            act['use_mkldnn'] = use_mkldnn
        act_type = act.pop('type')
        if in_dygraph_mode():
            res = _append_activation_in_dygraph(input_var, act_type, use_cudnn, use_mkldnn)
            return res
        else:
            tmp = self.create_variable_for_type_inference(dtype=input_var.dtype)
            self.append_op(type=act_type, inputs={'X': [input_var]}, outputs={'Out': [tmp]}, attrs=act)
            return tmp

    def _get_default_initializer(self, dtype):
        if False:
            return 10
        if dtype is None or dtype_is_floating(dtype) is True:
            return paddle.nn.initializer.XavierUniform()
        else:
            return paddle.nn.initializer.Constant()

    def is_instance(self, param_name, cls):
        if False:
            i = 10
            return i + 15
        param = self.kwargs.get(param_name, None)
        if not isinstance(param, cls):
            raise TypeError('The input {0} parameter of method {1} must be {2}', param_name, self.layer_type, cls.__name__)
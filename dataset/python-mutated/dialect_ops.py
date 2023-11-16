from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil import Operation
from coremltools.converters.mil.mil.input_type import *
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
register_op = SSAOpRegistry.register_op

@register_op(doc_str='TODO', namespace='tf')
class tf_make_list(Operation):
    input_spec = InputSpec(init_length=IntInputType(optional=True, default=1), dynamic_length=BoolInputType(optional=True, default=True), elem_shape=TensorInputType(const=True, optional=True), dtype=StringInputType(const=True, optional=True, default='fp32'))

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(tf_make_list, self).__init__(**kwargs)

    def type_inference(self):
        if False:
            i = 10
            return i + 15
        init_length = self.init_length.val
        if self.elem_shape is None or self.elem_shape.sym_val is None:
            return types.list(types.unknown, init_length=init_length, dynamic_length=self.dynamic_length.val)
        builtin_dtype = types.string_to_builtin(self.dtype.val)
        if builtin_dtype is None:
            raise ValueError('Unsupported dtype {}'.format(self.dtype.val))
        elem_type = types.tensor(builtin_dtype, self.elem_shape.sym_val)
        return types.list(elem_type, init_length=init_length, dynamic_length=self.dynamic_length.val)

class TfLSTMBase(Operation):
    """
    Common LSTM inputs for BlockLSTMCell and BlockLSTM.
    """
    input_spec = InputSpec(c_prev=TensorInputType(), h_prev=TensorInputType(), weight=TensorInputType(const=True), forget_bias=FloatInputType(const=True, optional=True, default=1.0), cell_clip=FloatInputType(const=True, optional=True), use_peephole=BoolInputType(const=True, optional=True, default=False), weight_peep_i=TensorInputType(const=True, optional=True), weight_peep_f=TensorInputType(const=True, optional=True), weight_peep_o=TensorInputType(const=True, optional=True), bias=TensorInputType(const=True))

    def _check_peephole_weights(self):
        if False:
            for i in range(10):
                print('nop')
        if self.use_peephole.val:
            if self.weight_peep_i is None or self.weight_peep_f is None or self.weight_peep_o is None:
                raise ValueError('weight_peep_* cannot be None when use_peephole is True')

@register_op(doc_str='\n                     xh = [x, h_prev]\n                     [i, ci, f, o] = xh * w + b\n                     f = f + forget_bias\n                     if not use_peephole:\n                       wci = wcf = wco = 0\n                     i = sigmoid(cs_prev .* wci + i)\n                     f = sigmoid(cs_prev .* wcf + f)\n                     ci = tanh(ci)\n                     cs = ci .* i + cs_prev .* f\n                     cs = clip(cs, cell_clip)\n                     o = sigmoid(cs * wco + o)\n                     co = tanh(cs)\n                     h = co .* o\n                     ', namespace='tf')
class tf_lstm_block_cell(TfLSTMBase):
    input_spec = InputSpec(x=TensorInputType()) + TfLSTMBase.input_spec

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(tf_lstm_block_cell, self).__init__(**kwargs)

    def type_inference(self):
        if False:
            while True:
                i = 10
        self._check_peephole_weights()
        ret_shape = self.c_prev.shape
        dtype = self.x.dtype
        return (types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape))

@register_op(doc_str='\n                     Apply LSTM to an input sequence\n                     ', namespace='tf')
class tf_lstm_block(TfLSTMBase):
    input_spec = InputSpec(seq_len=IntInputType(), x=TensorInputType()) + TfLSTMBase.input_spec

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(tf_lstm_block, self).__init__(**kwargs)

    def type_inference(self):
        if False:
            print('Hello World!')
        self._check_peephole_weights()
        padded_len = self.x.shape[0]
        ret_shape = [padded_len] + list(self.c_prev.shape)
        dtype = self.x.dtype
        return (types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape), types.tensor(dtype, ret_shape))
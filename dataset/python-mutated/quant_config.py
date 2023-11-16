SUPPORT_WEIGHT_QUANTIZATION_OP_DICT = {'conv2d': [['Input', 'Filter'], ['Output']], 'depthwise_conv2d': [['Input', 'Filter'], ['Output']], 'conv2d_transpose': [['Input', 'Filter'], ['Output']], 'mul': [['X', 'Y'], ['Out']], 'matmul': [['X', 'Y'], ['Out']], 'matmul_v2': [['X', 'Y'], ['Out']]}
SUPPORT_ACT_QUANTIZATION_OP_DICT = {'mul': [['X', 'Y'], ['Out']], 'matmul': [['X', 'Y'], ['Out']], 'matmul_v2': [['X', 'Y'], ['Out']], 'pool2d': [['X'], ['Out']], 'elementwise_add': [['X', 'Y'], ['Out']], 'concat': [['X'], ['Out']], 'softmax': [['X'], ['Out']], 'argmax': [['X'], ['Out']], 'transpose': [['X'], ['Out']], 'equal': [['X', 'Y'], ['Out']], 'gather': [['X'], ['Out']], 'greater_equal': [['X', 'Y'], ['Out']], 'greater_than': [['X', 'Y'], ['Out']], 'less_equal': [['X', 'Y'], ['Out']], 'less_than': [['X', 'Y'], ['Out']], 'mean': [['X'], ['Out']], 'not_equal': [['X', 'Y'], ['Out']], 'reshape': [['X'], ['Out']], 'reshape2': [['X'], ['Out']], 'transpose2': [['X'], ['Out']], 'nearest_interp': [['X'], ['Out']], 'trilinear_interp': [['X'], ['Out']], 'slice': [['Input'], ['Out']], 'squeeze': [['X'], ['Out']], 'elementwise_sub': [['X', 'Y'], ['Out']], 'relu': [['X'], ['Out']], 'relu6': [['X'], ['Out']], 'leaky_relu': [['X'], ['Out']], 'prelu': [['X', 'Alpha'], ['Out']], 'tanh': [['X'], ['Out']], 'swish': [['X'], ['Out']], 'dropout': [['X'], ['Out']], 'batch_norm': [['X'], ['Y']], 'layer_norm': [['X'], ['Y']], 'sigmoid': [['X'], ['Out']], 'elementwise_mul': [['X', 'Y'], ['Out']], 'elementwise_pow': [['X', 'Y'], ['Out']], 'hard_swish': [['X'], ['Out']], 'hard_sigmoid': [['X'], ['Out']], 'gru': [['Input', 'Weight'], ['Hidden']], 'lstm': [['Input', 'Weight'], ['Hidden']], 'pad2d': [['X'], ['Out']], 'pad3d': [['X'], ['Out']], 'flatten': [['X'], ['Out']], 'flatten2': [['X'], ['Out']], 'unsqueeze2': [['X'], ['Out']], 'flatten_contiguous_range': [['X'], ['Out']], 'split': [['X'], ['Out']], 'squeeze2': [['X'], ['Out']], 'nearest_interp_v2': [['X'], ['Out']], 'bilinear_interp': [['X'], ['Out']], 'bilinear_interp_v2': [['X'], ['Out']], 'fill_constant_batch_size_like': [['Input'], ['Out']], 'arg_max': [['X'], ['Out']], 'abs': [['X'], ['Out']], 'assign': [['X'], ['Out']], 'cast': [['X'], ['Out']], 'clip': [['X'], ['Out']], 'box_coder': [['PriorBox'], ['OutputBox']], 'crop': [['X'], ['Out']], 'cumsum': [['X'], ['Out']], 'expand_v2': [['X'], ['Out']], 'fill_any_like': [['X'], ['Out']], 'fill_constant': [[], ['Out']], 'gelu': [['X'], ['Out']], 'instance_norm': [['X'], ['Y']], 'lookup_table': [['W', 'Ids'], ['Out']], 'lookup_table_v2': [['W', 'Ids'], ['Out']], 'norm': [['X'], ['Norm']], 'p_norm': [['X'], ['Out']], 'pow': [['X'], ['Out']], 'reduce_mean': [['X'], ['Out']], 'stack': [['X'], ['Y']], 'top_k_v2': [['X'], ['Out', 'Indices']], 'logical_and': [['X', 'Y'], ['Out']], 'logical_not': [['X'], ['Out']], 'meshgrid': [['X'], ['Out']], 'roi_align': [['X', 'ROIs'], ['Out']], 'strided_slice': [['Input'], ['Out']], 'where': [['Condition', 'X', 'Y'], ['Out']], 'grid_sampler': [['X', 'Grid'], ['Output']], 'tile': [['X'], ['Out']], 'group_norm': [['X'], ['Y', 'Mean', 'Variance']], 'reduce_sum': [['X'], ['Out']], 'square': [['X'], ['Out']], 'softplus': [['X'], ['Out']], 'shuffle_channel': [['X'], ['Out']], 'reduce_max': [['X'], ['Out']], 'scale': [['X'], ['Out']]}
SUPPORT_QUANTIZATION_OP_DICT = SUPPORT_WEIGHT_QUANTIZATION_OP_DICT.copy()
SUPPORT_QUANTIZATION_OP_DICT.update(SUPPORT_ACT_QUANTIZATION_OP_DICT)

class BaseQuantizer:
    """
    Basic quantization configuration class, which configures some hyperparameters
    required for quantization, including the list of op types to be quantized,
    quantization bit number for weight and activation and the range of quantization values.
    Args:
        quantizable_op_type(list[str], optional): List the type of ops
            that will be quantized. Default is []. If quantizable_op_type is [],
            it will use the default quantization op type of the qunat config in
            the current Quantizer.
        quant_bits(int, optional): Quantization bit number for weight and activation.
            Default is 8.
    """

    def __init__(self, quantizable_op_type=[], quant_bits=8):
        if False:
            print('Hello World!')
        self._quantizable_op_type = quantizable_op_type
        self._quant_bits = quant_bits
        self._quant_min = -128
        self._quant_max = 127

    @property
    def weight_quant_operation_types(self):
        if False:
            i = 10
            return i + 15
        '\n        Operation type list which should support weight quantization.\n        And before these ops, quant dequant nodes will be inserted.\n        '
        base_weight_op_type_list = list(SUPPORT_WEIGHT_QUANTIZATION_OP_DICT.keys())
        if self._quantizable_op_type:
            weight_list = []
            for _op_type in self._quantizable_op_type:
                if _op_type in base_weight_op_type_list:
                    weight_list.append(_op_type)
            return weight_list
        else:
            return base_weight_op_type_list

    @property
    def activation_quant_operation_types(self):
        if False:
            return 10
        '\n        Operation type list which should support activation quantization.\n        And before these ops, quant dequant nodes will be inserted.\n        '
        base_act_op_type_list = list(SUPPORT_ACT_QUANTIZATION_OP_DICT.keys())
        act_quant_op_list = []
        if self._quantizable_op_type:
            for _op_type in self._quantizable_op_type:
                if _op_type in base_act_op_type_list:
                    act_quant_op_list.append(_op_type)
        else:
            act_quant_op_list = ['mul', 'matmul', 'matmul_v2']
        return act_quant_op_list

    @property
    def observer_operation_types(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Operation type list for observer in quantization. These nodes only count the\n        calibration boundary scale and do not participate in the fake quantization.\n        In order to facilitate the deployment of the prediction engine, quant\n        and dequant nodes will be inserted after these ops when exporting the model.\n        '
        return list(SUPPORT_ACT_QUANTIZATION_OP_DICT.keys())

class TensorRTQuantizer(BaseQuantizer):
    """
    TensorRT quantization configuration class.
    Args:
        quantizable_op_type(list[str], optional): List the type of ops
            that will be quantized. Default is []. If quantizable_op_type is [],
            it will use the default quantization op type of the qunat config in
            the current Quantizer.
        quant_bits(int, optional): Quantization bit number for weight and activation.
            Default is 8.
    """

    def __init__(self, quantizable_op_type=[], quant_bits=8):
        if False:
            print('Hello World!')
        super().__init__()
        self._quantizable_op_type = quantizable_op_type
        self._quant_bits = quant_bits
        self._quant_min = -128
        self._quant_max = 127

    @property
    def activation_quant_operation_types(self):
        if False:
            return 10
        '\n        Operation type list which should support activation quantization.\n        And before these ops, quant dequant nodes will be inserted.\n        '
        return ['pool2d', 'elementwise_add', 'elementwise_sub', 'elementwise_mul', 'elementwise_pow', 'concat', 'softmax', 'argmax', 'mean', 'relu', 'relu6', 'leaky_relu', 'tanh', 'swish', 'softplus', 'gelu', 'hard_sigmoid', 'hard_swish', 'sigmoid', 'layer_norm', 'matmul_v2', 'split', 'bilinear_interp', 'nearest_interp', 'trilinear_interp', 'nearest_interp_v2', 'bilinear_interp', 'bilinear_interp_v2', 'clip', 'pow', 'reduce_mean', 'reduce_sum', 'reduce_max']

class MKLDNNQuantizer(BaseQuantizer):
    """
    MKLDNN quantization configuration class.
    Args:
        quantizable_op_type(list[str], optional): List the type of ops
            that will be quantized. Default is []. If quantizable_op_type is [],
            it will use the default quantization op type of the qunat config in
            the current Quantizer.
        quant_bits(int, optional): Quantization bit number for weight and activation.
            Default is 8.
    """

    def __init__(self, quantizable_op_type=[], quant_bits=8):
        if False:
            print('Hello World!')
        super().__init__()
        self._quantizable_op_type = quantizable_op_type
        self._quant_bits = quant_bits
        self._quant_min = -128
        self._quant_max = 127

    @property
    def activation_quant_operation_types(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Operation type list which should support activation quantization.\n        And before these ops, quant dequant nodes will be inserted.\n        '
        return ['pool2d', 'elementwise_add', 'elementwise_mul', 'concat', 'nearest_interp', 'nearest_interp_v2', 'split']

class ARMCPUQuantizer(BaseQuantizer):
    """
    ARM CPU with Paddle Lite quantization configuration class.
    Args:
        quantizable_op_type(list[str], optional): List the type of ops
            that will be quantized. Default is []. If quantizable_op_type is [],
            it will use the default quantization op type of the qunat config in
            the current Quantizer.
        quant_bits(int, optional): Quantization bit number for weight and activation.
            Default is 8.
    """

    def __init__(self, quantizable_op_type=[], quant_bits=8):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._quantizable_op_type = quantizable_op_type
        self._quant_bits = quant_bits
        self._quant_min = -127
        self._quant_max = 127
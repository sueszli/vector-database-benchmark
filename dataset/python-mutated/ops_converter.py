from bigdl.dllib.nn.layer import SpatialAveragePooling, SpatialBatchNormalization
from bigdl.dllib.nn.layer import SpatialConvolution, SpatialMaxPooling, JoinTable
from bigdl.dllib.nn.layer import ReLU, SoftMax, CAddTable, Unsqueeze
from bigdl.dllib.nn.onnx.layer import Constant, Gather, Gemm, Shape, Reshape
from .converter_utils import *

def average_pool(inputs, prev_modules, attrs, outputs):
    if False:
        print('Hello World!')
    auto_pad = attrs.get('auto_pad', 'NOTSET')
    ceil_mode = True if attrs.get('ceil_mode', 0) == 1 else False
    count_include_pad = True if attrs.get('count_include_pad', 0) == 1 else False
    (kernel_width, kernel_height) = map(int, attrs.get('kernel_shape', (1, 1))[:2])
    (stride_width, stride_height) = map(int, attrs.get('strides', (1, 1))[:2])
    (padding_width, padding_height) = map(int, attrs.get('pads', (0, 0))[:2])
    (_, data_tensor_shape) = inputs[0]
    (input_height, input_width) = data_tensor_shape[-2:]
    output_height = calc_output_shape(input_height, kernel_height, padding=padding_height, stride=stride_height, ceil_mode=ceil_mode)
    output_width = calc_output_shape(input_width, kernel_width, padding=padding_width, stride=stride_width, ceil_mode=ceil_mode)
    out_tensor_shape = list(data_tensor_shape)
    out_tensor_shape[-2] = output_height
    out_tensor_shape[-1] = output_width
    out_tensor_shape = tuple(out_tensor_shape)
    module = SpatialAveragePooling(kw=kernel_width, kh=kernel_height, dw=stride_width, dh=stride_height, pad_w=padding_width, pad_h=padding_height, ceil_mode=ceil_mode, count_include_pad=count_include_pad)(prev_modules)
    return (module, [out_tensor_shape])

def batch_norm(inputs, prev_modules, attrs, outputs):
    if False:
        i = 10
        return i + 15
    epsilon = float(attrs.get('epsilon', 1e-05))
    momentum = float(attrs.get('momentum', 0.9))
    (_, data_tensor_shape) = inputs[0]
    (scale_tensor_val, _) = inputs[1]
    (bias_tensor_val, _) = inputs[2]
    (mean_tensor_val, _) = inputs[3]
    (var_tensor_val, _) = inputs[4]
    out_tensor_shape = data_tensor_shape
    n_output = int(data_tensor_shape[1])
    temp_module = SpatialBatchNormalization(n_output=n_output, eps=epsilon, momentum=momentum, init_weight=scale_tensor_val, init_bias=bias_tensor_val)
    if mean_tensor_val is not None:
        temp_module.set_running_mean(mean_tensor_val)
    if var_tensor_val is not None:
        temp_module.set_running_std(var_tensor_val)
    module = temp_module(prev_modules[0])
    return (module, [out_tensor_shape])

def concat(inputs, prev_modules, attrs, outputs):
    if False:
        return 10
    axis = int(attrs.get('axis'))
    (_, data_tensor_shape) = inputs[0]
    dim_rank = 0
    for i in range(len(inputs)):
        (_, curr_input_shape) = inputs[i]
        for j in range(len(data_tensor_shape)):
            if axis != j:
                if curr_input_shape[i] != data_tensor_shape[i]:
                    invalidInputError(False, 'Input shape mismatch. Expect receive input shape ' + data_tensor_shape[i] + ' but got ' + curr_input_shape[i])
            else:
                dim_rank += curr_input_shape[axis]
    out_tensor_shape = list(data_tensor_shape)
    out_tensor_shape[axis] = dim_rank
    out_tensor_shape = tuple(out_tensor_shape)
    module = JoinTable(dimension=axis + 1, n_input_dims=len(data_tensor_shape))(prev_modules)
    return (module, [out_tensor_shape])

def constant(inputs, prev_modules, attrs, outputs):
    if False:
        i = 10
        return i + 15
    value = parse_tensor_data(attrs.get('value'))
    out_tensor_shape = value.shape
    module = Constant(value)(prev_modules[0])
    return (module, [out_tensor_shape])

def conv(inputs, prev_modules, attrs, outputs):
    if False:
        i = 10
        return i + 15
    auto_pad = attrs.get('auto_pad', 'NOTSET')
    (padW, padH) = map(int, attrs.get('pads', (0, 0))[:2])
    (kernelW, kernelH) = map(int, attrs.get('kernel_shape', (0, 0))[:2])
    (strideW, strideH) = map(int, attrs.get('strides', (1, 1))[:2])
    (dilationW, dilationH) = map(int, attrs.get('dilations', (1, 1))[:2])
    group = int(attrs.get('group', 1))
    withBias = len(inputs) == 3 and inputs[2] is not None
    (data_tensor_val, data_tensor_shape) = inputs[0]
    (weight_tensor_val, weight_tensor_shape) = inputs[1]
    bias_tensor_val = None
    if withBias:
        (bias_tensor_val, _) = inputs[2]
    (input_batch_size, n_input_plane) = map(int, data_tensor_shape[:2])
    n_output_plane = weight_tensor_shape[0]
    (input_height, input_width) = data_tensor_shape[-2:]
    output_height = calc_output_shape(input_height, kernelH, padding=padH, stride=strideH)
    output_width = calc_output_shape(input_width, kernelW, padding=padW, stride=strideW)
    out_tensor_shape = (input_batch_size, n_output_plane, output_height, output_width)
    module = SpatialConvolution(n_input_plane=n_input_plane, n_output_plane=n_output_plane, kernel_w=kernelW, kernel_h=kernelH, stride_w=strideW, stride_h=strideH, pad_w=padW, pad_h=padH, n_group=group, init_weight=weight_tensor_val, init_bias=bias_tensor_val, with_bias=withBias)(prev_modules[0])
    return (module, [out_tensor_shape])

def gather(inputs, prev_modules, attrs, outputs):
    if False:
        i = 10
        return i + 15
    axis = int(attrs.get('axis', 0))
    if axis != 0:
        invalidInputError(False, 'Gather layer axis value')
    (data_tensor_val, data_tensor_shape) = inputs[0]
    (indices_val, indices) = inputs[1]
    out_tensor_shape = tuple(data_tensor_shape[:axis] + indices + data_tensor_shape[axis + 1:])
    module = Gather()(prev_modules)
    return (module, [out_tensor_shape])

def gemm(inputs, prev_modules, attrs, outputs):
    if False:
        return 10
    alpha = float(attrs.get('alpha', 1.0))
    beta = float(attrs.get('beta', 1.0))
    trans_a = int(attrs.get('transA', 0))
    trans_b = int(attrs.get('transB', 0))
    (_, tensor_a_shape) = inputs[0]
    (tensor_b_val, tensor_b_shape) = inputs[1]
    (tensor_c_val, tensor_c_shape) = inputs[2]
    module = Gemm(alpha=alpha, beta=beta, trans_a=trans_a, trans_b=trans_b, matrix_b=tensor_b_val, matrix_c=tensor_c_val)(prev_modules)
    return (module, [tensor_c_shape])

def max_pool(inputs, prev_modules, attrs, outputs):
    if False:
        while True:
            i = 10
    auto_pad = attrs.get('auto_pad', 'NOTSET')
    (kernelW, kernelH) = map(int, attrs.get('kernel_shape')[:2])
    (strideW, strideH) = map(int, attrs.get('strides', (1, 1))[:2])
    (dilationW, dilationH) = map(int, attrs.get('dilations', (1, 1))[:2])
    (padW, padH) = map(int, attrs.get('pads', (0, 0))[:2])
    ceil_mode = True if attrs.get('ceil_mode', 0) != 0 else False
    storage_order = int(attrs.get('storage_order', 0))
    (_, data_tensor_shape) = inputs[0]
    (input_width, input_height) = data_tensor_shape[-2:]
    output_width = calc_output_shape(input_width, kernelW, padding=padW, stride=strideW, dilation=dilationW, ceil_mode=ceil_mode)
    output_height = calc_output_shape(input_height, kernelH, padding=padH, stride=strideH, dilation=dilationH, ceil_mode=ceil_mode)
    out_tensor_shape_list = list(data_tensor_shape)
    out_tensor_shape_list[2] = output_height
    out_tensor_shape_list[3] = output_width
    out_tensor_shape = tuple(out_tensor_shape_list)
    module = SpatialMaxPooling(kw=kernelW, kh=kernelH, dw=strideW, dh=strideH, pad_w=padW, pad_h=padH, to_ceil=ceil_mode)(prev_modules[0])
    return (module, [out_tensor_shape])

def relu(inputs, prev_modules, attrs, outputs):
    if False:
        i = 10
        return i + 15
    (_, data_tensor_shape) = inputs[0]
    output_shape = data_tensor_shape
    module = ReLU()(prev_modules[0])
    return (module, [output_shape])

def reshape(inputs, prev_modules, attrs, outputs):
    if False:
        return 10
    (_, data_tensor_shape) = inputs[0]
    (shape_tensor_val, _) = inputs[1]
    shape_arry = None
    if shape_tensor_val is not None:
        shape_arry = np.squeeze(shape_tensor_val).astype(int).tolist()
    module = Reshape(shape_arry)(prev_modules)
    return (module, [shape_tensor_val])

def shape(inputs, prev_modules, attrs, outputs):
    if False:
        return 10
    (_, data_tensor_shape) = inputs[0]
    module = Shape()(prev_modules[0])
    return (module, [(len(data_tensor_shape),)])

def softmax(inputs, prev_modules, attrs, outputs):
    if False:
        for i in range(10):
            print('nop')
    (_, data_tensor_shape) = inputs[0]
    out_tensor_shape = data_tensor_shape
    axis = int(attrs.get('axis', 1))
    module = SoftMax()(prev_modules[0])
    return (module, [out_tensor_shape])

def _sum(inputs, prev_modules, attrs, outputs):
    if False:
        while True:
            i = 10
    (_, data_tensor_shape) = inputs[0]
    out_tensor_shape = data_tensor_shape
    module = CAddTable()(prev_modules)
    return (module, [data_tensor_shape])

def unsqueeze(inputs, prev_modules, attrs, outputs):
    if False:
        return 10
    axes = list(map(int, attrs.get('axes')))
    (data_tensor_val, data_tensor_shape) = inputs[0]
    out_tensor_shape = list(data_tensor_shape)
    for idx in axes:
        out_tensor_shape.insert(idx, 1)
    out_tensor_shape = tuple(out_tensor_shape)
    module = Unsqueeze(axes[0], len(data_tensor_shape))(prev_modules)
    return (module, [out_tensor_shape])
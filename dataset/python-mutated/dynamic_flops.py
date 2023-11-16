import warnings
import numpy as np
import paddle
from paddle import nn
from paddle.jit.dy2static.program_translator import unwrap_decorators
from .static_flops import Table, static_flops
__all__ = []

def flops(net, input_size, custom_ops=None, print_detail=False):
    if False:
        for i in range(10):
            print('nop')
    "Print a table about the FLOPs of network.\n\n    Args:\n        net (paddle.nn.Layer||paddle.static.Program): The network which could be a instance of paddle.nn.Layer in\n                    dygraph or paddle.static.Program in static graph.\n        input_size (list): size of input tensor. Note that the batch_size in argument ``input_size`` only support 1.\n        custom_ops (A dict of function, optional): A dictionary which key is the class of specific operation such as\n                    paddle.nn.Conv2D and the value is the function used to count the FLOPs of this operation. This\n                    argument only work when argument ``net`` is an instance of paddle.nn.Layer. The details could be found\n                    in following example code. Default is None.\n        print_detail (bool, optional): Whether to print the detail information, like FLOPs per layer, about the net FLOPs.\n                    Default is False.\n\n    Returns:\n        Int: A number about the FLOPs of total network.\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n            >>> import paddle.nn as nn\n\n            >>> class LeNet(nn.Layer):\n            ...     def __init__(self, num_classes=10):\n            ...         super().__init__()\n            ...         self.num_classes = num_classes\n            ...         self.features = nn.Sequential(\n            ...             nn.Conv2D(1, 6, 3, stride=1, padding=1),\n            ...             nn.ReLU(),\n            ...             nn.MaxPool2D(2, 2),\n            ...             nn.Conv2D(6, 16, 5, stride=1, padding=0),\n            ...             nn.ReLU(),\n            ...             nn.MaxPool2D(2, 2))\n            ...\n            ...         if num_classes > 0:\n            ...             self.fc = nn.Sequential(\n            ...                 nn.Linear(400, 120),\n            ...                 nn.Linear(120, 84),\n            ...                 nn.Linear(84, 10))\n            ...\n            ...     def forward(self, inputs):\n            ...         x = self.features(inputs)\n            ...\n            ...         if self.num_classes > 0:\n            ...             x = paddle.flatten(x, 1)\n            ...             x = self.fc(x)\n            ...         return x\n            ...\n            >>> lenet = LeNet()\n            >>> # m is the instance of nn.Layer, x is the intput of layer, y is the output of layer.\n            >>> def count_leaky_relu(m, x, y):\n            ...     x = x[0]\n            ...     nelements = x.numel()\n            ...     m.total_ops += int(nelements)\n            ...\n            >>> FLOPs = paddle.flops(lenet,\n            ...                      [1, 1, 28, 28],\n            ...                      custom_ops= {nn.LeakyReLU: count_leaky_relu},\n            ...                      print_detail=True)\n            >>> print(FLOPs)\n            <class 'paddle.nn.layer.conv.Conv2D'>'s flops has been counted\n            <class 'paddle.nn.layer.activation.ReLU'>'s flops has been counted\n            Cannot find suitable count function for <class 'paddle.nn.layer.pooling.MaxPool2D'>. Treat it as zero FLOPs.\n            <class 'paddle.nn.layer.common.Linear'>'s flops has been counted\n            +--------------+-----------------+-----------------+--------+--------+\n            |  Layer Name  |   Input Shape   |   Output Shape  | Params | Flops  |\n            +--------------+-----------------+-----------------+--------+--------+\n            |   conv2d_0   |  [1, 1, 28, 28] |  [1, 6, 28, 28] |   60   | 47040  |\n            |   re_lu_0    |  [1, 6, 28, 28] |  [1, 6, 28, 28] |   0    |   0    |\n            | max_pool2d_0 |  [1, 6, 28, 28] |  [1, 6, 14, 14] |   0    |   0    |\n            |   conv2d_1   |  [1, 6, 14, 14] | [1, 16, 10, 10] |  2416  | 241600 |\n            |   re_lu_1    | [1, 16, 10, 10] | [1, 16, 10, 10] |   0    |   0    |\n            | max_pool2d_1 | [1, 16, 10, 10] |  [1, 16, 5, 5]  |   0    |   0    |\n            |   linear_0   |     [1, 400]    |     [1, 120]    | 48120  | 48000  |\n            |   linear_1   |     [1, 120]    |     [1, 84]     | 10164  | 10080  |\n            |   linear_2   |     [1, 84]     |     [1, 10]     |  850   |  840   |\n            +--------------+-----------------+-----------------+--------+--------+\n            Total Flops: 347560     Total Params: 61610\n            347560\n    "
    if isinstance(net, nn.Layer):
        (_, net.forward) = unwrap_decorators(net.forward)
        inputs = paddle.randn(input_size)
        return dynamic_flops(net, inputs=inputs, custom_ops=custom_ops, print_detail=print_detail)
    elif isinstance(net, paddle.static.Program):
        return static_flops(net, print_detail=print_detail)
    else:
        warnings.warn('Your model must be an instance of paddle.nn.Layer or paddle.static.Program.')
        return -1

def count_convNd(m, x, y):
    if False:
        print('Hello World!')
    x = x[0]
    kernel_ops = np.prod(m.weight.shape[2:])
    bias_ops = 1 if m.bias is not None else 0
    total_ops = int(y.numel()) * (x.shape[1] / m._groups * kernel_ops + bias_ops)
    m.total_ops += abs(int(total_ops))

def count_leaky_relu(m, x, y):
    if False:
        return 10
    x = x[0]
    nelements = x.numel()
    m.total_ops += int(nelements)

def count_bn(m, x, y):
    if False:
        print('Hello World!')
    x = x[0]
    nelements = x.numel()
    if not m.training:
        total_ops = 2 * nelements
    m.total_ops += abs(int(total_ops))

def count_linear(m, x, y):
    if False:
        return 10
    total_mul = m.weight.shape[0]
    num_elements = y.numel()
    total_ops = total_mul * num_elements
    m.total_ops += abs(int(total_ops))

def count_avgpool(m, x, y):
    if False:
        while True:
            i = 10
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    m.total_ops += int(total_ops)

def count_adap_avgpool(m, x, y):
    if False:
        for i in range(10):
            print('nop')
    kernel = np.array(x[0].shape[2:]) // np.array(y.shape[2:])
    total_add = np.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    m.total_ops += abs(int(total_ops))

def count_zero_ops(m, x, y):
    if False:
        i = 10
        return i + 15
    m.total_ops += 0

def count_parameters(m, x, y):
    if False:
        return 10
    total_params = 0
    for p in m.parameters():
        total_params += p.numel()
    m.total_params[0] = abs(int(total_params))

def count_io_info(m, x, y):
    if False:
        print('Hello World!')
    m.register_buffer('input_shape', paddle.to_tensor(x[0].shape))
    if isinstance(y, (list, tuple)):
        m.register_buffer('output_shape', paddle.to_tensor(y[0].shape))
    else:
        m.register_buffer('output_shape', paddle.to_tensor(y.shape))
register_hooks = {nn.Conv1D: count_convNd, nn.Conv2D: count_convNd, nn.Conv3D: count_convNd, nn.Conv1DTranspose: count_convNd, nn.Conv2DTranspose: count_convNd, nn.Conv3DTranspose: count_convNd, nn.layer.norm.BatchNorm2D: count_bn, nn.BatchNorm: count_bn, nn.ReLU: count_zero_ops, nn.ReLU6: count_zero_ops, nn.LeakyReLU: count_leaky_relu, nn.Linear: count_linear, nn.Dropout: count_zero_ops, nn.AvgPool1D: count_avgpool, nn.AvgPool2D: count_avgpool, nn.AvgPool3D: count_avgpool, nn.AdaptiveAvgPool1D: count_adap_avgpool, nn.AdaptiveAvgPool2D: count_adap_avgpool, nn.AdaptiveAvgPool3D: count_adap_avgpool}

def dynamic_flops(model, inputs, custom_ops=None, print_detail=False):
    if False:
        i = 10
        return i + 15
    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if False:
            for i in range(10):
                print('nop')
        if len(list(m.children())) > 0:
            return
        m.register_buffer('total_ops', paddle.zeros([1], dtype='int64'))
        m.register_buffer('total_params', paddle.zeros([1], dtype='int64'))
        m_type = type(m)
        flops_fn = None
        if m_type in custom_ops:
            flops_fn = custom_ops[m_type]
            if m_type not in types_collection:
                print(f'Customize Function has been applied to {m_type}')
        elif m_type in register_hooks:
            flops_fn = register_hooks[m_type]
            if m_type not in types_collection:
                print(f"{m_type}'s flops has been counted")
        elif m_type not in types_collection:
            print(f'Cannot find suitable count function for {m_type}. Treat it as zero FLOPs.')
        if flops_fn is not None:
            flops_handler = m.register_forward_post_hook(flops_fn)
            handler_collection.append(flops_handler)
        params_handler = m.register_forward_post_hook(count_parameters)
        io_handler = m.register_forward_post_hook(count_io_info)
        handler_collection.append(params_handler)
        handler_collection.append(io_handler)
        types_collection.add(m_type)
    training = model.training
    model.eval()
    model.apply(add_hooks)
    with paddle.framework.no_grad():
        model(inputs)
    total_ops = 0
    total_params = 0
    for m in model.sublayers():
        if len(list(m.children())) > 0:
            continue
        if {'total_ops', 'total_params', 'input_shape', 'output_shape'}.issubset(set(m._buffers.keys())):
            total_ops += m.total_ops
            total_params += m.total_params
    if training:
        model.train()
    for handler in handler_collection:
        handler.remove()
    table = Table(['Layer Name', 'Input Shape', 'Output Shape', 'Params', 'Flops'])
    for (n, m) in model.named_sublayers():
        if len(list(m.children())) > 0:
            continue
        if {'total_ops', 'total_params', 'input_shape', 'output_shape'}.issubset(set(m._buffers.keys())):
            table.add_row([m.full_name(), list(m.input_shape.numpy()), list(m.output_shape.numpy()), int(m.total_params), int(m.total_ops)])
            m._buffers.pop('total_ops')
            m._buffers.pop('total_params')
            m._buffers.pop('input_shape')
            m._buffers.pop('output_shape')
    if print_detail:
        table.print_table()
    print(f'Total Flops: {int(total_ops)}     Total Params: {int(total_params)}')
    return int(total_ops)
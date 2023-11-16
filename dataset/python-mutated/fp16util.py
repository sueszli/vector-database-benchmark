import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class tofp16(nn.Module):
    """
    Model wrapper that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(tofp16, self).__init__()

    def forward(self, input):
        if False:
            print('Hello World!')
        return input.half()

def copy_in_params(net, params):
    if False:
        print('Hello World!')
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)

def set_grad(params, params_with_grad):
    if False:
        i = 10
        return i + 15
    for (param, param_w_grad) in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
        param.grad.data.copy_(param_w_grad.grad.data)

def BN_convert_float(module):
    if False:
        print('Hello World!')
    "\n    Designed to work with network_to_half.\n    BatchNorm layers need parameters in single precision.\n    Find all layers and convert them back to float. This can't\n    be done with built in .apply as that function will apply\n    fn to all modules, parameters, and buffers. Thus we wouldn't\n    be able to guard the float conversion based on the module type.\n    "
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module

def network_to_half(network):
    if False:
        while True:
            i = 10
    '\n    Convert model to half precision in a batchnorm-safe way.\n    '
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))

def backwards_debug_hook(grad):
    if False:
        for i in range(10):
            print('nop')
    print('Uh oh, main_params is receiving a gradient in the backward pass!')

def create_main_params(model):
    if False:
        while True:
            i = 10
    main_params = _flatten_dense_tensors([param.data for param in model.parameters()]).float()
    main_params = torch.nn.Parameter(main_params)
    main_params.requires_grad = True
    if main_params.grad is None:
        main_params.grad = main_params.new(*main_params.size())
    return main_params

def model_grads_to_main_grads(model, main_params):
    if False:
        i = 10
        return i + 15
    main_params.grad.data.copy_(_flatten_dense_tensors([p.grad.data for p in model.parameters() if p.requires_grad]))

def main_params_to_model_params(model, main_params):
    if False:
        i = 10
        return i + 15
    params = [param.data for param in model.parameters()]
    for (param, main) in zip(params, _unflatten_dense_tensors(main_params.data, params)):
        param.copy_(main)

def params_to_type(params, totype):
    if False:
        return 10
    new_params = []
    for param in params:
        new_params.append(param.type(totype))
    return new_params

def params_to_fp16(params):
    if False:
        i = 10
        return i + 15
    return params_to_type(params, torch.cuda.HalfTensor)

def params_to_fp32(params):
    if False:
        for i in range(10):
            print('nop')
    return params_to_type(params, torch.cuda.FloatTensor)

def clone_params(net):
    if False:
        print('Hello World!')
    new_params = []
    for param in list(net.parameters()):
        new_params.append(param.data.clone())
    return new_params

def clone_grads(net):
    if False:
        for i in range(10):
            print('nop')
    new_params = []
    for param in list(net.parameters()):
        new_params.append(param.grad.data.clone())
    return new_params

def copy_into_params(net, input_tens):
    if False:
        print('Hello World!')
    net_params = list(net.parameters())
    for i in range(len(input_tens)):
        net_params[i].data.copy_(input_tens[i])

def copy_in_grads(params, params_with_grad):
    if False:
        for i in range(10):
            print('nop')
    for (param, param_w_grad) in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
        param.grad.data.copy_(param_w_grad.grad.data)

class DynamicLossScaler:

    def __init__(self, init_scale=2.0 ** 15, scale_factor=2.0, scale_window=100):
        if False:
            for i in range(10):
                print('nop')
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    def has_overflow(self, tensors):
        if False:
            return 10
        try:
            for tens in tensors:
                if tens is None:
                    continue
                if DynamicLossScaler._has_inf_or_nan(tens):
                    return True
        except TypeError:
            return DynamicLossScaler._has_inf_or_nan(tensors)
        return False

    def _has_inf_or_nan(x):
        if False:
            return 10
        if torch.is_tensor(x):
            max_val = x.abs().max()
        else:
            max_val = x
        if max_val == float('inf'):
            return True
        nan_count = torch.sum(x != x)
        return nan_count > 0

    def update_scale(self, overflow):
        if False:
            for i in range(10):
                print('nop')
        if overflow:
            self.cur_scale /= self.scale_factor
            self.last_overflow_iter = self.cur_iter
        elif (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
            self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        if False:
            return 10
        return self.cur_scale
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class tofp16(nn.Module):
    """
    Utility module that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super(tofp16, self).__init__()

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        return input.half()

def BN_convert_float(module):
    if False:
        while True:
            i = 10
    '\n    Utility function for network_to_half().\n\n    Retained for legacy purposes.\n    '
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module

def network_to_half(network):
    if False:
        i = 10
        return i + 15
    '\n    Convert model to half precision in a batchnorm-safe way.\n\n    Retained for legacy purposes. It is recommended to use FP16Model.\n    '
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))

def convert_module(module, dtype):
    if False:
        return 10
    "\n    Converts a module's immediate parameters and buffers to dtype.\n    "
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data.to(dtype=dtype)
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data.to(dtype=dtype)
    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data.to(dtype=dtype)

def convert_network(network, dtype):
    if False:
        return 10
    "\n    Converts a network's parameters and buffers to dtype.\n    "
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype)
        if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
            module.flatten_parameters()
    return network

class FP16Model(nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network):
        if False:
            i = 10
            return i + 15
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)

    def forward(self, *inputs):
        if False:
            return 10
        inputs = tuple((t.half() for t in inputs))
        return self.network(*inputs)

def backwards_debug_hook(grad):
    if False:
        for i in range(10):
            print('nop')
    raise RuntimeError('master_params recieved a gradient in the backward pass!')

def prep_param_lists(model, flat_master=False):
    if False:
        print('Hello World!')
    "\n    Creates a list of FP32 master parameters for a given model, as in\n    `Training Neural Networks with Mixed Precision:  Real Examples`_.\n\n    Args:\n        model (torch.nn.Module): Existing Pytorch model\n        flat_master (bool, optional, default=False):  Flatten the master parameters into a single tensor, as a performance optimization.\n    Returns:\n        A tuple (``model_params``, ``master_params``). ``model_params`` is a list of the model's parameters for later use with :func:`model_grads_to_master_grads` and :func:`master_params_to_model_params`.  ``master_params`` is a list of FP32 master gradients.  If ``flat_master=True``, ``master_params`` will be a list with one element.\n\n    Example::\n\n        model_params, master_params = prep_param_lists(model)\n\n    .. warning::\n        Currently, if ``flat_master=True``, all the model's parameters must be the same type.  If the model has parameters of different types, use ``flat_master=False``, or use :class:`FP16_Optimizer`.\n\n    .. _`Training Neural Networks with Mixed Precision:  Real Examples`:\n        http://on-demand.gputechconf.com/gtc/2018/video/S81012/\n    "
    model_params = [param for param in model.parameters() if param.requires_grad]
    if flat_master:
        try:
            master_params = _flatten_dense_tensors([param.data for param in model_params]).float()
        except:
            print('Error in prep_param_lists:  model may contain a mixture of parameters of different types.  Use flat_master=False, or use F16_Optimizer.')
            raise
        master_params = torch.nn.Parameter(master_params)
        master_params.requires_grad = True
        if master_params.grad is None:
            master_params.grad = master_params.new(*master_params.size())
        return (model_params, [master_params])
    else:
        master_params = [param.clone().float().detach() for param in model_params]
        for param in master_params:
            param.requires_grad = True
        return (model_params, master_params)

def model_grads_to_master_grads(model_params, master_params, flat_master=False):
    if False:
        print('Hello World!')
    '\n    Copy model gradients to master gradients.  \n\n    Args:\n        model_params:  List of model parameters created by :func:`prep_param_lists`.\n        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.  If ``master_params`` was created with ``flat_master=True``, ``flat_master=True`` should also be supplied to :func:`model_grads_to_master_grads`.\n    '
    if flat_master:
        master_params[0].grad.data.copy_(_flatten_dense_tensors([p.grad.data for p in model_params]))
    else:
        for (model, master) in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None:
                    master.grad = Variable(master.data.new(*master.data.size()))
                master.grad.data.copy_(model.grad.data)
            else:
                master.grad = None

def master_params_to_model_params(model_params, master_params, flat_master=False):
    if False:
        while True:
            i = 10
    '\n    Copy master parameters to model parameters.\n\n    Args:\n        model_params:  List of model parameters created by :func:`prep_param_lists`.\n        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.  If ``master_params`` was created with ``flat_master=True``, ``flat_master=True`` should also be supplied to :func:`master_params_to_model_params`.\n    '
    if flat_master:
        for (model, master) in zip(model_params, _unflatten_dense_tensors(master_params[0].data, model_params)):
            model.data.copy_(master)
    else:
        for (model, master) in zip(model_params, master_params):
            model.data.copy_(master.data)

def to_python_float(t):
    if False:
        return 10
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
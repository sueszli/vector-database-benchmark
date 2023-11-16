import torch
import onnx
import onnx.numpy_helper
from ..utils import set_nested_attr
'\nThe main function of this page is to convert pytorch model to onnx model.\nConvertion from pytorch model to onnx model is primary so that a critical\nproblem is caused that Layer name of pytorch model fail to convert to onnx\nlayer name directly. To solve it, we wrap pytorch model in new wrapper which\nmultiply bits number and input before computation of each op. Only in this\nway can onnx model get bits number of corresponded layer.\n'

class LayernameModuleWrapper(torch.nn.Module):

    def __init__(self, module, module_bits) -> None:
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        module : torch.nn.Module\n            Layer module of pytorch model\n        module_bits : int\n            Bits width setting for module\n        '
        super().__init__()
        self.module = module
        self.module_bits = module_bits

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        inputs = inputs * self.module_bits
        inputs = self.module(inputs)
        return inputs

def unwrapper(model_onnx, index2name, config):
    if False:
        i = 10
        return i + 15
    '\n    Fill onnx config and remove wrapper node in onnx\n\n    Parameters\n    ----------\n    model_onnx : onnx model\n        Onnx model which is converted from pytorch model\n    index2name : dict\n        Dictionary of layer index and name\n    config : dict\n        Config recording name of layers and calibration parameters\n\n    Returns\n    -------\n    onnx model\n        Onnx model which is converted from pytorch model\n    dict\n        The configuration of onnx model layers and calibration parameters\n    '
    support_op = ['Gemm', 'Conv', 'Relu', 'Clip', 'MaxP']
    idx = 0
    onnx_config = {}
    while idx < len(model_onnx.graph.node):
        nd = model_onnx.graph.node[idx]
        if nd.name[0:4] in support_op and idx > 1:
            const_nd = model_onnx.graph.node[idx - 2]
            mul_nd = model_onnx.graph.node[idx - 1]
            index = int(onnx.numpy_helper.to_array(const_nd.attribute[0].t))
            if index != -1:
                name = index2name[index]
                onnx_config[nd.name] = config[name]
            nd.input[0] = mul_nd.input[0]
            model_onnx.graph.node.remove(const_nd)
            model_onnx.graph.node.remove(mul_nd)
            idx = idx - 2
        idx = idx + 1
    return (model_onnx, onnx_config)

def torch_to_onnx(model, config, input_shape, model_path, input_names, output_names):
    if False:
        while True:
            i = 10
    '\n    Convert torch model to onnx model and get layer bits config of onnx model.\n\n    Parameters\n    ----------\n    model : pytorch model\n        The model to speedup by quantization\n    config : dict\n        Config recording bits number and name of layers\n    input_shape : tuple\n        The input shape of model, shall pass it to torch.onnx.export\n    model_path : str\n        The path user want to store onnx model which is converted from pytorch model\n    input_names : list\n        Input name of onnx model providing for torch.onnx.export to generate onnx model\n    output_name : list\n        Output name of onnx model providing for torch.onnx.export to generate onnx model\n\n    Returns\n    -------\n    onnx model\n        Onnx model which is converted from pytorch model\n    dict\n        The configuration of onnx model layers and calibration parameters\n    '
    support_op = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.MaxPool2d]
    index2name = {}
    name2index = {}
    if config is not None:
        for (i, name) in enumerate(config.keys()):
            index2name[i * 2] = name
            name2index[name] = i * 2
    for (name, module) in model.named_modules():
        if config is not None and name in config:
            assert type(module) in support_op
            wrapper_module = LayernameModuleWrapper(module, name2index[name])
            set_nested_attr(model, name, wrapper_module)
        elif type(module) in support_op:
            wrapper_module = LayernameModuleWrapper(module, -1)
            set_nested_attr(model, name, wrapper_module)
    device = torch.device('cpu')
    dummy_input = torch.randn(input_shape)
    dummy_input = dummy_input.to(device)
    model.to(device)
    torch.onnx.export(model, dummy_input, model_path, verbose=False, input_names=input_names, output_names=output_names, export_params=True)
    model_onnx = onnx.load(model_path)
    (model_onnx, onnx_config) = unwrapper(model_onnx, index2name, config)
    onnx.save(model_onnx, model_path)
    onnx.checker.check_model(model_onnx)
    return (model_onnx, onnx_config)
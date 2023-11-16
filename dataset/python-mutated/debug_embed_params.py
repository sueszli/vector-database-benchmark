import sys
import caffe2.python.onnx.backend as c2
import onnx
import pytorch_test_common
import torch
import torch.jit
from torch.autograd import Variable
torch.set_default_tensor_type('torch.FloatTensor')
try:
    import torch
except ImportError:
    print('Cannot import torch, hence caffe2-torch test will not run.')
    sys.exit(0)

def run_embed_params(proto, model, input, state_dict=None, use_gpu=True):
    if False:
        i = 10
        return i + 15
    '\n    This is only a helper debug function so we can test embed_params=False\n    case as well on pytorch front\n    This should likely be removed from the release version of the code\n    '
    device = 'CPU'
    if use_gpu:
        device = 'CUDA'
    model_def = onnx.ModelProto.FromString(proto)
    onnx.checker.check_model(model_def)
    prepared = c2.prepare(model_def, device=device)
    if state_dict:
        parameters = []
        for k in model.state_dict():
            if k in state_dict:
                parameters.append(state_dict[k])
    else:
        parameters = list(model.state_dict().values())
    W = {}
    for (k, v) in zip(model_def.graph.input, pytorch_test_common.flatten((input, parameters))):
        if isinstance(v, Variable):
            W[k.name] = v.data.cpu().numpy()
        else:
            W[k.name] = v.cpu().numpy()
    caffe2_out = prepared.run(inputs=W)
    return caffe2_out
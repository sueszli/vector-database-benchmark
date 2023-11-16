import sys
import io
import torch
from bigdl.dllib.nn.layer import Layer
from bigdl.dllib.utils.common import JTensor
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.orca.torch.utils import trainable_param
from bigdl.orca.torch import zoo_pickle_module
from importlib.util import find_spec
from bigdl.dllib.utils.log4Error import *
from deprecated import deprecated
if sys.version < '3.7':
    print('WARN: detect python < 3.7, if you meet zlib not available ' + 'exception on yarn, please update your python to 3.7')
if find_spec('jep') is None:
    invalidInputError(False, 'jep not found, please install jep first.')

@deprecated(version='2.3.0', reason='Please use spark engine and ray engine.')
class TorchModel(Layer):
    """
    TorchModel wraps a PyTorch model as a single layer, thus the PyTorch model can be used for
    distributed inference or training.
    """

    def __init__(self, jvalue, module_bytes, bigdl_type='float'):
        if False:
            return 10
        self.value = jvalue
        self.module_bytes = module_bytes
        self.bigdl_type = bigdl_type

    @staticmethod
    def from_value(model_value):
        if False:
            return 10
        model_bytes = callZooFunc('float', 'getTorchModelBytes', model_value)
        net = TorchModel(model_value, model_bytes)
        return net

    @staticmethod
    def from_pytorch(model):
        if False:
            return 10
        '\n        Create a TorchModel directly from PyTorch model, e.g. model in torchvision.models.\n        :param model: a PyTorch model, or a function to create PyTorch model\n        '
        weights = []
        import types
        if isinstance(model, types.FunctionType) or isinstance(model, type):
            for param in trainable_param(model()):
                weights.append(param.view(-1))
        else:
            for param in trainable_param(model):
                weights.append(param.view(-1))
        flatten_weight = torch.nn.utils.parameters_to_vector(weights).data.numpy()
        bys = io.BytesIO()
        torch.save(model, bys, pickle_module=zoo_pickle_module)
        weights = JTensor.from_ndarray(flatten_weight)
        jvalue = callZooFunc('float', 'createTorchModel', bys.getvalue(), weights)
        net = TorchModel(jvalue, bys.getvalue())
        return net

    def to_pytorch(self):
        if False:
            print('Hello World!')
        '\n        Convert to pytorch model\n        :return: a pytorch model\n        '
        new_weight = self.get_weights()
        invalidInputError(len(new_weight) == 1, "TorchModel's weights should be one tensor")
        m = torch.load(io.BytesIO(self.module_bytes), pickle_module=zoo_pickle_module)
        import types
        if isinstance(m, types.FunctionType) or isinstance(m, type):
            m = m()
        w = torch.Tensor(new_weight[0])
        torch.nn.utils.vector_to_parameters(w, trainable_param(m))
        new_extra_params = callZooFunc(self.bigdl_type, 'getModuleExtraParameters', self.value)
        if len(new_extra_params) != 0:
            idx = 0
            for named_buffer in m.named_buffers():
                named_buffer[1].copy_(torch.reshape(torch.Tensor(new_extra_params[idx].to_ndarray()), named_buffer[1].size()))
                idx += 1
        return m

    def saveModel(self, path, over_write=False):
        if False:
            i = 10
            return i + 15
        from bigdl.dllib.utils.common import callBigDlFunc
        callBigDlFunc(self.bigdl_type, 'modelSave', self.value, path, over_write)

    @staticmethod
    def loadModel(path, bigdl_type='float'):
        if False:
            print('Hello World!')
        from bigdl.dllib.utils.common import callBigDlFunc
        jmodel = callBigDlFunc(bigdl_type, 'loadBigDL', path)
        return Layer.of(jmodel)
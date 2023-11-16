from pathlib import Path
import yaml
import os
import operator
import torch
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from ..core import version as inc_version
from neural_compressor.utils.pytorch import load
from neural_compressor.model.model import PyTorchModel
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.pytorch.context_manager import generate_context_manager
from bigdl.nano.pytorch.lightning import LightningModule
from bigdl.nano.utils.common import compare_version

class PytorchQuantizedModel(AcceleratedLightningModule):

    def __init__(self, model, thread_num=None):
        if False:
            print('Hello World!')
        super().__init__(model.model)
        self.quantized = model
        self.thread_num = thread_num
        self._nano_context_manager = generate_context_manager(accelerator=None, precision='int8', thread_num=thread_num)

    @property
    def _nargs(self):
        if False:
            print('Hello World!')
        return -1

    @property
    def status(self):
        if False:
            return 10
        status = super().status
        status.update({'thread_num': self.thread_num})
        return status

    @staticmethod
    def _load(path, model, example_inputs=None):
        if False:
            i = 10
            return i + 15
        status = PytorchQuantizedModel._load_status(path)
        invalidInputError(model is not None, errMsg='FP32 model is required to create a quantized model.')
        ipex_quantization = False
        if isinstance(path, dict) and 'best_configure.json' in path:
            ipex_quantization = True
            import intel_extension_for_pytorch as ipex
        if not isinstance(path, dict) and os.path.exists(os.path.join(path, 'best_configure.json')):
            ipex_quantization = True
            import intel_extension_for_pytorch as ipex
        if ipex_quantization:
            invalidInputError(example_inputs is not None, 'For INC ipex quantizated model, you need to set input_sample when loading model.')
        if isinstance(path, dict):
            weights_file = path['best_model.pt']
            stat_dict = torch.jit.load(weights_file)
            if isinstance(model, LightningModule) and compare_version('neural_compressor', operator.ge, '2.0'):
                qmodel = PyTorchModel(load(stat_dict, model.model, example_inputs=example_inputs))
            else:
                qmodel = PyTorchModel(load(stat_dict, model, example_inputs=example_inputs))
        elif isinstance(model, LightningModule) and compare_version('neural_compressor', operator.ge, '2.0'):
            qmodel = PyTorchModel(load(path, model.model, example_inputs=example_inputs))
        else:
            qmodel = PyTorchModel(load(path, model, example_inputs=example_inputs))
        from packaging import version
        if version.parse(inc_version) < version.parse('1.11'):
            if isinstance(path, dict):
                tune_cfg = yaml.safe_load(path['best_configure.yaml'])
                qmodel.tune_cfg = tune_cfg
            else:
                path = Path(path)
                tune_cfg_file = path / 'best_configure.yaml'
                with open(tune_cfg_file, 'r') as f:
                    tune_cfg = yaml.safe_load(f)
                    qmodel.tune_cfg = tune_cfg
        thread_num = status.get('thread_num', None)
        if thread_num == {}:
            thread_num = None
        if thread_num is not None:
            thread_num = int(status['thread_num'])
        return PytorchQuantizedModel(qmodel, thread_num=thread_num)

    def _save_model(self, path, compression='fp32'):
        if False:
            for i in range(10):
                print('nop')
        self.quantized.save(path)
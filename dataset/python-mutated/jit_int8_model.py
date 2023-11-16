from bigdl.nano.pytorch.context_manager import generate_context_manager
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.pytorch import jit_convert
from bigdl.nano.utils.common import compare_version
import torch
from torch.ao.quantization import QConfig, get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import operator
from collections.abc import Sequence

class PytorchJITINT8Model(AcceleratedLightningModule):

    def __init__(self, model: torch.nn.Module, calib_data, q_config=None, input_sample=None, channels_last=False, thread_num=None, from_load=False, jit_strict=True, jit_method=None, enable_onednn=False, example_kwarg_inputs=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        This is the accelerated model for pytorch fx quantization and jit.\n        All the external API is based on InferenceOptimizer, so what we have here is\n        basically internal APIs and subject to change.\n\n        :param model: the model(nn.module) to be transform if from_load is False\n               the accelerated model if from_load is True.\n        :param calib_data: calibration data is required for static quantization.\n        :param q_config: We support 2 types of customized quantization config:\n\n               | 1. Qconfig (https://pytorch.org/docs/stable/generated/torch.quantization.\n               | qconfig.QConfig.html#qconfig) is the configuration for how we insert\n               | observers for a particular operator. Quantization preparation function\n               | will instantiate observers multiple times for each of the layers.\n               |\n               | 2. QConfigMapping (https://pytorch.org/docs/stable/generated/torch.ao.\n               | quantization.qconfig_mapping.QConfigMapping.html#qconfigmapping)\n               | (recommended) is a collection of quantization configurations, user\n               | can set the qconfig for each operator (torch op calls, functional\n               | calls, module calls) in the model through qconfig_mapping.\n\n        :param input_sample: torch tensor indicate the data sample to be used\n               for tracing.\n        :param channels_last: if set model and data to be channels-last mode.\n        :param thread_num: the thread num allocated for this model.\n        :param from_load: this will only be set by _load method.\n        :param jit_strict: Whether recording your mutable container types.\n        :param jit_method: use ``jit.trace`` or ``jit.script`` to convert a model\n               to TorchScript.\n        :param enable_onednn: Whether to use PyTorch JIT graph fuser based on\n               oneDNN Graph API, which provides a flexible API for aggressive\n               fusion. Default to ``False``.\n        :param example_kwarg_inputs: keyword arguments of example inputs that will be passed\n               to ``torch.jit.trace``. Default to None. Either this argument or input_sample\n               should be specified when use_jit is ``True`` and torch > 2.0,\n               otherwise will be ignored.\n        '
        super().__init__(model)
        enable_onednn = False
        if from_load:
            self.channels_last = channels_last
            self.jit_strict = jit_strict
            self.jit_method = jit_method
            self.enable_onednn = enable_onednn
            self._nano_context_manager = generate_context_manager(accelerator='jit', precision='int8', thread_num=thread_num, enable_onednn=enable_onednn)
            return
        self.original_state_dict = model.state_dict()
        self.channels_last = channels_last
        self.jit_strict = jit_strict
        self.jit_method = jit_method
        self.enable_onednn = enable_onednn
        self._nano_context_manager = generate_context_manager(accelerator='jit', precision='int8', thread_num=thread_num, enable_onednn=enable_onednn)
        self.thread_num = thread_num
        self.original_model = model
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if q_config is None:
            self.q_config = get_default_qconfig_mapping('fbgemm')
        elif isinstance(q_config, QConfig):
            self.q_config = {'': q_config}
        else:
            self.q_config = q_config
        if input_sample is None:
            input_sample = next(iter(calib_data))
            if isinstance(input_sample, (tuple, list)) and len(input_sample) > 1:
                input_sample = input_sample[0]
                if self.channels_last:
                    if isinstance(input_sample, torch.Tensor):
                        input_sample = input_sample.to(memory_format=torch.channels_last)
                    else:
                        input_sample = tuple(map(lambda x: x.to(memory_format=torch.channels_last), input_sample))
        self.model = prepare_fx(self.model, self.q_config, example_inputs=(input_sample,))
        for x in calib_data:
            if isinstance(x, (tuple, list)) and len(x) > 1:
                x = x[0]
            if isinstance(x, Sequence):
                self.model(*x)
            else:
                self.model(x)
        self.model = convert_fx(self.model)
        with torch.no_grad():
            if example_kwarg_inputs is not None:
                input_sample = None
            self.model = jit_convert(self.model, input_sample, jit_method=jit_method, jit_strict=jit_strict, example_kwarg_inputs=example_kwarg_inputs)
            self.model = torch.jit.freeze(self.model)

    def on_forward_start(self, inputs):
        if False:
            i = 10
            return i + 15
        return inputs

    def forward_step(self, *inputs):
        if False:
            for i in range(10):
                print('nop')
        if self.channels_last:
            inputs = tuple(map(lambda x: x.to(memory_format=torch.channels_last), inputs))
        return self.model(*inputs)

    def on_forward_end(self, outputs):
        if False:
            while True:
                i = 10
        return outputs

    def __getattr__(self, name: str):
        if False:
            print('Hello World!')
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_model, name)

    @property
    def status(self):
        if False:
            return 10
        status = super().status
        status.update({'channels_last': self.channels_last, 'checkpoint': 'ckpt.pth', 'thread_num': self.thread_num, 'jit_strict': self.jit_strict, 'jit_method': self.jit_method, 'enable_onednn': self.enable_onednn})
        return status

    @staticmethod
    def _load(path):
        if False:
            return 10
        status = PytorchJITINT8Model._load_status(path)
        checkpoint_path = path / status['checkpoint']
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model = torch.jit.freeze(model)
        from_load = True
        thread_num = None
        if status['thread_num'] is not None and status['thread_num'] != {}:
            thread_num = int(status['thread_num'])
        return PytorchJITINT8Model(model, calib_data=None, channels_last=status.get('channels_last', False), from_load=from_load, thread_num=thread_num, jit_strict=status.get('jit_strict', True), jit_method=status.get('jit_method', None), enable_onednn=status.get('enable_onednn', False))

    def _save_model(self, path, compression='fp32'):
        if False:
            print('Hello World!')
        torch.jit.save(self.model, path / 'ckpt.pth')
from logging import warning
import torch
import os
from contextlib import redirect_stdout
from bigdl.nano.utils.pytorch import generate_channels_last_available, apply_proper_channels_last
from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_10, TORCH_VERSION_LESS_1_12, transform_state_dict_to_dtype
from bigdl.nano.utils.common import _bf16_checker
from bigdl.nano.pytorch.context_manager import generate_context_manager
invalidInputError(not TORCH_VERSION_LESS_1_10, errMsg='Require torch>=1.10 to convert type as bfloat16.')

class BF16Model(AcceleratedLightningModule):
    """Model of BFloat16 with auto mixed precision."""

    def __init__(self, model, input_sample=None, channels_last=None, channels_last_available=[], thread_num=None, compression='fp32'):
        if False:
            while True:
                i = 10
        '\n        This is the accelerated model for BFloat16 with auto mixed precision.\n\n        :param model: the model(nn.module) to be transform.\n        :param channels_last: if set model and data to be channels-last mode.\n        :param channels_last_available: only passed by _load method,\n               to decide which input can be converted to channels-last mode.\n        :param thread_num: the thread num allocated for this model.\n        '
        super().__init__(model)
        self._bf16_check()
        self.model = model
        self.channels_last = channels_last
        self.thread_num = thread_num
        self.compression = compression
        if self.channels_last is True:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception as _e:
                self.model = self.model.to(memory_format=torch.channels_last_3d)
            if channels_last_available:
                self.channels_last_available = channels_last_available
            else:
                self.channels_last_available = generate_channels_last_available(input_sample)
        else:
            self.channels_last_available = []
        self._nano_context_manager = generate_context_manager(accelerator=None, precision='bf16', thread_num=thread_num)

    @property
    def _has_bf16_isa(self):
        if False:
            print('Hello World!')
        'Indicator to verify if bf16 instructions are available.'
        return _bf16_checker()

    @property
    def _allow_non_bf16(self):
        if False:
            while True:
                i = 10
        "\n        ALLOW_NON_BF16_ISA indicates if we restrict bf16 instructions support to be available.\n        ALLOW_NON_BF16_ISA='1' sometimes helps debug and test cases without AVX512 or AMX\n\n        :return: The bool value of ALLOW_NON_BF16_ISA\n        "
        return os.environ.get('ALLOW_NON_BF16_ISA', None) == '1'

    def _max_bf16_isa(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Run inference once and check the log to confirm if bf16 instructions are used.\n\n        :return:True/False\n        '
        dnnl_log_file = 'dnnl_log.log'
        with redirect_stdout(dnnl_log_file):
            os.environ['DNNL_VERBOSE'] = '1'
            self.bf16_model(*args, **kwargs)
        dnnl_log = ''
        with open(dnnl_log_file, 'r') as f:
            dnnl_log = f.read()
        if os.path.exists(dnnl_log_file):
            os.remove(dnnl_log_file)
        max_bf16_isa = None
        if 'amx_bf16' in dnnl_log:
            max_bf16_isa = 'AMX'
        elif 'avx512_core_bf16' in dnnl_log:
            max_bf16_isa = 'AVX512'
        return max_bf16_isa

    def __getattr__(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def on_forward_start(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        return inputs

    def forward_step(self, *inputs):
        if False:
            print('Hello World!')
        if self.channels_last:
            if not self.channels_last_available:
                self.channels_last_available = generate_channels_last_available(inputs)
            converted_input_length = min(len(self.channels_last_available), len(inputs))
            inputs = tuple(map(lambda idx: apply_proper_channels_last(self.channels_last_available[idx], inputs[idx]), range(converted_input_length))) + inputs[converted_input_length:]
        return self.model(*inputs)

    def on_forward_end(self, outputs):
        if False:
            print('Hello World!')
        return outputs

    def _bf16_check(self):
        if False:
            return 10
        if getattr(self, '_is_bf16', None) is not None:
            return self._is_bf16
        invalidInputError(not TORCH_VERSION_LESS_1_12, errMsg='Require torch>=1.12 to obtain bfloat16 acceleration.')
        if self._has_bf16_isa:
            self._is_bf16 = True
        else:
            self._is_bf16 = False
        if not self._is_bf16:
            warning("Your machine or OS doesn't support BF16 instructions. You are running BF16 model without ISA support, and the performance might be quite low.")

    @property
    def status(self):
        if False:
            i = 10
            return i + 15
        status = super().status
        status.update({'channels_last': self.channels_last, 'channels_last_available': self.channels_last_available, 'checkpoint': 'ckpt.pth', 'thread_num': self.thread_num, 'compression': self.compression})
        return status

    @staticmethod
    def _load(path, model):
        if False:
            i = 10
            return i + 15
        status = BF16Model._load_status(path)
        checkpoint_path = path / status['checkpoint']
        state_dict = torch.load(checkpoint_path)
        model.eval()
        if status['compression'] == 'bf16':
            state_dict = transform_state_dict_to_dtype(state_dict, dtype='fp32')
        model.load_state_dict(state_dict)
        thread_num = status.get('thread_num', None)
        if thread_num == {}:
            thread_num = None
        if thread_num is not None:
            thread_num = int(status['thread_num'])
        return BF16Model(model, channels_last=status['channels_last'], channels_last_available=status['channels_last_available'], thread_num=thread_num, compression=status['compression'])

    def _save_model(self, path, compression='fp32'):
        if False:
            i = 10
            return i + 15
        if compression == 'bf16':
            bf16_model = self.model.bfloat16()
            torch.save(bf16_model.state_dict(), path / 'ckpt.pth')
            self.compression = 'bf16'
            self.model.float()
        else:
            torch.save(self.model.state_dict(), path / 'ckpt.pth')
            self.compression = 'fp32'
import torch
import intel_extension_for_pytorch as ipex
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
SYM_INT4 = ggml_tensor_qtype['sym_int4']
SYM_INT8 = ggml_tensor_qtype['sym_int8']
NF4 = ggml_tensor_qtype['nf4']
NF3 = ggml_tensor_qtype['nf3']
FP8 = ggml_tensor_qtype['fp8']
FP4 = ggml_tensor_qtype['fp4']
MOFQ4 = ggml_tensor_qtype['mixed_fp4']
MOFQ8 = ggml_tensor_qtype['mixed_fp8']

class XMXChecker:

    def __init__(self):
        if False:
            return 10
        self.support_xmx = self.check_xmx()
        self.supported_qtype = [SYM_INT4, SYM_INT8, FP8]

    @staticmethod
    def check_xmx():
        if False:
            while True:
                i = 10
        name = torch.xpu.get_device_name(0)
        return 'Arc(TM)' in name or 'GPU Max' in name or 'GPU Flex' in name

    def check(self, input_tensor: torch.Tensor, qtype: int):
        if False:
            for i in range(10):
                print('nop')
        return self.support_xmx and 1 < input_tensor.shape[0] <= 8 and (qtype in self.supported_qtype)
xmx_checker = XMXChecker()

def use_xmx(input_tensor: torch.Tensor, qtype: int):
    if False:
        i = 10
        return i + 15
    return xmx_checker.check(input_tensor, qtype)
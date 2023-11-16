import os
import multiprocessing
from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
import torch
from typing import Optional, Union
from lm_eval.base import BaseLM
from transformers import AutoTokenizer, LlamaTokenizer

def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if False:
        print('Hello World!')
    'Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig'
    if isinstance(dtype, str) and dtype != 'auto':
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

class BigDLLM(BaseLM):

    def __init__(self, device='xpu', pretrained='gpt2', revision='main', low_cpu_mem_usage=None, subfolder=None, tokenizer=None, batch_size=1, load_in_8bit: Optional[bool]=False, trust_remote_code: Optional[bool]=False, load_in_low_bit=None, dtype: Optional[Union[str, torch.dtype]]='auto'):
        if False:
            return 10
        super().__init__()
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))
        if 'xpu' in device:
            import intel_extension_for_pytorch as ipex
        model = AutoModelForCausalLM.from_pretrained(pretrained, load_in_low_bit=load_in_low_bit, optimize_model=True, trust_remote_code=True, use_cache=True, torch_dtype=_get_dtype(dtype))
        print(model)
        self._device = device
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        if batch_size == 'auto':
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size)

    @property
    def eot_token_id(self):
        if False:
            while True:
                i = 10
        return self.model.token_eos()

    @property
    def max_length(self):
        if False:
            for i in range(10):
                print('nop')
        return 2048

    @property
    def max_gen_toks(self):
        if False:
            while True:
                i = 10
        return 256

    @property
    def batch_size(self):
        if False:
            while True:
                i = 10
        return self.batch_size_per_gpu

    @property
    def device(self):
        if False:
            while True:
                i = 10
        return torch.device(self._device)

    def tok_encode(self, string: str):
        if False:
            while True:
                i = 10
        input_ids = self.tokenizer.encode(string)
        return input_ids

    def tok_decode(self, tokens):
        if False:
            while True:
                i = 10
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _model_call(self, inps):
        if False:
            i = 10
            return i + 15
        '\n        inps: a torch tensor of shape [batch, sequence]\n        the size of sequence may vary from call to call\n\n        returns: a torch tensor of shape [batch, sequence, vocab] with the\n        logits returned from the model\n        '
        with torch.inference_mode():
            inps = inps.to(self.device)
            res = self.model(inps)[0]
            return res

    def _model_generate(self, context, max_length, eos_token_id):
        if False:
            return 10
        return self.model(context, max_tokens=max_length, stop=['Q:', '\n'], echo=True)
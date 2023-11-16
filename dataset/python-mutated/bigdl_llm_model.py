from fastchat.model.model_adapter import register_model_adapter, BaseModelAdapter, ChatGLMAdapter
from fastchat.modules.gptq import GptqConfig, load_gptq_quantized
import accelerate
from fastchat.modules.awq import AWQConfig, load_awq_quantized
from fastchat.model.model_adapter import get_model_adapter, raise_warning_for_incompatible_cpu_offloading_configuration
from fastchat.model.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations
from fastchat.constants import CPU_ISA
from fastchat.utils import get_gpu_memory
import torch
import warnings
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import math
import psutil
from bigdl.llm.utils.common import invalidInputError
is_fastchat_patched = False
_mapping_fastchat = None

def _get_patch_map():
    if False:
        for i in range(10):
            print('nop')
    global _mapping_fastchat
    if _mapping_fastchat is None:
        _mapping_fastchat = []
    from fastchat.model import model_adapter
    _mapping_fastchat += [[BaseModelAdapter, 'load_model', load_model_base, None], [ChatGLMAdapter, 'load_model', load_model_chatglm, None], [model_adapter, 'load_model', load_model, None]]
    return _mapping_fastchat

def load_model_base(self, model_path: str, from_pretrained_kwargs: dict):
    if False:
        return 10
    revision = from_pretrained_kwargs.get('revision', 'main')
    print('Customized bigdl-llm loader')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=self.use_fast_tokenizer, revision=revision)
    from bigdl.llm.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs)
    return (model, tokenizer)

def load_model_chatglm(self, model_path: str, from_pretrained_kwargs: dict):
    if False:
        i = 10
        return i + 15
    revision = from_pretrained_kwargs.get('revision', 'main')
    print('Customized bigdl-llm loader')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision=revision)
    from bigdl.llm.transformers import AutoModel
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True, **from_pretrained_kwargs)
    return (model, tokenizer)

def load_model(model_path: str, device: str='cuda', num_gpus: int=1, max_gpu_memory: Optional[str]=None, load_8bit: bool=False, cpu_offloading: bool=False, gptq_config: Optional[GptqConfig]=None, awq_config: Optional[AWQConfig]=None, revision: str='main', debug: bool=False):
    if False:
        return 10
    'Load a model from Hugging Face.'
    adapter = get_model_adapter(model_path)
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(device, load_8bit, cpu_offloading)
    if device == 'cpu':
        kwargs = {'torch_dtype': 'auto'}
        if CPU_ISA in ['avx512_bf16', 'amx']:
            try:
                import intel_extension_for_pytorch as ipex
                kwargs = {'torch_dtype': torch.bfloat16}
            except ImportError:
                warnings.warn('Intel Extension for PyTorch is not installed, it can be installed to accelerate cpu inference')
    elif device == 'cuda':
        kwargs = {'torch_dtype': torch.float16}
        if num_gpus != 1:
            kwargs['device_map'] = 'auto'
            if max_gpu_memory is None:
                kwargs['device_map'] = 'sequential'
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs['max_memory'] = {i: str(int(available_gpu_memory[i] * 0.85)) + 'GiB' for i in range(num_gpus)}
            else:
                kwargs['max_memory'] = {i: max_gpu_memory for i in range(num_gpus)}
    elif device == 'mps':
        kwargs = {'torch_dtype': torch.float16}
        replace_llama_attn_with_non_inplace_operations()
    elif device == 'xpu':
        kwargs = {}
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            warnings.warn('Intel Extension for PyTorch is not installed, but is required for xpu inference.')
    else:
        invalidInputError(False, f'Invalid device: {device}')
    if cpu_offloading:
        from transformers import BitsAndBytesConfig
        if 'max_memory' in kwargs:
            kwargs['max_memory']['cpu'] = str(math.floor(psutil.virtual_memory().available / 2 ** 20)) + 'Mib'
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=cpu_offloading)
        kwargs['load_in_8bit'] = load_8bit
    elif load_8bit:
        if num_gpus != 1:
            warnings.warn('8-bit quantization is not supported for multi-gpu inference.')
        else:
            (model, tokenizer) = adapter.load_compress_model(model_path=model_path, device=device, torch_dtype=kwargs['torch_dtype'], revision=revision)
            if debug:
                print(model)
            return (model, tokenizer)
    elif awq_config and awq_config.wbits < 16:
        invalidInputError(awq_config.wbits != 4, 'Currently we only support 4-bit inference for AWQ.')
        (model, tokenizer) = load_awq_quantized(model_path, awq_config, device)
        if num_gpus != 1:
            device_map = accelerate.infer_auto_device_map(model, max_memory=kwargs['max_memory'], no_split_module_classes=['OPTDecoderLayer', 'LlamaDecoderLayer', 'BloomBlock', 'MPTBlock', 'DecoderLayer'])
            model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True)
        else:
            model.to(device)
        return (model, tokenizer)
    elif gptq_config and gptq_config.wbits < 16:
        (model, tokenizer) = load_gptq_quantized(model_path, gptq_config)
        if num_gpus != 1:
            device_map = accelerate.infer_auto_device_map(model, max_memory=kwargs['max_memory'], no_split_module_classes=['LlamaDecoderLayer'])
            model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True)
        else:
            model.to(device)
        return (model, tokenizer)
    kwargs['revision'] = revision
    (model, tokenizer) = adapter.load_model(model_path, kwargs)
    if device == 'cpu' and kwargs['torch_dtype'] is torch.bfloat16 and (CPU_ISA is not None):
        model = ipex.optimize(model, dtype=kwargs['torch_dtype'])
    if device == 'cuda' and num_gpus == 1 and (not cpu_offloading) or device in ('mps', 'xpu'):
        model.to(device)
    if debug:
        print(model)
    return (model, tokenizer)

class BigDLLLMAdapter(BaseModelAdapter):
    """Model adapter for bigdl-llm backend models"""

    def match(self, model_path: str):
        if False:
            i = 10
            return i + 15
        return 'bigdl' in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        if False:
            print('Hello World!')
        revision = from_pretrained_kwargs.get('revision', 'main')
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, revision=revision)
        print('Customized bigdl-llm loader')
        from bigdl.llm.transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, low_cpu_mem_usage=True, **from_pretrained_kwargs)
        return (model, tokenizer)

class BigDLLMLOWBITAdapter(BaseModelAdapter):
    """Model adapter for bigdl-llm backend low-bit models"""

    def match(self, model_path: str):
        if False:
            i = 10
            return i + 15
        return 'bigdl-lowbit' in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        if False:
            i = 10
            return i + 15
        revision = from_pretrained_kwargs.get('revision', 'main')
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, revision=revision)
        print('Customized bigdl-llm loader')
        from bigdl.llm.transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.load_low_bit(model_path)
        return (model, tokenizer)

def patch_fastchat():
    if False:
        for i in range(10):
            print('nop')
    global is_fastchat_patched
    if is_fastchat_patched:
        return
    register_model_adapter(BigDLLMLOWBITAdapter)
    register_model_adapter(BigDLLLMAdapter)
    mapping_fastchat = _get_patch_map()
    for mapping_iter in mapping_fastchat:
        if mapping_iter[3] is None:
            mapping_iter[3] = getattr(mapping_iter[0], mapping_iter[1], None)
        setattr(mapping_iter[0], mapping_iter[1], mapping_iter[2])
    is_fastchat_patched = True
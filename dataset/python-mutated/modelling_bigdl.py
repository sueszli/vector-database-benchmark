import importlib
import logging
from bigdl.llm.utils.common import invalidInputError
from .model import *

class BigdlNativeForCausalLM:
    """
    A generic model class that mimics the behavior of
    ``transformers.LlamaForCausalLM.from_pretrained`` API
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, model_family: str='llama', dtype: str='int4', **kwargs):
        if False:
            return 10
        '\n        :param pretrained_model_name_or_path: Path for converted BigDL-LLM optimized ggml\n               binary checkpoint. The checkpoint should be converted by ``bigdl.llm.llm_convert``.\n        :param model_family: The model family of the pretrained checkpoint.\n               Currently we support ``"llama"``, ``"bloom"``, ``"gptneox"``, ``"starcoder"``\n               and ``"chatglm"``.\n        :param dtype: Which quantized precision will be converted.\n                Now only `int4` and `int8` are supported, and `int8` only works for `llama`\n                , `gptneox` and `starcoder`.\n        :param cache_dir: (optional) This parameter will only be used when\n               ``pretrained_model_name_or_path`` is a huggingface checkpoint or hub repo id.\n               It indicates the saving path for the converted low precision model.\n        :param tmp_path: (optional) Which path to store the intermediate fp16 model during the\n               conversion process. Default to `None` so that intermediate model will not be saved.\n        :param kwargs: keyword arguments which will be passed to the model instance\n\n        :return: a model instance\n        '
        logging.warning('BigdlNativeForCausalLM has been deprecated, please switch to the new CausalLM API for sepcific models.')
        invalidInputError(model_family in ['llama', 'gptneox', 'bloom', 'starcoder', 'chatglm'], "Now we only support model family: 'llama', 'gptneox', 'bloom', 'starcoder', 'chatglm', '{}' is not in the list.".format(model_family))
        invalidInputError(dtype.lower() in ['int4', 'int8'], 'Now we only support int4 and int8 as date type for weight')
        ggml_model_path = pretrained_model_name_or_path
        if model_family == 'llama':
            from bigdl.llm.ggml.model.llama import Llama
            return Llama(model_path=ggml_model_path, **kwargs)
        elif model_family == 'gptneox':
            from bigdl.llm.ggml.model.gptneox import Gptneox
            return Gptneox(model_path=ggml_model_path, **kwargs)
        elif model_family == 'bloom':
            from bigdl.llm.ggml.model.bloom import Bloom
            return Bloom(model_path=ggml_model_path, **kwargs)
        elif model_family == 'starcoder':
            from bigdl.llm.ggml.model.starcoder import Starcoder
            return Starcoder(model_path=ggml_model_path, **kwargs)
        elif model_family == 'chatglm':
            from bigdl.llm.ggml.model.chatglm import ChatGLM
            return ChatGLM(model_path=ggml_model_path, **kwargs)

class _BaseGGMLClass:
    GGML_Model = None
    HF_Class = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, native: bool=True, dtype: str='int4', *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param pretrained_model_name_or_path: Path for model checkpoint.\n               If running with ``native int4``, the path should be converted BigDL-LLM optimized\n               ggml binary checkpoint, which should be converted by ``bigdl.llm.llm_convert``.\n               If running with ``transformers int4``, the path should be the huggingface repo id\n               to be downloaded or the huggingface checkpoint folder.\n        :param native: Load model to either BigDL-LLM optimized Transformer or Native (ggml) int4.\n        :param dtype: Which quantized precision will be converted.\n               Now only `int4` and `int8` are supported, and `int8` only works for `llama`\n               , `gptneox` and `starcoder`.\n        :param kwargs: keyword arguments which will be passed to the model instance.\n\n        :return: a model instance\n        '
        try:
            module = importlib.import_module(cls.GGML_Module)
            class_ = getattr(module, cls.GGML_Model)
            if native:
                invalidInputError(dtype.lower() in ['int4', 'int8'], 'Now we only support int4 and int8 as date type for weight')
                ggml_model_path = pretrained_model_name_or_path
                model = class_(model_path=ggml_model_path, **kwargs)
            else:
                model = cls.HF_Class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except Exception as e:
            invalidInputError(False, f'Could not load model from path: {pretrained_model_name_or_path}. Please make sure the CausalLM class matches the model you want to load.Received error {e}')
        return model

class LlamaForCausalLM(_BaseGGMLClass):
    GGML_Module = 'bigdl.llm.models'
    GGML_Model = 'Llama'
    HF_Class = AutoModelForCausalLM

class ChatGLMForCausalLM(_BaseGGMLClass):
    GGML_Module = 'bigdl.llm.ggml.model.chatglm'
    GGML_Model = 'ChatGLM'
    HF_Class = AutoModel

class GptneoxForCausalLM(_BaseGGMLClass):
    GGML_Module = 'bigdl.llm.models'
    GGML_Model = 'Gptneox'
    HF_Class = AutoModelForCausalLM

class BloomForCausalLM(_BaseGGMLClass):
    GGML_Module = 'bigdl.llm.models'
    GGML_Model = 'Bloom'
    HF_Class = AutoModelForCausalLM

class StarcoderForCausalLM(_BaseGGMLClass):
    GGML_Module = 'bigdl.llm.models'
    GGML_Model = 'Starcoder'
    HF_Class = AutoModelForCausalLM
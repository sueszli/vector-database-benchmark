import os
from transformers import AutoConfig as AutoConfigHF
from transformers import AutoModel as AutoModelHF
from transformers import AutoModelForCausalLM as AutoModelForCausalLMHF
from transformers import AutoModelForSeq2SeqLM as AutoModelForSeq2SeqLMHF
from transformers import AutoModelForSequenceClassification as AutoModelForSequenceClassificationHF
from transformers import AutoModelForTokenClassification as AutoModelForTokenClassificationHF
from transformers import AutoTokenizer as AutoTokenizerHF
from transformers import BitsAndBytesConfig as BitsAndBytesConfigHF
from transformers import GenerationConfig as GenerationConfigHF
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from modelscope import snapshot_download
from modelscope.utils.automodel_utils import fix_upgrade
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, Invoke
try:
    from transformers import GPTQConfig as GPTQConfigHF
except ImportError:
    GPTQConfigHF = None

def user_agent(invoked_by=None):
    if False:
        while True:
            i = 10
    if invoked_by is None:
        invoked_by = Invoke.PRETRAINED
    uagent = '%s/%s' % (Invoke.KEY, invoked_by)
    return uagent

def patch_tokenizer_base():
    if False:
        print('Hello World!')
    ' Monkey patch PreTrainedTokenizerBase.from_pretrained to adapt to modelscope hub.\n    '
    ori_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ignore_file_pattern = ['\\w+\\.bin', '\\w+\\.safetensors']
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(pretrained_model_name_or_path, revision=revision, ignore_file_pattern=ignore_file_pattern)
        else:
            model_dir = pretrained_model_name_or_path
        return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)
    PreTrainedTokenizerBase.from_pretrained = from_pretrained

def patch_model_base():
    if False:
        return 10
    ' Monkey patch PreTrainedModel.from_pretrained to adapt to modelscope hub.\n    '
    ori_from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if False:
            print('Hello World!')
        ignore_file_pattern = ['\\w+\\.safetensors']
        if not os.path.exists(pretrained_model_name_or_path):
            revision = kwargs.pop('revision', None)
            model_dir = snapshot_download(pretrained_model_name_or_path, revision=revision, ignore_file_pattern=ignore_file_pattern)
        else:
            model_dir = pretrained_model_name_or_path
        return ori_from_pretrained(cls, model_dir, *model_args, **kwargs)
    PreTrainedModel.from_pretrained = from_pretrained
patch_tokenizer_base()
patch_model_base()

def get_wrapped_class(module_class, ignore_file_pattern=[], **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Get a custom wrapper class for  auto classes to download the models from the ModelScope hub\n    Args:\n        module_class: The actual module class\n        ignore_file_pattern (`str` or `List`, *optional*, default to `None`):\n            Any file pattern to be ignored in downloading, like exact file names or file extensions.\n    Returns:\n        The wrapper\n    '
    default_ignore_file_pattern = ignore_file_pattern

    class ClassWrapper(module_class):

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            if False:
                while True:
                    i = 10
            ignore_file_pattern = kwargs.pop('ignore_file_pattern', default_ignore_file_pattern)
            if not os.path.exists(pretrained_model_name_or_path):
                revision = kwargs.pop('revision', DEFAULT_MODEL_REVISION)
                model_dir = snapshot_download(pretrained_model_name_or_path, revision=revision, ignore_file_pattern=ignore_file_pattern, user_agent=user_agent())
            else:
                model_dir = pretrained_model_name_or_path
            module_obj = module_class.from_pretrained(model_dir, *model_args, **kwargs)
            if module_class.__name__.startswith('AutoModel'):
                module_obj.model_dir = model_dir
            fix_upgrade(module_obj)
            return module_obj
    ClassWrapper.__name__ = module_class.__name__
    ClassWrapper.__qualname__ = module_class.__qualname__
    return ClassWrapper
AutoModel = get_wrapped_class(AutoModelHF)
AutoModelForCausalLM = get_wrapped_class(AutoModelForCausalLMHF)
AutoModelForSeq2SeqLM = get_wrapped_class(AutoModelForSeq2SeqLMHF)
AutoModelForSequenceClassification = get_wrapped_class(AutoModelForSequenceClassificationHF)
AutoModelForTokenClassification = get_wrapped_class(AutoModelForTokenClassificationHF)
AutoTokenizer = get_wrapped_class(AutoTokenizerHF, ignore_file_pattern=['\\w+\\.bin', '\\w+\\.safetensors'])
AutoConfig = get_wrapped_class(AutoConfigHF, ignore_file_pattern=['\\w+\\.bin', '\\w+\\.safetensors'])
GenerationConfig = get_wrapped_class(GenerationConfigHF, ignore_file_pattern=['\\w+\\.bin', '\\w+\\.safetensors'])
GPTQConfig = GPTQConfigHF
BitsAndBytesConfig = BitsAndBytesConfigHF
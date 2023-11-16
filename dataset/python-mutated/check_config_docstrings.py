import inspect
import re
from transformers.utils import direct_transformers_import
PATH_TO_TRANSFORMERS = 'src/transformers'
transformers = direct_transformers_import(PATH_TO_TRANSFORMERS)
CONFIG_MAPPING = transformers.models.auto.configuration_auto.CONFIG_MAPPING
_re_checkpoint = re.compile('\\[(.+?)\\]\\((https://huggingface\\.co/.+?)\\)')
CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK = {'DecisionTransformerConfig', 'EncoderDecoderConfig', 'MusicgenConfig', 'RagConfig', 'SpeechEncoderDecoderConfig', 'TimmBackboneConfig', 'VisionEncoderDecoderConfig', 'VisionTextDualEncoderConfig', 'LlamaConfig'}

def get_checkpoint_from_config_class(config_class):
    if False:
        return 10
    checkpoint = None
    config_source = inspect.getsource(config_class)
    checkpoints = _re_checkpoint.findall(config_source)
    for (ckpt_name, ckpt_link) in checkpoints:
        if ckpt_link.endswith('/'):
            ckpt_link = ckpt_link[:-1]
        ckpt_link_from_name = f'https://huggingface.co/{ckpt_name}'
        if ckpt_link == ckpt_link_from_name:
            checkpoint = ckpt_name
            break
    return checkpoint

def check_config_docstrings_have_checkpoints():
    if False:
        while True:
            i = 10
    configs_without_checkpoint = []
    for config_class in list(CONFIG_MAPPING.values()):
        if 'models.deprecated' in config_class.__module__:
            continue
        checkpoint = get_checkpoint_from_config_class(config_class)
        name = config_class.__name__
        if checkpoint is None and name not in CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK:
            configs_without_checkpoint.append(name)
    if len(configs_without_checkpoint) > 0:
        message = '\n'.join(sorted(configs_without_checkpoint))
        raise ValueError(f"The following configurations don't contain any valid checkpoint:\n{message}\n\nThe requirement is to include a link pointing to one of the models of this architecture in the docstring of the config classes listed above. The link should have be a markdown format like [myorg/mymodel](https://huggingface.co/myorg/mymodel).")
if __name__ == '__main__':
    check_config_docstrings_have_checkpoints()
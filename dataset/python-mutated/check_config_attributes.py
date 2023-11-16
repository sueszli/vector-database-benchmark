import inspect
import os
import re
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import direct_transformers_import
PATH_TO_TRANSFORMERS = 'src/transformers'
transformers = direct_transformers_import(PATH_TO_TRANSFORMERS)
CONFIG_MAPPING = transformers.models.auto.configuration_auto.CONFIG_MAPPING
SPECIAL_CASES_TO_ALLOW = {'EncodecConfig': ['overlap'], 'DPRConfig': True, 'FuyuConfig': True, 'FSMTConfig': ['langs'], 'GPTNeoConfig': ['attention_types'], 'EsmConfig': ['is_folding_model'], 'Mask2FormerConfig': ['ignore_value'], 'OneFormerConfig': ['ignore_value', 'norm'], 'GraphormerConfig': ['spatial_pos_max'], 'T5Config': ['feed_forward_proj'], 'MT5Config': ['feed_forward_proj', 'tokenizer_class'], 'UMT5Config': ['feed_forward_proj', 'tokenizer_class'], 'LongT5Config': ['feed_forward_proj'], 'Pop2PianoConfig': ['feed_forward_proj'], 'SwitchTransformersConfig': ['feed_forward_proj'], 'BioGptConfig': ['layer_norm_eps'], 'GLPNConfig': ['layer_norm_eps'], 'SegformerConfig': ['layer_norm_eps'], 'CvtConfig': ['layer_norm_eps'], 'PerceiverConfig': ['layer_norm_eps'], 'InformerConfig': ['num_static_real_features', 'num_time_features'], 'TimeSeriesTransformerConfig': ['num_static_real_features', 'num_time_features'], 'AutoformerConfig': ['num_static_real_features', 'num_time_features'], 'SamVisionConfig': ['mlp_ratio'], 'ClapAudioConfig': ['num_classes'], 'SpeechT5HifiGanConfig': ['sampling_rate'], 'SeamlessM4TConfig': ['max_new_tokens', 't2u_max_new_tokens', 't2u_decoder_attention_heads', 't2u_decoder_ffn_dim', 't2u_decoder_layers', 't2u_encoder_attention_heads', 't2u_encoder_ffn_dim', 't2u_encoder_layers', 't2u_max_position_embeddings']}
SPECIAL_CASES_TO_ALLOW.update({'CLIPSegConfig': True, 'DeformableDetrConfig': True, 'DetaConfig': True, 'DinatConfig': True, 'DonutSwinConfig': True, 'EfficientFormerConfig': True, 'FSMTConfig': True, 'JukeboxConfig': True, 'LayoutLMv2Config': True, 'MaskFormerSwinConfig': True, 'MT5Config': True, 'MptConfig': True, 'MptAttentionConfig': True, 'NatConfig': True, 'OneFormerConfig': True, 'PerceiverConfig': True, 'RagConfig': True, 'SpeechT5Config': True, 'SwinConfig': True, 'Swin2SRConfig': True, 'Swinv2Config': True, 'SwitchTransformersConfig': True, 'TableTransformerConfig': True, 'TapasConfig': True, 'TransfoXLConfig': True, 'UniSpeechConfig': True, 'UniSpeechSatConfig': True, 'WavLMConfig': True, 'WhisperConfig': True, 'JukeboxPriorConfig': True, 'Pix2StructTextConfig': True, 'IdeficsConfig': True, 'IdeficsVisionConfig': True, 'IdeficsPerceiverConfig': True})

def check_attribute_being_used(config_class, attributes, default_value, source_strings):
    if False:
        i = 10
        return i + 15
    'Check if any name in `attributes` is used in one of the strings in `source_strings`\n\n    Args:\n        config_class (`type`):\n            The configuration class for which the arguments in its `__init__` will be checked.\n        attributes (`List[str]`):\n            The name of an argument (or attribute) and its variant names if any.\n        default_value (`Any`):\n            A default value for the attribute in `attributes` assigned in the `__init__` of `config_class`.\n        source_strings (`List[str]`):\n            The python source code strings in the same modeling directory where `config_class` is defined. The file\n            containing the definition of `config_class` should be excluded.\n    '
    attribute_used = False
    for attribute in attributes:
        for modeling_source in source_strings:
            if f'config.{attribute}' in modeling_source or f'getattr(config, "{attribute}"' in modeling_source or f'getattr(self.config, "{attribute}"' in modeling_source:
                attribute_used = True
            elif re.search(f'getattr[ \\t\\v\\n\\r\\f]*\\([ \\t\\v\\n\\r\\f]*(self\\.)?config,[ \\t\\v\\n\\r\\f]*"{attribute}"', modeling_source) is not None:
                attribute_used = True
            elif attribute in ['summary_type', 'summary_use_proj', 'summary_activation', 'summary_last_dropout', 'summary_proj_to_labels', 'summary_first_dropout']:
                if 'SequenceSummary' in modeling_source:
                    attribute_used = True
            if attribute_used:
                break
        if attribute_used:
            break
    attributes_to_allow = ['bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index', 'image_size', 'use_cache', 'out_features', 'out_indices', 'sampling_rate']
    attributes_used_in_generation = ['encoder_no_repeat_ngram_size']
    case_allowed = True
    if not attribute_used:
        case_allowed = False
        for attribute in attributes:
            if attribute in ['is_encoder_decoder'] and default_value is True:
                case_allowed = True
            elif attribute in ['tie_word_embeddings'] and default_value is False:
                case_allowed = True
            elif attribute in attributes_to_allow + attributes_used_in_generation:
                case_allowed = True
            elif attribute.endswith('_token_id'):
                case_allowed = True
            if not case_allowed:
                allowed_cases = SPECIAL_CASES_TO_ALLOW.get(config_class.__name__, [])
                case_allowed = allowed_cases is True or attribute in allowed_cases
    return attribute_used or case_allowed

def check_config_attributes_being_used(config_class):
    if False:
        print('Hello World!')
    'Check the arguments in `__init__` of `config_class` are used in the modeling files in the same directory\n\n    Args:\n        config_class (`type`):\n            The configuration class for which the arguments in its `__init__` will be checked.\n    '
    signature = dict(inspect.signature(config_class.__init__).parameters)
    parameter_names = [x for x in list(signature.keys()) if x not in ['self', 'kwargs']]
    parameter_defaults = [signature[param].default for param in parameter_names]
    reversed_attribute_map = {}
    if len(config_class.attribute_map) > 0:
        reversed_attribute_map = {v: k for (k, v) in config_class.attribute_map.items()}
    config_source_file = inspect.getsourcefile(config_class)
    model_dir = os.path.dirname(config_source_file)
    modeling_paths = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.startswith('modeling_')]
    modeling_sources = []
    for path in modeling_paths:
        if os.path.isfile(path):
            with open(path, encoding='utf8') as fp:
                modeling_sources.append(fp.read())
    unused_attributes = []
    for (config_param, default_value) in zip(parameter_names, parameter_defaults):
        attributes = [config_param]
        if config_param in reversed_attribute_map:
            attributes.append(reversed_attribute_map[config_param])
        if not check_attribute_being_used(config_class, attributes, default_value, modeling_sources):
            unused_attributes.append(attributes[0])
    return sorted(unused_attributes)

def check_config_attributes():
    if False:
        for i in range(10):
            print('nop')
    'Check the arguments in `__init__` of all configuration classes are used in  python files'
    configs_with_unused_attributes = {}
    for _config_class in list(CONFIG_MAPPING.values()):
        if 'models.deprecated' in _config_class.__module__:
            continue
        config_classes_in_module = [cls for (name, cls) in inspect.getmembers(inspect.getmodule(_config_class), lambda x: inspect.isclass(x) and issubclass(x, PretrainedConfig) and (inspect.getmodule(x) == inspect.getmodule(_config_class)))]
        for config_class in config_classes_in_module:
            unused_attributes = check_config_attributes_being_used(config_class)
            if len(unused_attributes) > 0:
                configs_with_unused_attributes[config_class.__name__] = unused_attributes
    if len(configs_with_unused_attributes) > 0:
        error = 'The following configuration classes contain unused attributes in the corresponding modeling files:\n'
        for (name, attributes) in configs_with_unused_attributes.items():
            error += f'{name}: {attributes}\n'
        raise ValueError(error)
if __name__ == '__main__':
    check_config_attributes()
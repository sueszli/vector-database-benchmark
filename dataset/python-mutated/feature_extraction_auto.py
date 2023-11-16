""" AutoFeatureExtractor class."""
import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...feature_extraction_utils import FeatureExtractionMixin
from ...utils import CONFIG_NAME, FEATURE_EXTRACTOR_NAME, get_file_from_repo, logging
from .auto_factory import _LazyAutoMapping
from .configuration_auto import CONFIG_MAPPING_NAMES, AutoConfig, model_type_to_module_name, replace_list_option_in_docstrings
logger = logging.get_logger(__name__)
FEATURE_EXTRACTOR_MAPPING_NAMES = OrderedDict([('audio-spectrogram-transformer', 'ASTFeatureExtractor'), ('beit', 'BeitFeatureExtractor'), ('chinese_clip', 'ChineseCLIPFeatureExtractor'), ('clap', 'ClapFeatureExtractor'), ('clip', 'CLIPFeatureExtractor'), ('clipseg', 'ViTFeatureExtractor'), ('clvp', 'ClvpFeatureExtractor'), ('conditional_detr', 'ConditionalDetrFeatureExtractor'), ('convnext', 'ConvNextFeatureExtractor'), ('cvt', 'ConvNextFeatureExtractor'), ('data2vec-audio', 'Wav2Vec2FeatureExtractor'), ('data2vec-vision', 'BeitFeatureExtractor'), ('deformable_detr', 'DeformableDetrFeatureExtractor'), ('deit', 'DeiTFeatureExtractor'), ('detr', 'DetrFeatureExtractor'), ('dinat', 'ViTFeatureExtractor'), ('donut-swin', 'DonutFeatureExtractor'), ('dpt', 'DPTFeatureExtractor'), ('encodec', 'EncodecFeatureExtractor'), ('flava', 'FlavaFeatureExtractor'), ('glpn', 'GLPNFeatureExtractor'), ('groupvit', 'CLIPFeatureExtractor'), ('hubert', 'Wav2Vec2FeatureExtractor'), ('imagegpt', 'ImageGPTFeatureExtractor'), ('layoutlmv2', 'LayoutLMv2FeatureExtractor'), ('layoutlmv3', 'LayoutLMv3FeatureExtractor'), ('levit', 'LevitFeatureExtractor'), ('maskformer', 'MaskFormerFeatureExtractor'), ('mctct', 'MCTCTFeatureExtractor'), ('mobilenet_v1', 'MobileNetV1FeatureExtractor'), ('mobilenet_v2', 'MobileNetV2FeatureExtractor'), ('mobilevit', 'MobileViTFeatureExtractor'), ('nat', 'ViTFeatureExtractor'), ('owlvit', 'OwlViTFeatureExtractor'), ('perceiver', 'PerceiverFeatureExtractor'), ('poolformer', 'PoolFormerFeatureExtractor'), ('pop2piano', 'Pop2PianoFeatureExtractor'), ('regnet', 'ConvNextFeatureExtractor'), ('resnet', 'ConvNextFeatureExtractor'), ('seamless_m4t', 'SeamlessM4TFeatureExtractor'), ('segformer', 'SegformerFeatureExtractor'), ('sew', 'Wav2Vec2FeatureExtractor'), ('sew-d', 'Wav2Vec2FeatureExtractor'), ('speech_to_text', 'Speech2TextFeatureExtractor'), ('speecht5', 'SpeechT5FeatureExtractor'), ('swiftformer', 'ViTFeatureExtractor'), ('swin', 'ViTFeatureExtractor'), ('swinv2', 'ViTFeatureExtractor'), ('table-transformer', 'DetrFeatureExtractor'), ('timesformer', 'VideoMAEFeatureExtractor'), ('tvlt', 'TvltFeatureExtractor'), ('unispeech', 'Wav2Vec2FeatureExtractor'), ('unispeech-sat', 'Wav2Vec2FeatureExtractor'), ('van', 'ConvNextFeatureExtractor'), ('videomae', 'VideoMAEFeatureExtractor'), ('vilt', 'ViltFeatureExtractor'), ('vit', 'ViTFeatureExtractor'), ('vit_mae', 'ViTFeatureExtractor'), ('vit_msn', 'ViTFeatureExtractor'), ('wav2vec2', 'Wav2Vec2FeatureExtractor'), ('wav2vec2-conformer', 'Wav2Vec2FeatureExtractor'), ('wavlm', 'Wav2Vec2FeatureExtractor'), ('whisper', 'WhisperFeatureExtractor'), ('xclip', 'CLIPFeatureExtractor'), ('yolos', 'YolosFeatureExtractor')])
FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)

def feature_extractor_class_from_name(class_name: str):
    if False:
        i = 10
        return i + 15
    for (module_name, extractors) in FEATURE_EXTRACTOR_MAPPING_NAMES.items():
        if class_name in extractors:
            module_name = model_type_to_module_name(module_name)
            module = importlib.import_module(f'.{module_name}', 'transformers.models')
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue
    for (_, extractor) in FEATURE_EXTRACTOR_MAPPING._extra_content.items():
        if getattr(extractor, '__name__', None) == class_name:
            return extractor
    main_module = importlib.import_module('transformers')
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)
    return None

def get_feature_extractor_config(pretrained_model_name_or_path: Union[str, os.PathLike], cache_dir: Optional[Union[str, os.PathLike]]=None, force_download: bool=False, resume_download: bool=False, proxies: Optional[Dict[str, str]]=None, token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, local_files_only: bool=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Loads the tokenizer configuration from a pretrained model tokenizer configuration.\n\n    Args:\n        pretrained_model_name_or_path (`str` or `os.PathLike`):\n            This can be either:\n\n            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on\n              huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced\n              under a user or organization name, like `dbmdz/bert-base-german-cased`.\n            - a path to a *directory* containing a configuration file saved using the\n              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.\n\n        cache_dir (`str` or `os.PathLike`, *optional*):\n            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard\n            cache should not be used.\n        force_download (`bool`, *optional*, defaults to `False`):\n            Whether or not to force to (re-)download the configuration files and override the cached versions if they\n            exist.\n        resume_download (`bool`, *optional*, defaults to `False`):\n            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.\n        proxies (`Dict[str, str]`, *optional*):\n            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n            \'http://hostname\': \'foo.bar:4012\'}.` The proxies are used on each request.\n        token (`str` or *bool*, *optional*):\n            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated\n            when running `huggingface-cli login` (stored in `~/.huggingface`).\n        revision (`str`, *optional*, defaults to `"main"`):\n            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n            identifier allowed by git.\n        local_files_only (`bool`, *optional*, defaults to `False`):\n            If `True`, will only try to load the tokenizer configuration from local files.\n\n    <Tip>\n\n    Passing `token=True` is required when you want to use a private model.\n\n    </Tip>\n\n    Returns:\n        `Dict`: The configuration of the tokenizer.\n\n    Examples:\n\n    ```python\n    # Download configuration from huggingface.co and cache.\n    tokenizer_config = get_tokenizer_config("bert-base-uncased")\n    # This model does not have a tokenizer config so the result will be an empty dict.\n    tokenizer_config = get_tokenizer_config("xlm-roberta-base")\n\n    # Save a pretrained tokenizer locally and you can reload its config\n    from transformers import AutoTokenizer\n\n    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")\n    tokenizer.save_pretrained("tokenizer-test")\n    tokenizer_config = get_tokenizer_config("tokenizer-test")\n    ```'
    use_auth_token = kwargs.pop('use_auth_token', None)
    if use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
        if token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        token = use_auth_token
    resolved_config_file = get_file_from_repo(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, token=token, revision=revision, local_files_only=local_files_only)
    if resolved_config_file is None:
        logger.info('Could not locate the feature extractor configuration file, will try to use the model config instead.')
        return {}
    with open(resolved_config_file, encoding='utf-8') as reader:
        return json.load(reader)

class AutoFeatureExtractor:
    """
    This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the
    library when created with the [`AutoFeatureExtractor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        raise EnvironmentError('AutoFeatureExtractor is designed to be instantiated using the `AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)` method.')

    @classmethod
    @replace_list_option_in_docstrings(FEATURE_EXTRACTOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Instantiate one of the feature extractor classes of the library from a pretrained model vocabulary.\n\n        The feature extractor class to instantiate is selected based on the `model_type` property of the config object\n        (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it\'s\n        missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:\n\n        List options\n\n        Params:\n            pretrained_model_name_or_path (`str` or `os.PathLike`):\n                This can be either:\n\n                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on\n                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or\n                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.\n                - a path to a *directory* containing a feature extractor file saved using the\n                  [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] method, e.g.,\n                  `./my_model_directory/`.\n                - a path or url to a saved feature extractor JSON *file*, e.g.,\n                  `./my_model_directory/preprocessor_config.json`.\n            cache_dir (`str` or `os.PathLike`, *optional*):\n                Path to a directory in which a downloaded pretrained model feature extractor should be cached if the\n                standard cache should not be used.\n            force_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to force to (re-)download the feature extractor files and override the cached versions\n                if they exist.\n            resume_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to delete incompletely received file. Attempts to resume the download if such a file\n                exists.\n            proxies (`Dict[str, str]`, *optional*):\n                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n                \'http://hostname\': \'foo.bar:4012\'}.` The proxies are used on each request.\n            token (`str` or *bool*, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated\n                when running `huggingface-cli login` (stored in `~/.huggingface`).\n            revision (`str`, *optional*, defaults to `"main"`):\n                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n                identifier allowed by git.\n            return_unused_kwargs (`bool`, *optional*, defaults to `False`):\n                If `False`, then this function returns just the final feature extractor object. If `True`, then this\n                functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused_kwargs* is a dictionary\n                consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of\n                `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.\n            trust_remote_code (`bool`, *optional*, defaults to `False`):\n                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option\n                should only be set to `True` for repositories you trust and in which you have read the code, as it will\n                execute code present on the Hub on your local machine.\n            kwargs (`Dict[str, Any]`, *optional*):\n                The values in kwargs of any keys which are feature extractor attributes will be used to override the\n                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is\n                controlled by the `return_unused_kwargs` keyword parameter.\n\n        <Tip>\n\n        Passing `token=True` is required when you want to use a private model.\n\n        </Tip>\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoFeatureExtractor\n\n        >>> # Download feature extractor from huggingface.co and cache.\n        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")\n\n        >>> # If feature extractor files are in a directory (e.g. feature extractor was saved using *save_pretrained(\'./test/saved_model/\')*)\n        >>> # feature_extractor = AutoFeatureExtractor.from_pretrained("./test/saved_model/")\n        ```'
        use_auth_token = kwargs.pop('use_auth_token', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if kwargs.get('token', None) is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            kwargs['token'] = use_auth_token
        config = kwargs.pop('config', None)
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        kwargs['_from_auto'] = True
        (config_dict, _) = FeatureExtractionMixin.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)
        feature_extractor_class = config_dict.get('feature_extractor_type', None)
        feature_extractor_auto_map = None
        if 'AutoFeatureExtractor' in config_dict.get('auto_map', {}):
            feature_extractor_auto_map = config_dict['auto_map']['AutoFeatureExtractor']
        if feature_extractor_class is None and feature_extractor_auto_map is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            feature_extractor_class = getattr(config, 'feature_extractor_type', None)
            if hasattr(config, 'auto_map') and 'AutoFeatureExtractor' in config.auto_map:
                feature_extractor_auto_map = config.auto_map['AutoFeatureExtractor']
        if feature_extractor_class is not None:
            feature_extractor_class = feature_extractor_class_from_name(feature_extractor_class)
        has_remote_code = feature_extractor_auto_map is not None
        has_local_code = feature_extractor_class is not None or type(config) in FEATURE_EXTRACTOR_MAPPING
        trust_remote_code = resolve_trust_remote_code(trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code)
        if has_remote_code and trust_remote_code:
            feature_extractor_class = get_class_from_dynamic_module(feature_extractor_auto_map, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop('code_revision', None)
            if os.path.isdir(pretrained_model_name_or_path):
                feature_extractor_class.register_for_auto_class()
            return feature_extractor_class.from_dict(config_dict, **kwargs)
        elif feature_extractor_class is not None:
            return feature_extractor_class.from_dict(config_dict, **kwargs)
        elif type(config) in FEATURE_EXTRACTOR_MAPPING:
            feature_extractor_class = FEATURE_EXTRACTOR_MAPPING[type(config)]
            return feature_extractor_class.from_dict(config_dict, **kwargs)
        raise ValueError(f"Unrecognized feature extractor in {pretrained_model_name_or_path}. Should have a `feature_extractor_type` key in its {FEATURE_EXTRACTOR_NAME} of {CONFIG_NAME}, or one of the following `model_type` keys in its {CONFIG_NAME}: {', '.join((c for c in FEATURE_EXTRACTOR_MAPPING_NAMES.keys()))}")

    @staticmethod
    def register(config_class, feature_extractor_class, exist_ok=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Register a new feature extractor for this class.\n\n        Args:\n            config_class ([`PretrainedConfig`]):\n                The configuration corresponding to the model to register.\n            feature_extractor_class ([`FeatureExtractorMixin`]): The feature extractor to register.\n        '
        FEATURE_EXTRACTOR_MAPPING.register(config_class, feature_extractor_class, exist_ok=exist_ok)
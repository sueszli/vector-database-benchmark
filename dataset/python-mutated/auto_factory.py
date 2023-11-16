"""Factory function to build auto-model classes."""
import copy
import importlib
import json
import os
import warnings
from collections import OrderedDict
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, cached_file, copy_func, extract_commit_hash, find_adapter_config_file, is_peft_available, logging, requires_backends
from .configuration_auto import AutoConfig, model_type_to_module_name, replace_list_option_in_docstrings
logger = logging.get_logger(__name__)
CLASS_DOCSTRING = '\n    This is a generic model class that will be instantiated as one of the model classes of the library when created\n    with the [`~BaseAutoModelClass.from_pretrained`] class method or the [`~BaseAutoModelClass.from_config`] class\n    method.\n\n    This class cannot be instantiated directly using `__init__()` (throws an error).\n'
FROM_CONFIG_DOCSTRING = '\n        Instantiates one of the model classes of the library from a configuration.\n\n        Note:\n            Loading a model from its configuration file does **not** load the model weights. It only affects the\n            model\'s configuration. Use [`~BaseAutoModelClass.from_pretrained`] to load the model weights.\n\n        Args:\n            config ([`PretrainedConfig`]):\n                The model class to instantiate is selected based on the configuration class:\n\n                List options\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoConfig, BaseAutoModelClass\n\n        >>> # Download configuration from huggingface.co and cache.\n        >>> config = AutoConfig.from_pretrained("checkpoint_placeholder")\n        >>> model = BaseAutoModelClass.from_config(config)\n        ```\n'
FROM_PRETRAINED_TORCH_DOCSTRING = '\n        Instantiate one of the model classes of the library from a pretrained model.\n\n        The model class to instantiate is selected based on the `model_type` property of the config object (either\n        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it\'s missing, by\n        falling back to using pattern matching on `pretrained_model_name_or_path`:\n\n        List options\n\n        The model is set in evaluation mode by default using `model.eval()` (so for instance, dropout modules are\n        deactivated). To train the model, you should first set it back in training mode with `model.train()`\n\n        Args:\n            pretrained_model_name_or_path (`str` or `os.PathLike`):\n                Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In\n                      this case, `from_tf` should be set to `True` and a configuration object should be provided as\n                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a\n                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.\n            model_args (additional positional arguments, *optional*):\n                Will be passed along to the underlying model `__init__()` method.\n            config ([`PretrainedConfig`], *optional*):\n                Configuration for the model to use instead of an automatically loaded configuration. Configuration can\n                be automatically loaded when:\n\n                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained\n                      model).\n                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the\n                      save directory.\n                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a\n                      configuration JSON file named *config.json* is found in the directory.\n            state_dict (*Dict[str, torch.Tensor]*, *optional*):\n                A state dictionary to use instead of a state dictionary loaded from saved weights file.\n\n                This option can be used if you want to create a model from a pretrained configuration but load your own\n                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and\n                [`~PreTrainedModel.from_pretrained`] is not a simpler option.\n            cache_dir (`str` or `os.PathLike`, *optional*):\n                Path to a directory in which a downloaded pretrained model configuration should be cached if the\n                standard cache should not be used.\n            from_tf (`bool`, *optional*, defaults to `False`):\n                Load the model weights from a TensorFlow checkpoint save file (see docstring of\n                `pretrained_model_name_or_path` argument).\n            force_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to force the (re-)download of the model weights and configuration files, overriding the\n                cached versions if they exist.\n            resume_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to delete incompletely received files. Will attempt to resume the download if such a\n                file exists.\n            proxies (`Dict[str, str]`, *optional*):\n                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n                \'http://hostname\': \'foo.bar:4012\'}`. The proxies are used on each request.\n            output_loading_info(`bool`, *optional*, defaults to `False`):\n                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.\n            local_files_only(`bool`, *optional*, defaults to `False`):\n                Whether or not to only look at local files (e.g., not try downloading the model).\n            revision (`str`, *optional*, defaults to `"main"`):\n                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n                identifier allowed by git.\n            trust_remote_code (`bool`, *optional*, defaults to `False`):\n                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option\n                should only be set to `True` for repositories you trust and in which you have read the code, as it will\n                execute code present on the Hub on your local machine.\n            code_revision (`str`, *optional*, defaults to `"main"`):\n                The specific revision to use for the code on the Hub, if the code leaves in a different repository than\n                the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based\n                system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier\n                allowed by git.\n            kwargs (additional keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or\n                automatically loaded:\n\n                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the\n                      underlying model\'s `__init__` method (we assume all relevant updates to the configuration have\n                      already been done)\n                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class\n                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that\n                      corresponds to a configuration attribute will be used to override said attribute with the\n                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute\n                      will be passed to the underlying model\'s `__init__` function.\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoConfig, BaseAutoModelClass\n\n        >>> # Download model and configuration from huggingface.co and cache.\n        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder")\n\n        >>> # Update configuration during loading\n        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder", output_attentions=True)\n        >>> model.config.output_attentions\n        True\n\n        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)\n        >>> config = AutoConfig.from_pretrained("./tf_model/shortcut_placeholder_tf_model_config.json")\n        >>> model = BaseAutoModelClass.from_pretrained(\n        ...     "./tf_model/shortcut_placeholder_tf_checkpoint.ckpt.index", from_tf=True, config=config\n        ... )\n        ```\n'
FROM_PRETRAINED_TF_DOCSTRING = '\n        Instantiate one of the model classes of the library from a pretrained model.\n\n        The model class to instantiate is selected based on the `model_type` property of the config object (either\n        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it\'s missing, by\n        falling back to using pattern matching on `pretrained_model_name_or_path`:\n\n        List options\n\n        Args:\n            pretrained_model_name_or_path (`str` or `os.PathLike`):\n                Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In this\n                      case, `from_pt` should be set to `True` and a configuration object should be provided as `config`\n                      argument. This loading path is slower than converting the PyTorch model in a TensorFlow model\n                      using the provided conversion scripts and loading the TensorFlow model afterwards.\n            model_args (additional positional arguments, *optional*):\n                Will be passed along to the underlying model `__init__()` method.\n            config ([`PretrainedConfig`], *optional*):\n                Configuration for the model to use instead of an automatically loaded configuration. Configuration can\n                be automatically loaded when:\n\n                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained\n                      model).\n                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the\n                      save directory.\n                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a\n                      configuration JSON file named *config.json* is found in the directory.\n            cache_dir (`str` or `os.PathLike`, *optional*):\n                Path to a directory in which a downloaded pretrained model configuration should be cached if the\n                standard cache should not be used.\n            from_pt (`bool`, *optional*, defaults to `False`):\n                Load the model weights from a PyTorch checkpoint save file (see docstring of\n                `pretrained_model_name_or_path` argument).\n            force_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to force the (re-)download of the model weights and configuration files, overriding the\n                cached versions if they exist.\n            resume_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to delete incompletely received files. Will attempt to resume the download if such a\n                file exists.\n            proxies (`Dict[str, str]`, *optional*):\n                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n                \'http://hostname\': \'foo.bar:4012\'}`. The proxies are used on each request.\n            output_loading_info(`bool`, *optional*, defaults to `False`):\n                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.\n            local_files_only(`bool`, *optional*, defaults to `False`):\n                Whether or not to only look at local files (e.g., not try downloading the model).\n            revision (`str`, *optional*, defaults to `"main"`):\n                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n                identifier allowed by git.\n            trust_remote_code (`bool`, *optional*, defaults to `False`):\n                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option\n                should only be set to `True` for repositories you trust and in which you have read the code, as it will\n                execute code present on the Hub on your local machine.\n            code_revision (`str`, *optional*, defaults to `"main"`):\n                The specific revision to use for the code on the Hub, if the code leaves in a different repository than\n                the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based\n                system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier\n                allowed by git.\n            kwargs (additional keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or\n                automatically loaded:\n\n                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the\n                      underlying model\'s `__init__` method (we assume all relevant updates to the configuration have\n                      already been done)\n                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class\n                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that\n                      corresponds to a configuration attribute will be used to override said attribute with the\n                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute\n                      will be passed to the underlying model\'s `__init__` function.\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoConfig, BaseAutoModelClass\n\n        >>> # Download model and configuration from huggingface.co and cache.\n        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder")\n\n        >>> # Update configuration during loading\n        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder", output_attentions=True)\n        >>> model.config.output_attentions\n        True\n\n        >>> # Loading from a PyTorch checkpoint file instead of a TensorFlow model (slower)\n        >>> config = AutoConfig.from_pretrained("./pt_model/shortcut_placeholder_pt_model_config.json")\n        >>> model = BaseAutoModelClass.from_pretrained(\n        ...     "./pt_model/shortcut_placeholder_pytorch_model.bin", from_pt=True, config=config\n        ... )\n        ```\n'
FROM_PRETRAINED_FLAX_DOCSTRING = '\n        Instantiate one of the model classes of the library from a pretrained model.\n\n        The model class to instantiate is selected based on the `model_type` property of the config object (either\n        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it\'s missing, by\n        falling back to using pattern matching on `pretrained_model_name_or_path`:\n\n        List options\n\n        Args:\n            pretrained_model_name_or_path (`str` or `os.PathLike`):\n                Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In this\n                      case, `from_pt` should be set to `True` and a configuration object should be provided as `config`\n                      argument. This loading path is slower than converting the PyTorch model in a TensorFlow model\n                      using the provided conversion scripts and loading the TensorFlow model afterwards.\n            model_args (additional positional arguments, *optional*):\n                Will be passed along to the underlying model `__init__()` method.\n            config ([`PretrainedConfig`], *optional*):\n                Configuration for the model to use instead of an automatically loaded configuration. Configuration can\n                be automatically loaded when:\n\n                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained\n                      model).\n                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the\n                      save directory.\n                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a\n                      configuration JSON file named *config.json* is found in the directory.\n            cache_dir (`str` or `os.PathLike`, *optional*):\n                Path to a directory in which a downloaded pretrained model configuration should be cached if the\n                standard cache should not be used.\n            from_pt (`bool`, *optional*, defaults to `False`):\n                Load the model weights from a PyTorch checkpoint save file (see docstring of\n                `pretrained_model_name_or_path` argument).\n            force_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to force the (re-)download of the model weights and configuration files, overriding the\n                cached versions if they exist.\n            resume_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to delete incompletely received files. Will attempt to resume the download if such a\n                file exists.\n            proxies (`Dict[str, str]`, *optional*):\n                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n                \'http://hostname\': \'foo.bar:4012\'}`. The proxies are used on each request.\n            output_loading_info(`bool`, *optional*, defaults to `False`):\n                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.\n            local_files_only(`bool`, *optional*, defaults to `False`):\n                Whether or not to only look at local files (e.g., not try downloading the model).\n            revision (`str`, *optional*, defaults to `"main"`):\n                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n                identifier allowed by git.\n            trust_remote_code (`bool`, *optional*, defaults to `False`):\n                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option\n                should only be set to `True` for repositories you trust and in which you have read the code, as it will\n                execute code present on the Hub on your local machine.\n            code_revision (`str`, *optional*, defaults to `"main"`):\n                The specific revision to use for the code on the Hub, if the code leaves in a different repository than\n                the rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based\n                system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier\n                allowed by git.\n            kwargs (additional keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or\n                automatically loaded:\n\n                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the\n                      underlying model\'s `__init__` method (we assume all relevant updates to the configuration have\n                      already been done)\n                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class\n                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that\n                      corresponds to a configuration attribute will be used to override said attribute with the\n                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute\n                      will be passed to the underlying model\'s `__init__` function.\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoConfig, BaseAutoModelClass\n\n        >>> # Download model and configuration from huggingface.co and cache.\n        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder")\n\n        >>> # Update configuration during loading\n        >>> model = BaseAutoModelClass.from_pretrained("checkpoint_placeholder", output_attentions=True)\n        >>> model.config.output_attentions\n        True\n\n        >>> # Loading from a PyTorch checkpoint file instead of a TensorFlow model (slower)\n        >>> config = AutoConfig.from_pretrained("./pt_model/shortcut_placeholder_pt_model_config.json")\n        >>> model = BaseAutoModelClass.from_pretrained(\n        ...     "./pt_model/shortcut_placeholder_pytorch_model.bin", from_pt=True, config=config\n        ... )\n        ```\n'

def _get_model_class(config, model_mapping):
    if False:
        for i in range(10):
            print('nop')
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models
    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, 'architectures', [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f'TF{arch}' in name_to_model:
            return name_to_model[f'TF{arch}']
        elif f'Flax{arch}' in name_to_model:
            return name_to_model[f'Flax{arch}']
    return supported_models[0]

class _BaseAutoModelClass:
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        raise EnvironmentError(f'{self.__class__.__name__} is designed to be instantiated using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or `{self.__class__.__name__}.from_config(config)` methods.')

    @classmethod
    def from_config(cls, config, **kwargs):
        if False:
            print('Hello World!')
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        has_remote_code = hasattr(config, 'auto_map') and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(trust_remote_code, config._name_or_path, has_local_code, has_remote_code)
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            if '--' in class_ref:
                (repo_id, class_ref) = class_ref.split('--')
            else:
                repo_id = config.name_or_path
            model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
            if os.path.isdir(config._name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            _ = kwargs.pop('code_revision', None)
            return model_class._from_config(config, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class._from_config(config, **kwargs)
        raise ValueError(f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\nModel type should be one of {', '.join((c.__name__ for c in cls._model_mapping.keys()))}.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if False:
            while True:
                i = 10
        config = kwargs.pop('config', None)
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        kwargs['_from_auto'] = True
        hub_kwargs_names = ['cache_dir', 'force_download', 'local_files_only', 'proxies', 'resume_download', 'revision', 'subfolder', 'use_auth_token', 'token']
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        code_revision = kwargs.pop('code_revision', None)
        commit_hash = kwargs.pop('_commit_hash', None)
        adapter_kwargs = kwargs.pop('adapter_kwargs', None)
        token = hub_kwargs.pop('token', None)
        use_auth_token = hub_kwargs.pop('use_auth_token', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if token is not None:
            hub_kwargs['token'] = token
        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                resolved_config_file = cached_file(pretrained_model_name_or_path, CONFIG_NAME, _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False, **hub_kwargs)
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, '_commit_hash', None)
        if is_peft_available():
            if adapter_kwargs is None:
                adapter_kwargs = {}
                if token is not None:
                    adapter_kwargs['token'] = token
            maybe_adapter_path = find_adapter_config_file(pretrained_model_name_or_path, _commit_hash=commit_hash, **adapter_kwargs)
            if maybe_adapter_path is not None:
                with open(maybe_adapter_path, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)
                    adapter_kwargs['_adapter_model_path'] = pretrained_model_name_or_path
                    pretrained_model_name_or_path = adapter_config['base_model_name_or_path']
        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            if kwargs.get('torch_dtype', None) == 'auto':
                _ = kwargs.pop('torch_dtype')
            if kwargs.get('quantization_config', None) is not None:
                _ = kwargs.pop('quantization_config')
            (config, kwargs) = AutoConfig.from_pretrained(pretrained_model_name_or_path, return_unused_kwargs=True, trust_remote_code=trust_remote_code, code_revision=code_revision, _commit_hash=commit_hash, **hub_kwargs, **kwargs)
            if kwargs_orig.get('torch_dtype', None) == 'auto':
                kwargs['torch_dtype'] = 'auto'
            if kwargs_orig.get('quantization_config', None) is not None:
                kwargs['quantization_config'] = kwargs_orig['quantization_config']
        has_remote_code = hasattr(config, 'auto_map') and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code)
        kwargs['adapter_kwargs'] = adapter_kwargs
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            model_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs)
            _ = hub_kwargs.pop('code_revision', None)
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs)
        raise ValueError(f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\nModel type should be one of {', '.join((c.__name__ for c in cls._model_mapping.keys()))}.")

    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        if False:
            return 10
        '\n        Register a new model for this class.\n\n        Args:\n            config_class ([`PretrainedConfig`]):\n                The configuration corresponding to the model to register.\n            model_class ([`PreTrainedModel`]):\n                The model to register.\n        '
        if hasattr(model_class, 'config_class') and model_class.config_class != config_class:
            raise ValueError(f'The model class you are passing has a `config_class` attribute that is not consistent with the config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix one of those so they match!')
        cls._model_mapping.register(config_class, model_class, exist_ok=exist_ok)

class _BaseAutoBackboneClass(_BaseAutoModelClass):
    _model_mapping = None

    @classmethod
    def _load_timm_backbone_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if False:
            return 10
        requires_backends(cls, ['vision', 'timm'])
        from ...models.timm_backbone import TimmBackboneConfig
        config = kwargs.pop('config', TimmBackboneConfig())
        use_timm = kwargs.pop('use_timm_backbone', True)
        if not use_timm:
            raise ValueError('`use_timm_backbone` must be `True` for timm backbones')
        if kwargs.get('out_features', None) is not None:
            raise ValueError('Cannot specify `out_features` for timm backbones')
        if kwargs.get('output_loading_info', False):
            raise ValueError('Cannot specify `output_loading_info=True` when loading from timm')
        num_channels = kwargs.pop('num_channels', config.num_channels)
        features_only = kwargs.pop('features_only', config.features_only)
        use_pretrained_backbone = kwargs.pop('use_pretrained_backbone', config.use_pretrained_backbone)
        out_indices = kwargs.pop('out_indices', config.out_indices)
        config = TimmBackboneConfig(backbone=pretrained_model_name_or_path, num_channels=num_channels, features_only=features_only, use_pretrained_backbone=use_pretrained_backbone, out_indices=out_indices)
        return super().from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if False:
            while True:
                i = 10
        if kwargs.get('use_timm_backbone', False):
            return cls._load_timm_backbone_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

def insert_head_doc(docstring, head_doc=''):
    if False:
        i = 10
        return i + 15
    if len(head_doc) > 0:
        return docstring.replace('one of the model classes of the library ', f'one of the model classes of the library (with a {head_doc} head) ')
    return docstring.replace('one of the model classes of the library ', 'one of the base model classes of the library ')

def auto_class_update(cls, checkpoint_for_example='bert-base-cased', head_doc=''):
    if False:
        return 10
    model_mapping = cls._model_mapping
    name = cls.__name__
    class_docstring = insert_head_doc(CLASS_DOCSTRING, head_doc=head_doc)
    cls.__doc__ = class_docstring.replace('BaseAutoModelClass', name)
    from_config = copy_func(_BaseAutoModelClass.from_config)
    from_config_docstring = insert_head_doc(FROM_CONFIG_DOCSTRING, head_doc=head_doc)
    from_config_docstring = from_config_docstring.replace('BaseAutoModelClass', name)
    from_config_docstring = from_config_docstring.replace('checkpoint_placeholder', checkpoint_for_example)
    from_config.__doc__ = from_config_docstring
    from_config = replace_list_option_in_docstrings(model_mapping._model_mapping, use_model_types=False)(from_config)
    cls.from_config = classmethod(from_config)
    if name.startswith('TF'):
        from_pretrained_docstring = FROM_PRETRAINED_TF_DOCSTRING
    elif name.startswith('Flax'):
        from_pretrained_docstring = FROM_PRETRAINED_FLAX_DOCSTRING
    else:
        from_pretrained_docstring = FROM_PRETRAINED_TORCH_DOCSTRING
    from_pretrained = copy_func(_BaseAutoModelClass.from_pretrained)
    from_pretrained_docstring = insert_head_doc(from_pretrained_docstring, head_doc=head_doc)
    from_pretrained_docstring = from_pretrained_docstring.replace('BaseAutoModelClass', name)
    from_pretrained_docstring = from_pretrained_docstring.replace('checkpoint_placeholder', checkpoint_for_example)
    shortcut = checkpoint_for_example.split('/')[-1].split('-')[0]
    from_pretrained_docstring = from_pretrained_docstring.replace('shortcut_placeholder', shortcut)
    from_pretrained.__doc__ = from_pretrained_docstring
    from_pretrained = replace_list_option_in_docstrings(model_mapping._model_mapping)(from_pretrained)
    cls.from_pretrained = classmethod(from_pretrained)
    return cls

def get_values(model_mapping):
    if False:
        print('Hello World!')
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)
    return result

def getattribute_from_module(module, attr):
    if False:
        print('Hello World!')
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple((getattribute_from_module(module, a) for a in attr))
    if hasattr(module, attr):
        return getattr(module, attr)
    transformers_module = importlib.import_module('transformers')
    if module != transformers_module:
        try:
            return getattribute_from_module(transformers_module, attr)
        except ValueError:
            raise ValueError(f'Could not find {attr} neither in {module} nor in {transformers_module}!')
    else:
        raise ValueError(f'Could not find {attr} in {transformers_module}!')

class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping):
        if False:
            print('Hello World!')
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for (k, v) in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key):
        if False:
            return 10
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)
        model_types = [k for (k, v) in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        if False:
            return 10
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f'.{module_name}', 'transformers.models')
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        if False:
            return 10
        mapping_keys = [self._load_attr_from_module(key, name) for (key, name) in self._config_mapping.items() if key in self._model_mapping.keys()]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        if False:
            while True:
                i = 10
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        if False:
            return 10
        return bool(self.keys())

    def values(self):
        if False:
            while True:
                i = 10
        mapping_values = [self._load_attr_from_module(key, name) for (key, name) in self._model_mapping.items() if key in self._config_mapping.keys()]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        if False:
            i = 10
            return i + 15
        mapping_items = [(self._load_attr_from_module(key, self._config_mapping[key]), self._load_attr_from_module(key, self._model_mapping[key])) for key in self._model_mapping.keys() if key in self._config_mapping.keys()]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.keys())

    def __contains__(self, item):
        if False:
            print('Hello World!')
        if item in self._extra_content:
            return True
        if not hasattr(item, '__name__') or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key, value, exist_ok=False):
        if False:
            i = 10
            return i + 15
        '\n        Register a new model in this mapping.\n        '
        if hasattr(key, '__name__') and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys() and (not exist_ok):
                raise ValueError(f"'{key}' is already used by a Transformers model.")
        self._extra_content[key] = value
import gc
import json
import os
import re
import warnings
from functools import partial
from pickle import UnpicklingError
from typing import Any, Dict, Optional, Set, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import FlaxGenerationMixin, GenerationConfig
from .modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from .utils import FLAX_WEIGHTS_INDEX_NAME, FLAX_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME, PushToHubMixin, add_code_sample_docstrings, add_start_docstrings_to_model_forward, cached_file, copy_func, download_url, has_file, is_offline_mode, is_remote_url, logging, replace_return_docstrings
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from .utils.import_utils import is_safetensors_available
if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.flax import load_file as safe_load_file
    from safetensors.flax import save_file as safe_save_file
logger = logging.get_logger(__name__)

def quick_gelu(x):
    if False:
        i = 10
        return i + 15
    return x * jax.nn.sigmoid(1.702 * x)
ACT2FN = {'gelu': partial(nn.gelu, approximate=False), 'relu': nn.relu, 'silu': nn.swish, 'swish': nn.swish, 'gelu_new': partial(nn.gelu, approximate=True), 'quick_gelu': quick_gelu}

def dtype_byte_size(dtype):
    if False:
        print('Hello World!')
    '\n    Returns the size (in bytes) occupied by one parameter of type `dtype`. Example:\n    ```py\n    >>> dtype_byte_size(np.float32)\n    4\n    ```\n    '
    if dtype == bool:
        return 1 / 8
    bit_search = re.search('[^\\d](\\d+)$', dtype.name)
    if bit_search is None:
        raise ValueError(f'`dtype` is not a valid dtype: {dtype}.')
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8

def flax_shard_checkpoint(params, max_shard_size='10GB'):
    if False:
        return 10
    '\n    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a\n    given size. The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so\n    there is no optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For\n    example, if the limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as\n    [6GB], [6+2GB], [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].\n\n    <Tip warning={true}>\n\n    If one of the model\'s weight is bigger that `max_shard_size`, it will end up in its own sub-checkpoint which will\n    have a size greater than `max_shard_size`.\n\n    </Tip>\n\n    Args:\n        params (`Union[Dict, FrozenDict]`): A `PyTree` of model parameters.\n        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):\n            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit\n            (like `"5MB"`).\n    '
    max_shard_size = convert_file_size_to_int(max_shard_size)
    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0
    weights = flatten_dict(params, sep='/')
    for item in weights:
        weight_size = weights[item].size * dtype_byte_size(weights[item].dtype)
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0
        current_block[item] = weights[item]
        current_block_size += weight_size
        total_size += weight_size
    sharded_state_dicts.append(current_block)
    if len(sharded_state_dicts) == 1:
        return ({FLAX_WEIGHTS_NAME: sharded_state_dicts[0]}, None)
    weight_map = {}
    shards = {}
    for (idx, shard) in enumerate(sharded_state_dicts):
        shard_file = FLAX_WEIGHTS_NAME.replace('.msgpack', f'-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.msgpack')
        shards[shard_file] = shard
        for weight_name in shard.keys():
            weight_map[weight_name] = shard_file
    metadata = {'total_size': total_size}
    index = {'metadata': metadata, 'weight_map': weight_map}
    return (shards, index)

class FlaxPreTrainedModel(PushToHubMixin, FlaxGenerationMixin):
    """
    Base class for all models.

    [`FlaxPreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """
    config_class = None
    base_model_prefix = ''
    main_input_name = 'input_ids'
    _auto_class = None
    _missing_keys = set()

    def __init__(self, config: PretrainedConfig, module: nn.Module, input_shape: Tuple=(1, 1), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True):
        if False:
            i = 10
            return i + 15
        if config is None:
            raise ValueError('config cannot be None')
        if module is None:
            raise ValueError('module cannot be None')
        self._config = config
        self._module = module
        self.key = PRNGKey(seed)
        self.dtype = dtype
        self.input_shape = input_shape
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        self._is_initialized = _do_init
        if _do_init:
            random_params = self.init_weights(self.key, input_shape)
            params_shape_tree = jax.eval_shape(lambda params: params, random_params)
        else:
            init_fn = partial(self.init_weights, input_shape=input_shape)
            params_shape_tree = jax.eval_shape(init_fn, self.key)
            logger.info(f'Model weights are not initialized as `_do_init` is set to `False`. Make sure to call `{self.__class__.__name__}.init_weights` manually to initialize the weights.')
        self._params_shape_tree = params_shape_tree
        self._required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())
        if _do_init:
            self.params = random_params

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> Dict:
        if False:
            while True:
                i = 10
        raise NotImplementedError(f'init method has to be implemented for {self}')

    def enable_gradient_checkpointing(self):
        if False:
            return 10
        raise NotImplementedError(f'gradient checkpointing method has to be implemented for {self}')

    @classmethod
    def _from_config(cls, config, **kwargs):
        if False:
            while True:
                i = 10
        '\n        All context managers that the model should be initialized under go here.\n        '
        return cls(config, **kwargs)

    @property
    def framework(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        :str: Identifies that this is a Flax model.\n        '
        return 'flax'

    @property
    def config(self) -> PretrainedConfig:
        if False:
            print('Hello World!')
        return self._config

    @property
    def module(self) -> nn.Module:
        if False:
            i = 10
            return i + 15
        return self._module

    @property
    def params(self) -> Union[Dict, FrozenDict]:
        if False:
            while True:
                i = 10
        if not self._is_initialized:
            raise ValueError('`params` cannot be accessed from model when the model is created with `_do_init=False`. You must call `init_weights` manually and store the params outside of the model and pass it explicitly where needed.')
        return self._params

    @property
    def required_params(self) -> Set:
        if False:
            for i in range(10):
                print('nop')
        return self._required_params

    @property
    def params_shape_tree(self) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        return self._params_shape_tree

    @params.setter
    def params(self, params: Union[Dict, FrozenDict]):
        if False:
            i = 10
            return i + 15
        if not self._is_initialized:
            raise ValueError('`params` cannot be set from model when the model is created with `_do_init=False`. You store the params outside of the model.')
        if isinstance(params, FrozenDict):
            params = unfreeze(params)
        param_keys = set(flatten_dict(params).keys())
        if len(self.required_params - param_keys) > 0:
            raise ValueError(f'Some parameters are missing. Make sure that `params` include the following parameters {self.required_params - param_keys}')
        self._params = params

    def _cast_floating_to(self, params: Union[Dict, FrozenDict], dtype: jnp.dtype, mask: Any=None) -> Any:
        if False:
            while True:
                i = 10
        '\n        Helper method to cast floating-point values of given parameter `PyTree` to given `dtype`.\n        '

        def conditional_cast(param):
            if False:
                while True:
                    i = 10
            if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype, jnp.floating):
                param = param.astype(dtype)
            return param
        if mask is None:
            return jax.tree_util.tree_map(conditional_cast, params)
        flat_params = flatten_dict(params)
        (flat_mask, _) = jax.tree_util.tree_flatten(mask)
        for (masked, key) in zip(flat_mask, flat_params.keys()):
            if masked:
                param = flat_params[key]
                flat_params[key] = conditional_cast(param)
        return unflatten_dict(flat_params)

    def to_bf16(self, params: Union[Dict, FrozenDict], mask: Any=None):
        if False:
            return 10
        '\n        Cast the floating-point `params` to `jax.numpy.bfloat16`. This returns a new `params` tree and does not cast\n        the `params` in place.\n\n        This method can be used on TPU to explicitly convert the model parameters to bfloat16 precision to do full\n        half-precision training or to save weights in bfloat16 for inference in order to save memory and improve speed.\n\n        Arguments:\n            params (`Union[Dict, FrozenDict]`):\n                A `PyTree` of model parameters.\n            mask (`Union[Dict, FrozenDict]`):\n                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params\n                you want to cast, and should be `False` for those you want to skip.\n\n        Examples:\n\n        ```python\n        >>> from transformers import FlaxBertModel\n\n        >>> # load model\n        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")\n        >>> # By default, the model parameters will be in fp32 precision, to cast these to bfloat16 precision\n        >>> model.params = model.to_bf16(model.params)\n        >>> # If you want don\'t want to cast certain parameters (for example layer norm bias and scale)\n        >>> # then pass the mask as follows\n        >>> from flax import traverse_util\n\n        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")\n        >>> flat_params = traverse_util.flatten_dict(model.params)\n        >>> mask = {\n        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))\n        ...     for path in flat_params\n        ... }\n        >>> mask = traverse_util.unflatten_dict(mask)\n        >>> model.params = model.to_bf16(model.params, mask)\n        ```'
        return self._cast_floating_to(params, jnp.bfloat16, mask)

    def to_fp32(self, params: Union[Dict, FrozenDict], mask: Any=None):
        if False:
            print('Hello World!')
        '\n        Cast the floating-point `parmas` to `jax.numpy.float32`. This method can be used to explicitly convert the\n        model parameters to fp32 precision. This returns a new `params` tree and does not cast the `params` in place.\n\n        Arguments:\n            params (`Union[Dict, FrozenDict]`):\n                A `PyTree` of model parameters.\n            mask (`Union[Dict, FrozenDict]`):\n                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params\n                you want to cast, and should be `False` for those you want to skip\n\n        Examples:\n\n        ```python\n        >>> from transformers import FlaxBertModel\n\n        >>> # Download model and configuration from huggingface.co\n        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")\n        >>> # By default, the model params will be in fp32, to illustrate the use of this method,\n        >>> # we\'ll first cast to fp16 and back to fp32\n        >>> model.params = model.to_f16(model.params)\n        >>> # now cast back to fp32\n        >>> model.params = model.to_fp32(model.params)\n        ```'
        return self._cast_floating_to(params, jnp.float32, mask)

    def to_fp16(self, params: Union[Dict, FrozenDict], mask: Any=None):
        if False:
            i = 10
            return i + 15
        '\n        Cast the floating-point `parmas` to `jax.numpy.float16`. This returns a new `params` tree and does not cast the\n        `params` in place.\n\n        This method can be used on GPU to explicitly convert the model parameters to float16 precision to do full\n        half-precision training or to save weights in float16 for inference in order to save memory and improve speed.\n\n        Arguments:\n            params (`Union[Dict, FrozenDict]`):\n                A `PyTree` of model parameters.\n            mask (`Union[Dict, FrozenDict]`):\n                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params\n                you want to cast, and should be `False` for those you want to skip\n\n        Examples:\n\n        ```python\n        >>> from transformers import FlaxBertModel\n\n        >>> # load model\n        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")\n        >>> # By default, the model params will be in fp32, to cast these to float16\n        >>> model.params = model.to_fp16(model.params)\n        >>> # If you want don\'t want to cast certain parameters (for example layer norm bias and scale)\n        >>> # then pass the mask as follows\n        >>> from flax import traverse_util\n\n        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")\n        >>> flat_params = traverse_util.flatten_dict(model.params)\n        >>> mask = {\n        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))\n        ...     for path in flat_params\n        ... }\n        >>> mask = traverse_util.unflatten_dict(mask)\n        >>> model.params = model.to_fp16(model.params, mask)\n        ```'
        return self._cast_floating_to(params, jnp.float16, mask)

    @classmethod
    def load_flax_weights(cls, resolved_archive_file):
        if False:
            print('Hello World!')
        try:
            if resolved_archive_file.endswith('.safetensors'):
                state = safe_load_file(resolved_archive_file)
                state = unflatten_dict(state, sep='.')
            else:
                with open(resolved_archive_file, 'rb') as state_f:
                    state = from_bytes(cls, state_f.read())
        except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
            try:
                with open(resolved_archive_file) as f:
                    if f.read().startswith('version'):
                        raise OSError('You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.')
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise EnvironmentError(f'Unable to convert {resolved_archive_file} to Flax deserializable object. ')
        return state

    @classmethod
    def load_flax_sharded_weights(cls, shard_files):
        if False:
            for i in range(10):
                print('nop')
        "\n        This is the same as [`flax.serialization.from_bytes`]\n        (https:lax.readthedocs.io/en/latest/_modules/flax/serialization.html#from_bytes) but for a sharded checkpoint.\n\n        This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being\n        loaded in the model.\n\n        Args:\n            shard_files (`List[str]`:\n                The list of shard files to load.\n\n        Returns:\n            `Dict`: A nested dictionary of the model parameters, in the expected format for flax models : `{'model':\n            {'params': {'...'}}}`.\n        "
        state_sharded_dict = {}
        for shard_file in shard_files:
            try:
                with open(shard_file, 'rb') as state_f:
                    state = from_bytes(cls, state_f.read())
            except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
                with open(shard_file) as f:
                    if f.read().startswith('version'):
                        raise OSError('You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.')
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise EnvironmentError(f'Unable to convert {shard_file} to Flax deserializable object. ')
            state = flatten_dict(state, sep='/')
            state_sharded_dict.update(state)
            del state
            gc.collect()
        return unflatten_dict(state_sharded_dict, sep='/')

    @classmethod
    def can_generate(cls) -> bool:
        if False:
            while True:
                i = 10
        '\n        Returns whether this model can generate sequences with `.generate()`. Returns:\n            `bool`: Whether this model can generate sequences with `.generate()`.\n        '
        if 'GenerationMixin' in str(cls.prepare_inputs_for_generation) and 'GenerationMixin' in str(cls.generate):
            return False
        return True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], dtype: jnp.dtype=jnp.float32, *model_args, config: Optional[Union[PretrainedConfig, str, os.PathLike]]=None, cache_dir: Optional[Union[str, os.PathLike]]=None, ignore_mismatched_sizes: bool=False, force_download: bool=False, local_files_only: bool=False, token: Optional[Union[str, bool]]=None, revision: str='main', **kwargs):
        if False:
            while True:
                i = 10
        '\n        Instantiate a pretrained flax model from a pre-trained model configuration.\n\n        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come\n        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning\n        task.\n\n        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those\n        weights are discarded.\n\n        Parameters:\n            pretrained_model_name_or_path (`str` or `os.PathLike`):\n                Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *pt index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In this case,\n                      `from_pt` should be set to `True`.\n            dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):\n                The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and\n                `jax.numpy.bfloat16` (on TPUs).\n\n                This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If\n                specified all the computation will be performed with the given `dtype`.\n\n                **Note that this only specifies the dtype of the computation and does not influence the dtype of model\n                parameters.**\n\n                If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and\n                [`~FlaxPreTrainedModel.to_bf16`].\n            model_args (sequence of positional arguments, *optional*):\n                All remaining positional arguments will be passed to the underlying model\'s `__init__` method.\n            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):\n                Can be either:\n\n                    - an instance of a class derived from [`PretrainedConfig`],\n                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].\n\n                Configuration for the model to use instead of an automatically loaded configuration. Configuration can\n                be automatically loaded when:\n\n                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained\n                      model).\n                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the\n                      save directory.\n                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a\n                      configuration JSON file named *config.json* is found in the directory.\n            cache_dir (`Union[str, os.PathLike]`, *optional*):\n                Path to a directory in which a downloaded pretrained model configuration should be cached if the\n                standard cache should not be used.\n            from_pt (`bool`, *optional*, defaults to `False`):\n                Load the model weights from a PyTorch checkpoint save file (see docstring of\n                `pretrained_model_name_or_path` argument).\n            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):\n                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size\n                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a\n                checkpoint with 3 labels).\n            force_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to force the (re-)download of the model weights and configuration files, overriding the\n                cached versions if they exist.\n            resume_download (`bool`, *optional*, defaults to `False`):\n                Whether or not to delete incompletely received files. Will attempt to resume the download if such a\n                file exists.\n            proxies (`Dict[str, str]`, *optional*):\n                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n                \'http://hostname\': \'foo.bar:4012\'}`. The proxies are used on each request.\n            local_files_only(`bool`, *optional*, defaults to `False`):\n                Whether or not to only look at local files (i.e., do not try to download the model).\n            token (`str` or `bool`, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use\n                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).\n            revision (`str`, *optional*, defaults to `"main"`):\n                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n                identifier allowed by git.\n\n\n                <Tip>\n\n                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".\n\n                </Tip>\n\n            subfolder (`str`, *optional*, defaults to `""`):\n                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can\n                specify the folder name here.\n            kwargs (remaining dictionary of keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or\n                automatically loaded:\n\n                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the\n                      underlying model\'s `__init__` method (we assume all relevant updates to the configuration have\n                      already been done)\n                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class\n                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that\n                      corresponds to a configuration attribute will be used to override said attribute with the\n                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute\n                      will be passed to the underlying model\'s `__init__` function.\n\n        Examples:\n\n        ```python\n        >>> from transformers import BertConfig, FlaxBertModel\n\n        >>> # Download model and configuration from huggingface.co and cache.\n        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")\n        >>> # Model was saved using *save_pretrained(\'./test/saved_model/\')* (for example purposes, not runnable).\n        >>> model = FlaxBertModel.from_pretrained("./test/saved_model/")\n        >>> # Loading from a PyTorch checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).\n        >>> config = BertConfig.from_json_file("./pt_model/config.json")\n        >>> model = FlaxBertModel.from_pretrained("./pt_model/pytorch_model.bin", from_pt=True, config=config)\n        ```'
        from_pt = kwargs.pop('from_pt', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        use_auth_token = kwargs.pop('use_auth_token', None)
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        from_pipeline = kwargs.pop('_from_pipeline', None)
        from_auto_class = kwargs.pop('_from_auto', False)
        _do_init = kwargs.pop('_do_init', True)
        subfolder = kwargs.pop('subfolder', '')
        commit_hash = kwargs.pop('_commit_hash', None)
        _ = kwargs.pop('adapter_kwargs', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if trust_remote_code is True:
            logger.warning('The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.')
        user_agent = {'file_type': 'model', 'framework': 'flax', 'from_auto_class': from_auto_class}
        if from_pipeline is not None:
            user_agent['using_pipeline'] = from_pipeline
        if is_offline_mode() and (not local_files_only):
            logger.info('Offline mode: forcing local_files_only=True')
            local_files_only = True
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            (config, model_kwargs) = cls.config_class.from_pretrained(config_path, cache_dir=cache_dir, return_unused_kwargs=True, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, subfolder=subfolder, _from_auto=from_auto_class, _from_pipeline=from_pipeline, _commit_hash=commit_hash, **kwargs)
        else:
            model_kwargs = kwargs.copy()
        if commit_hash is None:
            commit_hash = getattr(config, '_commit_hash', None)
        model_kwargs['dtype'] = dtype
        is_sharded = False
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_INDEX_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                elif is_safetensors_available() and os.path.isfile(os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                elif from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
                elif from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME)
                    is_sharded = True
                elif is_safetensors_available() and os.path.isfile(os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                    raise NotImplementedError('Support for sharded checkpoints using safetensors is coming soon!')
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)):
                    raise EnvironmentError(f'Error no file named {FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those weights.')
                else:
                    raise EnvironmentError(f'Error no file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}.')
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                if from_pt:
                    filename = WEIGHTS_NAME
                else:
                    filename = FLAX_WEIGHTS_NAME
                try:
                    cached_file_kwargs = {'cache_dir': cache_dir, 'force_download': force_download, 'proxies': proxies, 'resume_download': resume_download, 'local_files_only': local_files_only, 'token': token, 'user_agent': user_agent, 'revision': revision, 'subfolder': subfolder, '_raise_exceptions_for_missing_entries': False, '_commit_hash': commit_hash}
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
                    if resolved_archive_file is None and filename == FLAX_WEIGHTS_NAME:
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, FLAX_WEIGHTS_INDEX_NAME, **cached_file_kwargs)
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None and from_pt:
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **cached_file_kwargs)
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        filename = SAFE_WEIGHTS_NAME
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME, **cached_file_kwargs)
                    if resolved_archive_file is None:
                        has_file_kwargs = {'revision': revision, 'proxies': proxies, 'token': token}
                        if has_file(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME, **has_file_kwargs):
                            is_sharded = True
                            raise NotImplementedError('Support for sharded checkpoints using safetensors is coming soon!')
                        elif has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {FLAX_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those weights.')
                        elif has_file(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **has_file_kwargs):
                            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {FLAX_WEIGHTS_INDEX_NAME} but there is a sharded file for PyTorch weights. Use `from_pt=True` to load this model from those weights.')
                        else:
                            raise EnvironmentError(f'{pretrained_model_name_or_path} does not appear to have a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}.')
                except EnvironmentError:
                    raise
                except Exception:
                    raise EnvironmentError(f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}.")
            if is_local:
                logger.info(f'loading weights file {archive_file}')
                resolved_archive_file = archive_file
                filename = resolved_archive_file.split(os.path.sep)[-1]
            else:
                logger.info(f'loading weights file {filename} from cache at {resolved_archive_file}')
        else:
            resolved_archive_file = None
        if is_sharded:
            (resolved_archive_file, _) = get_checkpoint_shard_files(pretrained_model_name_or_path, resolved_archive_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, user_agent=user_agent, revision=revision, subfolder=subfolder, _commit_hash=commit_hash)
        safetensors_from_pt = False
        if filename == SAFE_WEIGHTS_NAME:
            with safe_open(resolved_archive_file, framework='flax') as f:
                safetensors_metadata = f.metadata()
            if safetensors_metadata is None or safetensors_metadata.get('format') not in ['pt', 'tf', 'flax']:
                raise OSError(f'The safetensors archive passed at {resolved_archive_file} does not contain the valid metadata. Make sure you save your model with the `save_pretrained` method.')
            safetensors_from_pt = safetensors_metadata.get('format') == 'pt'
        model = cls(config, *model_args, _do_init=_do_init, **model_kwargs)
        if from_pt or safetensors_from_pt:
            state = load_pytorch_checkpoint_in_flax_state_dict(model, resolved_archive_file, is_sharded)
        else:
            if is_sharded:
                state = cls.load_flax_sharded_weights(resolved_archive_file)
            else:
                state = cls.load_flax_weights(resolved_archive_file)
            if _do_init:
                state = jax.tree_util.tree_map(jnp.array, state)
            else:
                state = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices('cpu')[0]), state)
        if 'batch_stats' in state:
            if cls.base_model_prefix not in dict(model.params_shape_tree['params']) and cls.base_model_prefix in state['params']:
                state['params'] = state['params'][cls.base_model_prefix]
                state['batch_stats'] = state['batch_stats'][cls.base_model_prefix]
            if cls.base_model_prefix in dict(model.params_shape_tree['params']) and cls.base_model_prefix not in state['params']:
                state = {'params': {cls.base_model_prefix: state['params']}, 'batch_stats': {cls.base_model_prefix: state['batch_stats']}}
        else:
            if cls.base_model_prefix not in dict(model.params_shape_tree) and cls.base_model_prefix in state:
                state = state[cls.base_model_prefix]
            if cls.base_model_prefix in dict(model.params_shape_tree) and cls.base_model_prefix not in state:
                state = {cls.base_model_prefix: state}
        state = flatten_dict(state)
        random_state = flatten_dict(unfreeze(model.params if _do_init else model.params_shape_tree))
        missing_keys = model.required_params - set(state.keys())
        unexpected_keys = set(state.keys()) - model.required_params
        for unexpected_key in unexpected_keys.copy():
            if 'num_batches_tracked' in unexpected_key[-1]:
                unexpected_keys.remove(unexpected_key)
        if missing_keys and (not _do_init):
            logger.warning(f'The checkpoint {pretrained_model_name_or_path} is missing required keys: {missing_keys}. Make sure to call model.init_weights to initialize the missing weights.')
            cls._missing_keys = missing_keys
        mismatched_keys = []
        for key in state.keys():
            if key in random_state and state[key].shape != random_state[key].shape:
                if ignore_mismatched_sizes:
                    mismatched_keys.append((key, state[key].shape, random_state[key].shape))
                    state[key] = random_state[key]
                else:
                    raise ValueError(f'Trying to load the pretrained weight for {key} failed: checkpoint has shape {state[key].shape} which is incompatible with the model shape {random_state[key].shape}. Using `ignore_mismatched_sizes=True` if you really want to load this checkpoint inside this model.')
        if missing_keys and _do_init:
            for missing_key in missing_keys:
                state[missing_key] = random_state[missing_key]
        for unexpected_key in unexpected_keys:
            del state[unexpected_key]
        if len(unexpected_keys) > 0:
            logger.warning(f'Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).')
        else:
            logger.info(f'All model checkpoint weights were used when initializing {model.__class__.__name__}.\n')
        if len(missing_keys) > 0:
            logger.warning(f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
        elif len(mismatched_keys) == 0:
            logger.info(f'All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint was trained on, you can already use {model.__class__.__name__} for predictions without further training.')
        if len(mismatched_keys) > 0:
            mismatched_warning = '\n'.join([f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated' for (key, shape1, shape2) in mismatched_keys])
            logger.warning(f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized because the shapes did not match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
        param_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, state)
        fp16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.float16]
        bf16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.bfloat16]
        if len(fp16_params) > 0:
            logger.warning(f'Some of the weights of {model.__class__.__name__} were initialized in float16 precision from the model checkpoint at {pretrained_model_name_or_path}:\n{fp16_params}\nYou should probably UPCAST the model weights to float32 if this was not intended. See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this.')
        if len(bf16_params) > 0:
            logger.warning(f'Some of the weights of {model.__class__.__name__} were initialized in bfloat16 precision from the model checkpoint at {pretrained_model_name_or_path}:\n{bf16_params}\nYou should probably UPCAST the model weights to float32 if this was not intended. See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this.')
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, token=token, revision=revision, subfolder=subfolder, _from_auto=from_auto_class, _from_pipeline=from_pipeline, **kwargs)
            except OSError:
                logger.info('Generation config file not found, using a generation config created from the model config.')
                pass
        if _do_init:
            model.params = unflatten_dict(state)
            return model
        else:
            return (model, unflatten_dict(state))

    def save_pretrained(self, save_directory: Union[str, os.PathLike], params=None, push_to_hub=False, max_shard_size='10GB', token: Optional[Union[str, bool]]=None, safe_serialization: bool=False, **kwargs):
        if False:
            return 10
        '\n        Save a model and its configuration file to a directory, so that it can be re-loaded using the\n        `[`~FlaxPreTrainedModel.from_pretrained`]` class method\n\n        Arguments:\n            save_directory (`str` or `os.PathLike`):\n                Directory to which to save. Will be created if it doesn\'t exist.\n            push_to_hub (`bool`, *optional*, defaults to `False`):\n                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the\n                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your\n                namespace).\n            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):\n                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size\n                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).\n\n                <Tip warning={true}>\n\n                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard\n                which will be bigger than `max_shard_size`.\n\n                </Tip>\n\n            token (`str` or `bool`, *optional*):\n                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use\n                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).\n            kwargs (`Dict[str, Any]`, *optional*):\n                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.\n            safe_serialization (`bool`, *optional*, defaults to `False`):\n                Whether to save the model using `safetensors` or through msgpack.\n        '
        use_auth_token = kwargs.pop('use_auth_token', None)
        if use_auth_token is not None:
            warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
            if token is not None:
                raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
            token = use_auth_token
        if token is not None:
            kwargs['token'] = token
        if os.path.isfile(save_directory):
            logger.error(f'Provided path ({save_directory}) should be a directory, not a file')
            return
        os.makedirs(save_directory, exist_ok=True)
        if push_to_hub:
            commit_message = kwargs.pop('commit_message', None)
            repo_id = kwargs.pop('repo_id', save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        save_directory = os.path.abspath(save_directory)
        self.config.architectures = [self.__class__.__name__[4:]]
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)
        self.config.save_pretrained(save_directory)
        if self.can_generate():
            self.generation_config.save_pretrained(save_directory)
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else FLAX_WEIGHTS_NAME
        output_model_file = os.path.join(save_directory, weights_name)
        (shards, index) = flax_shard_checkpoint(params if params is not None else self.params, max_shard_size)
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            weights_no_suffix = weights_name.replace('.bin', '').replace('.safetensors', '')
            if filename.startswith(weights_no_suffix) and os.path.isfile(full_filename) and (filename not in shards.keys()):
                os.remove(full_filename)
        if index is None:
            if safe_serialization:
                params = params if params is not None else self.params
                flat_dict = flatten_dict(params, sep='.')
                safe_save_file(flat_dict, output_model_file, metadata={'format': 'flax'})
            else:
                with open(output_model_file, 'wb') as f:
                    params = params if params is not None else self.params
                    model_bytes = to_bytes(params)
                    f.write(model_bytes)
        else:
            save_index_file = os.path.join(save_directory, FLAX_WEIGHTS_INDEX_NAME)
            with open(save_index_file, 'w', encoding='utf-8') as f:
                content = json.dumps(index, indent=2, sort_keys=True) + '\n'
                f.write(content)
            logger.info(f'The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the index located at {save_index_file}.')
            for (shard_file, shard) in shards.items():
                with open(os.path.join(save_directory, shard_file), mode='wb') as f:
                    params = unflatten_dict(shard, sep='/')
                    shard_bytes = to_bytes(params)
                    f.write(shard_bytes)
        logger.info(f'Model weights saved in {output_model_file}')
        if push_to_hub:
            self._upload_modified_files(save_directory, repo_id, files_timestamps, commit_message=commit_message, token=token)

    @classmethod
    def register_for_auto_class(cls, auto_class='FlaxAutoModel'):
        if False:
            return 10
        '\n        Register this class with a given auto class. This should only be used for custom models as the ones in the\n        library are already mapped with an auto class.\n\n        <Tip warning={true}>\n\n        This API is experimental and may have some slight breaking changes in the next releases.\n\n        </Tip>\n\n        Args:\n            auto_class (`str` or `type`, *optional*, defaults to `"FlaxAutoModel"`):\n                The auto class to register this new model with.\n        '
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__
        import transformers.models.auto as auto_module
        if not hasattr(auto_module, auto_class):
            raise ValueError(f'{auto_class} is not a valid auto class.')
        cls._auto_class = auto_class
FlaxPreTrainedModel.push_to_hub = copy_func(FlaxPreTrainedModel.push_to_hub)
if FlaxPreTrainedModel.push_to_hub.__doc__ is not None:
    FlaxPreTrainedModel.push_to_hub.__doc__ = FlaxPreTrainedModel.push_to_hub.__doc__.format(object='model', object_class='FlaxAutoModel', object_files='model checkpoint')

def overwrite_call_docstring(model_class, docstring):
    if False:
        for i in range(10):
            print('nop')
    model_class.__call__ = copy_func(model_class.__call__)
    model_class.__call__.__doc__ = None
    model_class.__call__ = add_start_docstrings_to_model_forward(docstring)(model_class.__call__)

def append_call_sample_docstring(model_class, checkpoint, output_type, config_class, mask=None):
    if False:
        print('Hello World!')
    model_class.__call__ = copy_func(model_class.__call__)
    model_class.__call__ = add_code_sample_docstrings(checkpoint=checkpoint, output_type=output_type, config_class=config_class, model_cls=model_class.__name__)(model_class.__call__)

def append_replace_return_docstrings(model_class, output_type, config_class):
    if False:
        i = 10
        return i + 15
    model_class.__call__ = copy_func(model_class.__call__)
    model_class.__call__ = replace_return_docstrings(output_type=output_type, config_class=config_class)(model_class.__call__)
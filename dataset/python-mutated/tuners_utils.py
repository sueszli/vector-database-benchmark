from __future__ import annotations
import logging
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Union
from torch import nn
from peft.utils import COMMON_LAYERS_PATTERN
from ..config import PeftConfig
from ..utils import ModulesToSaveWrapper, _get_submodules
logger = logging.getLogger(__name__)

class BaseTuner(nn.Module, ABC):
    """
    A base tuner model that provides the common methods and attributes for all tuners that are injectable into a
    torch.nn.Module

    For adding a new Tuner class, one needs to overwrite the following methods:

    - **_prepare_adapter_config**:
        A private method to eventually prepare the adapter config, for example in case the field `target_modules` is
        missing.
    - **_check_target_module_exists**:
        A helper private method to check if the passed module's key name matches any of the target modules in the
        adatper_config.
    - **_create_and_replace**:
        A private method to create and replace the target module with the adapter module.
    - **_check_target_module_exists**:
        A private helper method to check if the passed module's key name matches any of the target modules in the
        adatper_config.

    The easiest is to check what is done in the `peft.tuners.lora.LoraModel` class.

    Attributes:
        model (`torch.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        forward (`Callable`):
            The forward method of the model.
        peft_config (`Union[`PeftConfig`, dict[str, PeftConfig]]`):
            The adapter configuration object, it should be a dictionary of `str` to `PeftConfig` objects. One can also
            pass a PeftConfig object and a new adapter will be created with the default name `adapter` or create a new
            dictionary with a key `adapter_name` and a value of that peft config.
        config (`dict[str, Any]`):
            The model configuration object, it should be a dictionary of `str` to `Any` objects.
    """

    def __init__(self, model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.model = model
        if not hasattr(self, 'peft_config'):
            self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        else:
            logger.info('Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!')
            if isinstance(peft_config, PeftConfig):
                self.peft_config[adapter_name] = peft_config
            else:
                self.peft_config.update(peft_config)
        self.active_adapter = adapter_name
        if not hasattr(self, 'config'):
            self.config = {'model_type': 'custom'}
        self.inject_adapter(self.model, adapter_name)
        self.model.peft_config = self.peft_config

    @property
    def active_adapters(self) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        return self.active_adapter

    def forward(self, *args: Any, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        return self.model.forward(*args, **kwargs)

    @abstractmethod
    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        if False:
            return 10
        '\n        A private method to eventually prepare the adapter config. For transformers based models, if\n        `peft_config.target_modules` is None, we can automatically infer the target modules from the\n        `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING`. This method can be further refactored in the future to\n        automatically infer it for all tuner models.\n\n        Check out `peft.tuner.lora.LoraModel._prepare_adapter_config` for an example.\n\n        Args:\n            peft_config (`str`):\n                The adapter config.\n            model_config (`str`):\n                The transformers model config, that config should contain the `model_type` key.\n        '
        ...

    @abstractmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "\n        A helper private method to check if the passed module's key name matches any of the target modules in the\n        `peft_config.target_modules` list. If it does, return `True`, else return `False`.\n\n        Args:\n            peft_config (`PeftConfig`):\n                The adapter config.\n            key (`str`):\n                The module's key name.\n        "
        ...

    @abstractmethod
    def _create_and_replace(self, peft_config: PeftConfig, adapter_name: str, target: nn.Module, target_name: str, parent: nn.Module, **optional_kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Inplace replacement of the target module with the adapter layer. This method needs to be overriden by all the\n        tuner classes.\n\n        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.\n\n        Args:\n            peft_config (`PeftConfig`):\n                The adapter config.\n            adapter_name (`str`):\n                The adapter name.\n            target (`nn.Module`):\n                The target module.\n            target_name (`str`):\n                The target module's name.\n            parent (`nn.Module`):\n                The parent module.\n            **optional_kwargs (`dict`):\n                The optional keyword arguments to pass to deal with particular cases (e.g. 8bit, 4bit quantization)\n        "
        ...

    @abstractmethod
    def _mark_only_adapters_as_trainable(self):
        if False:
            return 10
        '\n        A helper method to mark only the adapter layers as trainable (i.e. module.requires_grad = False) This needs to\n        be overriden for all tuner classes to match the correct key names.\n\n        Check `peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable` for an example.\n        '
        ...

    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        if False:
            while True:
                i = 10
        '\n        A helper method to check the config when a new adapter is being added.\n\n        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.\n\n        '
        pass

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the\n        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.\n\n        The corresponding PEFT config is directly retrieved from the `peft_config` attribute of the BaseTuner class.\n\n        Args:\n            model (`nn.Module`):\n                The model to be tuned.\n            adapter_name (`str`):\n                The adapter name.\n        '
        peft_config = self.peft_config[adapter_name]
        self._check_new_adapter_config(peft_config)
        is_target_modules_in_base_model = False
        key_list = [key for (key, _) in model.named_modules()]
        _check_for_modules_to_save = getattr(peft_config, 'modules_to_save', None) is not None
        _has_modules_to_save = False
        model_config = getattr(model, 'config', {'model_type': 'custom'})
        if hasattr(model_config, 'to_dict'):
            model_config = model_config.to_dict()
        peft_config = self._prepare_adapter_config(peft_config, model_config)
        for key in key_list:
            if _check_for_modules_to_save and any((key.endswith(f'{module_to_save}') for module_to_save in peft_config.modules_to_save)):
                (parent, target, target_name) = _get_submodules(model, key)
                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)
                _has_modules_to_save = True
                continue
            if not self._check_target_module_exists(peft_config, key):
                continue
            is_target_modules_in_base_model = True
            (parent, target, target_name) = _get_submodules(model, key)
            optional_kwargs = {'loaded_in_8bit': getattr(model, 'is_loaded_in_8bit', False), 'loaded_in_4bit': getattr(model, 'is_loaded_in_4bit', False), 'current_key': key}
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, **optional_kwargs)
        if not is_target_modules_in_base_model:
            raise ValueError(f'Target modules {peft_config.target_modules} not found in the base model. Please check the target modules and try again.')
        self._mark_only_adapters_as_trainable()
        if self.peft_config[adapter_name].inference_mode:
            for (n, p) in self.model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False
        if _has_modules_to_save:
            if not hasattr(model, 'modules_to_save'):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))

    def merge_adapter(self):
        if False:
            i = 10
            return i + 15
        '\n        This method merges the LoRa layers into the base model.\n        '
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.merge()

    def unmerge_adapter(self):
        if False:
            i = 10
            return i + 15
        '\n        This method unmerges the LoRa layers from the base model.\n        '
        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.unmerge()

class BaseTunerLayer(ABC):
    """
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_plugable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """
    active_adapter = None
    adapter_layer_names: tuple[str] = ()
    other_param_names: tuple[str] = ()
    _disable_adapters: bool = False
    _active_adapter: str | list[str] = 'default'
    merged_adapters: list[str] = []

    def merge(self, *args) -> None:
        if False:
            return 10
        raise NotImplementedError

    def unmerge(self, *args) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        if False:
            while True:
                i = 10
        return self._active_adapter

    @property
    def active_adapters(self):
        if False:
            return 10
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        return self.active_adapter

    def enable_adapters(self, enabled: bool):
        if False:
            return 10
        'Toggle the enabling and disabling of adapters\n\n        Takes care of setting the requires_grad flag for the adapter weights.\n\n        Args:\n            enabled (bool): True to enable adapters, False to disable adapters\n        '
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: str | list[str]):
        if False:
            return 10
        'Set the active adapter\n\n        Args:\n            adapter_name (str): The name of the adapter to set as active\n        '
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for (key, layer) in module_dict.items():
                if key in adapter_names:
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)
        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        'Return a sorted list of all available adapter names'
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            attr = getattr(self, name)
            if hasattr(attr, 'keys'):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Delete an adapter from the layer\n\n        This should be called on all adapter layers, or else we will get an inconsistent state.\n\n        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important\n        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.\n\n        Args:\n            adapter_name (`str`): The name of the adapter to delete\n\n        '
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]
        if adapter_name in self.active_adapters:
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(f'Adapter {adapter_name} was active which is now deleted. Setting active adapter to {new_active_adapter}.')
                    self.set_adapter(remaining_adapters[0])

def check_target_module_exists(config, key: str) -> bool | re.Match[str] | None:
    if False:
        for i in range(10):
            print('nop')
    "A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.\n\n    Args:\n        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from\n        key (`str`): A key to search any matches in config\n\n    Returns:\n        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or\n        None if no match found\n    "
    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    else:
        target_module_found = key in config.target_modules or any((key.endswith(f'.{target_key}') for target_key in config.target_modules))
        is_using_layer_indexes = getattr(config, 'layers_to_transform', None) is not None
        layer_indexing_pattern = getattr(config, 'layers_pattern', None)
        if is_using_layer_indexes and target_module_found:
            layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
            layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
            for pattern in layers_pattern:
                layer_index = re.match(f'.*.{pattern}\\.(\\d+)\\.*', key)
                if layer_index is not None:
                    layer_index = int(layer_index.group(1))
                    if isinstance(config.layers_to_transform, int):
                        target_module_found = layer_index == config.layers_to_transform
                    else:
                        target_module_found = layer_index in config.layers_to_transform
                    break
                else:
                    target_module_found = False
    return target_module_found

def inspect_matched_modules(tuner: BaseTuner, adapter_name: str='default') -> dict:
    if False:
        return 10
    '\n    A helper function to inspect the set of matched and unmatched modules for a PEFT model and the given adapter.\n    '
    config = tuner.peft_config[adapter_name]
    key_list = [key for (key, _) in tuner.model.named_modules()]
    module_dict = {'matched': [], 'unmatched': []}
    for key in key_list:
        if tuner._check_target_module_exists(config, key):
            module_dict['matched'].append(key)
        else:
            module_dict['unmatched'].append(key)
    return module_dict
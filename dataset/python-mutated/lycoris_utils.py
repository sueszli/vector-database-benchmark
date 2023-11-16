import re
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Optional, Set, Type, Union
import torch
import torch.nn as nn
from tqdm import tqdm
from peft.config import PeftConfig
from peft.utils import ModulesToSaveWrapper, _get_submodules
from .tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists

@dataclass
class LycorisConfig(PeftConfig):
    """
    A base config for LyCORIS like adapters
    """
    rank_pattern: Optional[dict] = field(default_factory=dict, metadata={'help': 'The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}'})
    alpha_pattern: Optional[dict] = field(default_factory=dict, metadata={'help': 'The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `alpha`. For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}'})

class LycorisLayer(BaseTunerLayer, nn.Module):
    """
    A base layer for LyCORIS like adapters
    """
    other_param_names = ('r', 'alpha', 'scaling', 'rank_dropout', 'module_dropout')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.r = {}
        self.alpha = {}
        self.scaling = {}
        self.rank_dropout = {}
        self.module_dropout = {}
        self._disable_adapters = False
        self.merged_adapters = []

    @property
    @abstractmethod
    def _available_adapters(self) -> Set[str]:
        if False:
            return 10
        ...

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        kwargs = kwargs.copy()
        final_device = kwargs.pop('device', 'cpu')
        cls.__init__(self, *args, device='meta', **kwargs)
        self.to_empty(device=final_device)

    def _op(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def create_adapter_parameters(self, adapter_name: str, r: int, **kwargs):
        if False:
            i = 10
            return i + 15
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._op(x, self.weight)
        elif self.merged:
            result = self._op(x, self.weight)
        else:
            weight = self.weight.data
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue
                module_dropout = self.module_dropout[active_adapter]
                if not self.training or (self.training and torch.rand(1) > module_dropout):
                    weight = weight + self.get_delta_weight(active_adapter)
            result = self._op(x, weight)
        result = result.to(previous_dtype)
        return result

    @abstractmethod
    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        if False:
            print('Hello World!')
        ...

    def merge(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.merged:
            warnings.warn(f"Already following adapters were merged {','.join(self.merged_adapters)}. You are now additionally merging {','.join(self.active_adapters)}.")
        for active_adapter in self.active_adapters:
            if active_adapter in self._available_adapters:
                self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    @abstractmethod
    def reset_adapter_parameters(self, adapter_name: str):
        if False:
            while True:
                i = 10
        ...

    def set_scale(self, adapter, scale):
        if False:
            print('Hello World!')
        if adapter not in self._available_adapters:
            return
        self.scaling[adapter] = scale * self.alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if False:
            i = 10
            return i + 15
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue
            self.scaling[active_adapter] *= scale

    def unmerge(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                self.weight.data -= self.get_delta_weight(active_adapter)

    def unscale_layer(self, scale=None) -> None:
        if False:
            while True:
                i = 10
        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue
            if scale is None:
                self.scaling[active_adapter] = self.alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    @abstractmethod
    def update_layer(self, adapter_name: str, r: int, alpha: float, **kwargs):
        if False:
            while True:
                i = 10
        ...

class LycorisTuner(BaseTuner):
    """
    A base tuner for LyCORIS like adapters
    """
    prefix: str
    layers_mapping: Dict[Type[torch.nn.Module], Type[LycorisLayer]]

    def __init__(self, model, config, adapter_name):
        if False:
            print('Hello World!')
        super().__init__(model, config, adapter_name)

    def __getattr__(self, name: str):
        if False:
            print('Hello World!')
        'Forward missing attributes to the wrapped module.'
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _check_target_module_exists(config, key):
        if False:
            return 10
        return check_target_module_exists(config, key)

    def _create_and_replace(self, config: LycorisConfig, adapter_name: str, target: Union[LycorisLayer, nn.Module], target_name, parent, current_key, **optional_kwargs):
        if False:
            while True:
                i = 10
        '\n        A private method to create and replace the target module with the adapter module.\n        '
        pattern_keys = list(chain(config.rank_pattern.keys(), config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(f'(.*\\.)?{key}$', current_key), pattern_keys), target_name)
        kwargs = config.to_dict()
        kwargs['r'] = config.rank_pattern.get(target_name_key, config.r)
        kwargs['alpha'] = config.alpha_pattern.get(target_name_key, config.alpha)
        if isinstance(target, LycorisLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @classmethod
    def _create_new_module(cls, config: LycorisConfig, adapter_name: str, target: nn.Module, **kwargs) -> LycorisLayer:
        if False:
            print('Hello World!')
        new_module_cls = None
        for (subtype, target_cls) in cls.layers_mapping.items():
            if isinstance(target, subtype):
                new_module_cls = target_cls
                break
        if new_module_cls is None:
            raise ValueError(f"Target module not found, currently only adapters for {', '.join([x.__name__ for x in cls.modules_mapping.keys()])} are supported")
        if isinstance(target, torch.nn.Conv2d):
            new_module = new_module_cls(target.in_channels, target.out_channels, target.weight.size()[2:], stride=target.stride, padding=target.padding, dilation=target.dilation, groups=target.groups, bias=target.bias is not None, padding_mode=target.padding_mode, device=target.weight.device, dtype=target.weight.dtype, adapter_name=adapter_name, **kwargs)
        elif isinstance(target, torch.nn.Linear):
            new_module = new_module_cls(target.in_features, target.out_features, bias=target.bias is not None, device=target.weight.device, dtype=target.weight.dtype, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError('Target module not found, currently only adapters for nn.Linear and nn.Conv2d are supported')
        return new_module

    def _mark_only_adapters_as_trainable(self) -> None:
        if False:
            i = 10
            return i + 15
        for (n, p) in self.model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if False:
            for i in range(10):
                print('nop')
        if peft_config.target_modules is None:
            raise ValueError('Please specify `target_modules` in `peft_config`')
        return peft_config

    def _replace_module(self, parent, child_name, new_module, child):
        if False:
            return 10
        setattr(parent, child_name, new_module)
        new_module.weight = child.weight
        if hasattr(child, 'bias'):
            new_module.bias = child.bias
        if getattr(child, 'state', None) is not None:
            new_module.state = child.state
            new_module.to(child.weight.device)
        for (name, module) in new_module.named_modules():
            if self.prefix in name:
                module.to(child.weight.device)

    def _set_adapter_layers(self, enabled=True):
        if False:
            for i in range(10):
                print('nop')
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool=False):
        if False:
            print('Hello World!')
        if merge:
            if getattr(self.model, 'quantization_method', None) == 'gptq':
                raise ValueError('Cannot merge LOHA layers when the model is gptq quantized')
        key_list = [key for (key, _) in self.model.named_modules() if 'hada' not in key]
        desc = 'Unloading ' + ('and merging ' if merge else '') + 'model'
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                (parent, target, target_name) = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LycorisLayer):
                if isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(target.in_channels, target.out_channels, kernel_size=target.kernel_size, stride=target.stride, padding=target.padding, dilation=target.dilation)
                elif isinstance(target, nn.Linear):
                    bias = target.bias is not None
                    new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias, device=target.weight.device)
                else:
                    raise ValueError('Cannot convert current module to torch module, currently only adapters for nn.Linear and nn.Conv2d are supported')
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])
        return self.model

    def enable_adapter_layers(self):
        if False:
            return 10
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        if False:
            return 10
        self._set_adapter_layers(enabled=False)

    def merge_and_unload(self, progressbar: bool=False):
        if False:
            i = 10
            return i + 15
        return self._unload_and_optionally_merge(progressbar=progressbar)

    def set_adapter(self, adapter_name):
        if False:
            while True:
                i = 10
        for module in self.model.modules():
            if isinstance(module, LycorisLayer):
                if module.merged:
                    warnings.warn('Adapter cannot be set when the model is merged. Unmerging the model first.')
                    module.unmerge()
                module.set_adapter(adapter_name)

    def delete_adapter(self, adapter_name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deletes an existing adapter.\n\n        Args:\n            adapter_name (`str`): Name of the adapter to be deleted.\n        '
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f'Adapter {adapter_name} does not exist')
        del self.peft_config[adapter_name]
        key_list = [key for (key, _) in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            (_, target, _) = _get_submodules(self.model, key)
            if isinstance(target, LycorisLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]
        self.active_adapter = new_adapter or []
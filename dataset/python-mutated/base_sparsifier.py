import abc
import copy
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple, List, Type
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import type_before_parametrizations
from .utils import module_contains_param, swap_module, FakeSparsity, get_arg_info_from_tensor_fqn, module_to_fqn
__all__ = ['BaseSparsifier']
SUPPORTED_MODULES = {nn.Linear}
KEYS_NOT_IN_STATE_DICT = ['module', 'module_fqn', 'tensor_name']
__all__ = ['BaseSparsifier']

class BaseSparsifier(abc.ABC):
    """Base class for all sparsifiers.

    Abstract methods that need to be implemented:

    - update_mask: Function to compute a new mask for all keys in the
        `groups`.

    Args:
        - model [nn.Module]: model to configure. The model itself is not saved
            but used for the state_dict saving / loading.
        - config [list]: configuration elements should be a dict map that includes
            `tensor_fqn` of tensors to sparsify
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.

    Example::

        >>> # xdoctest: +SKIP("Can't instantiate abstract class BaseSparsifier with abstract method update_mask")
        >>> config = [{'tensor_fqn': 'layer1.weight', 'tensor_fqn': 'linear2.weight2', 'sparsity_level': 0.5}]
        >>> defaults = {'sparsity_level': 0.7}
        >>> # model.layer1.weight will have `sparsity_level` = 0.7 (getting default)
        >>> sparsifier = BaseSparsifier(config, defaults)
    """

    def __init__(self, defaults: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.defaults: Dict[str, Any] = defaults or {}
        self.state: Dict[str, Dict] = defaultdict(dict)
        self.groups: List[Dict[str, Any]] = []
        self.enable_mask_update = True

    def __getstate__(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return {'defaults': self.defaults, 'state': self.state, 'groups': self.groups}

    def __setstate__(self, state: Dict[str, Dict[str, Any]]) -> None:
        if False:
            print('Hello World!')
        self.__dict__.update(state)

    def __repr__(self):
        if False:
            return 10
        format_string = self.__class__.__name__ + ' ('
        for (i, sparse_args) in enumerate(self.groups):
            module = sparse_args['module']
            format_string += '\n'
            format_string += f'\tGroup {i}\n'
            format_string += f'\t    module: {module}\n'
            for key in sorted(sparse_args.keys()):
                if key == 'module':
                    continue
                format_string += f'\t    {key}: {sparse_args[key]}\n'
        format_string += ')'
        return format_string

    def state_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        'Returns the state of the optimizer as a :class:`dict`.\n\n        It contains:\n        * state - current state of the sparsification.\n        * groups - a list containing all sparsity configuration groups\n            with the key \'tensor_fqn\' specifying the path to the sparsified tensor within a model\n\n        TODO: Need a clean way of loading the state of the "prepared" module\n        '
        groups: List[Dict[str, Any]] = [dict(filter(lambda key_value: key_value[0] not in KEYS_NOT_IN_STATE_DICT, mg.items())) for mg in self.groups]
        return {'state': self.state, 'groups': groups}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool=True):
        if False:
            print('Hello World!')
        groups = copy.deepcopy(state_dict['groups'])
        states = state_dict['state']
        for (tensor_fqn, s) in states.items():
            arg_info = get_arg_info_from_tensor_fqn(self.model, tensor_fqn)
            module = arg_info['module']
            tensor_name = arg_info['tensor_name']
            if strict and module is None:
                raise RuntimeError(f'Error loading {tensor_fqn} into the model')
            found = False
            for p in module.parametrizations[tensor_name]:
                if isinstance(p, FakeSparsity):
                    found = True
                    break
            if not found:
                p = FakeSparsity(torch.ones(getattr(module, tensor_name).shape))
                parametrize.register_parametrization(module, tensor_name, p)
            if s.get('mask', None) is not None:
                mask = s.pop('mask')
                p.mask = mask
            for mg in groups:
                if mg['tensor_fqn'] == tensor_fqn:
                    mg.update(arg_info)
        self.__setstate__({'state': states, 'groups': groups})

    def make_config_from_model(self, model: nn.Module, SUPPORTED_MODULES: Set[Type]=SUPPORTED_MODULES) -> None:
        if False:
            print('Hello World!')
        self.config = []
        stack = [model]
        while stack:
            module = stack.pop()
            for (name, child) in module.named_children():
                if type(child) in SUPPORTED_MODULES:
                    module_fqn = module_to_fqn(model, child)
                    assert isinstance(module_fqn, str)
                    self.config.append({'tensor_fqn': module_fqn + '.weight'})
                else:
                    stack.append(child)

    def prepare(self, model, config):
        if False:
            for i in range(10):
                print('nop')
        'Prepares a model, by adding the parametrizations.\n\n        Note::\n\n            The model is modified inplace. If you need to preserve the original\n            model, use copy.deepcopy.\n        '
        self.model = model
        self.config = config
        if self.config is None:
            self.make_config_from_model(model)
        for module_config in self.config:
            assert isinstance(module_config, dict), 'config elements should be dicts not modules i.e.:[{`tensor_fqn`: `foo.bar.weight`}, {`tensor_fqn`: ... }, ...]'
            assert isinstance(self.defaults, Dict)
            local_args = copy.deepcopy(self.defaults)
            local_args.update(module_config)
            tensor_fqn = local_args.get('tensor_fqn', None)
            assert tensor_fqn is not None, 'tensor_fqn is a required argument in the sparsity config whichreplaces previous `module` and [module]`fqn` arguments'
            info_from_tensor_fqn = get_arg_info_from_tensor_fqn(model, tensor_fqn)
            for key in info_from_tensor_fqn.keys():
                if key in local_args:
                    assert info_from_tensor_fqn[key] == local_args[key] or (key == 'tensor_fqn' and '.' + info_from_tensor_fqn[key] == local_args[key]), f'Given both `{key}` and `tensor_fqn` in the config, it is expected them to agree!'
            local_args.update(info_from_tensor_fqn)
            self.groups.append(local_args)
        self._prepare()

    def _prepare(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Adds mask parametrization to the layer weight'
        for config in self.groups:
            module = config['module']
            tensor_name = config['tensor_name']
            parametrization = config.get('parametrization', FakeSparsity)
            mask = config.get('mask', torch.ones_like(getattr(module, tensor_name)))
            self.state[config['tensor_fqn']]['mask'] = mask
            parametrize.register_parametrization(module, tensor_name, parametrization(mask))

    def squash_mask(self, params_to_keep: Optional[Tuple[str, ...]]=None, params_to_keep_per_layer: Optional[Dict[str, Tuple[str, ...]]]=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Squashes the sparse masks into the appropriate tensors.\n\n        If either the `params_to_keep` or `params_to_keep_per_layer` is set,\n        the module will have a `sparse_params` dict attached to it.\n\n        Args:\n            params_to_keep: List of keys to save in the module or a dict\n                            representing the modules and keys that will have\n                            sparsity parameters saved\n            params_to_keep_per_layer: Dict to specify the params that should be\n                            saved for specific layers. The keys in the dict\n                            should be the module fqn, while the values should\n                            be a list of strings with the names of the variables\n                            to save in the `sparse_params`\n\n        Examples:\n            >>> # xdoctest: +SKIP("locals are undefined")\n            >>> # Don\'t save any sparse params\n            >>> sparsifier.squash_mask()\n            >>> hasattr(model.submodule1, \'sparse_params\')\n            False\n\n            >>> # Keep sparse params per layer\n            >>> sparsifier.squash_mask(\n            ...     params_to_keep_per_layer={\n            ...         \'submodule1.linear1\': (\'foo\', \'bar\'),\n            ...         \'submodule2.linear42\': (\'baz\',)\n            ...     })\n            >>> print(model.submodule1.linear1.sparse_params)\n            {\'foo\': 42, \'bar\': 24}\n            >>> print(model.submodule2.linear42.sparse_params)\n            {\'baz\': 0.1}\n\n            >>> # Keep sparse params for all layers\n            >>> sparsifier.squash_mask(params_to_keep=(\'foo\', \'bar\'))\n            >>> print(model.submodule1.linear1.sparse_params)\n            {\'foo\': 42, \'bar\': 24}\n            >>> print(model.submodule2.linear42.sparse_params)\n            {\'foo\': 42, \'bar\': 24}\n\n            >>> # Keep some sparse params for all layers, and specific ones for\n            >>> # some other layers\n            >>> sparsifier.squash_mask(\n            ...     params_to_keep=(\'foo\', \'bar\'),\n            ...     params_to_keep_per_layer={\n            ...         \'submodule2.linear42\': (\'baz\',)\n            ...     })\n            >>> print(model.submodule1.linear1.sparse_params)\n            {\'foo\': 42, \'bar\': 24}\n            >>> print(model.submodule2.linear42.sparse_params)\n            {\'foo\': 42, \'bar\': 24, \'baz\': 0.1}\n        '
        for config in self.groups:
            module = config['module']
            tensor_name = config['tensor_name']
            parametrize.remove_parametrizations(module, tensor_name, leave_parametrized=True)
            sparse_params = {}
            if params_to_keep is not None:
                global_params = {k: config[k] for k in params_to_keep}
                sparse_params.update(global_params)
            if params_to_keep_per_layer is not None:
                params = params_to_keep_per_layer.get(config['module_fqn'], None)
                if params is not None:
                    per_layer_params = {k: config[k] for k in params}
                    sparse_params.update(per_layer_params)
            if sparse_params:
                module.sparse_params = sparse_params

    def convert(self, module: nn.Module, mapping: Optional[Dict[Type[nn.Module], Type[nn.Module]]]=None, inplace: bool=False, parameterization: Type[nn.Module]=FakeSparsity):
        if False:
            return 10
        'Converts submodules in input module to a different module according to `mapping`\n        by calling `from_dense` method on the target module class\n        Args:\n            module: input module\n            mapping: a dictionary that maps from source module type to target\n                module type, can be overwritten to allow swapping user defined\n                Modules\n            inplace: carry out model transformations in-place, the original module\n                is mutated\n        '
        if mapping is None:
            raise NotImplementedError('Need to auto generate mapping ')
        if not inplace:
            module = copy.deepcopy(module)
        reassign = {}
        for (name, mod) in module.named_children():
            if module_contains_param(mod, parameterization) and type_before_parametrizations(mod) in mapping:
                reassign[name] = swap_module(mod, mapping)
            else:
                reassign[name] = self.convert(mod, mapping=mapping, inplace=True, parameterization=parameterization)
        for (key, value) in reassign.items():
            module._modules[key] = value
        return module

    def step(self, use_path: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for config in self.groups:
                self.update_mask(**config)

    @abc.abstractmethod
    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs):
        if False:
            while True:
                i = 10
        pass
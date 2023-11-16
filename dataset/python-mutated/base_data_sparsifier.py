import abc
import torch
from typing import Optional, Tuple, List, Any, Dict
from ...sparsifier import base_sparsifier
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier import utils
from torch.nn.utils import parametrize
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('once')
__all__ = ['BaseDataSparsifier']
EMBEDDING_TYPES = {nn.Embedding, nn.EmbeddingBag}
SUPPORTED_TYPES = {torch.Tensor, nn.Parameter, *EMBEDDING_TYPES}

class _Container(nn.Module):
    pass

class BaseDataSparsifier(base_sparsifier.BaseSparsifier):
    """
    Base Data Sparsifier class for all Data sparsifiers.
    The abstract class accepts raw torch tensors / embedding / embedding bags (refer to SUPPORTED_TYPES above)
    to prepare for sparsification.
    In this case, mask (and parametrizations) is owned by the class and not by the user.
    Specifically, the container object inside the class maintains the mask and parametrizations of the input data

    Args:
        data_list (list of tuples)
            list of (name, data) tuples to sparsify. Lookup SUPPORTED_TYPES
            for type of data. Internally, a container module handles the data sparsification.

        defaults (dict)
            default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    Example::
        >>> # xdoctest: +SKIP
        >>> data_list = [('tensor_1', torch.randn(3,3)), ('tensor_2', torch.randn(4,4))]
        >>> defaults = {'sparsity_level': 0.7}
        >>> sparsifier = DerivedDataSparsifier(data_list = data_list, **defaults) # Some sparsifier that inherits BaseDataSparsifier
        >>> new_tensor_to_add = {'name': 'tensor_3', 'data': torch.randn(5,5), 'sparsity_level': 0.3}
        >>> sparsifier.add_data(**new_tensor_to_add)
        >>> # tensor_1 and tensor_2 will have sparsity_level of 0.7 but tensor_3 will have sparsity_level=0.3
    """

    def __init__(self, data_list: Optional[List[Tuple[str, Any]]]=None, **defaults):
        if False:
            print('Hello World!')
        super().__init__(defaults=defaults)
        self._container = _Container()
        self.data_groups: Dict[str, Dict] = defaultdict(dict)
        if data_list is not None:
            [self.add_data(name, data, **self.defaults) for (name, data) in data_list]

    def prepare(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('this function is undefined for this class')

    def _extract_weight(self, data):
        if False:
            for i in range(10):
                print('nop')
        if type(data) in [torch.Tensor, nn.Parameter]:
            return data
        elif type(data) in EMBEDDING_TYPES:
            return data.weight

    def add_data(self, name: str, data, reuse_mask=True, **config):
        if False:
            for i in range(10):
                print('nop')
        ' Configures and parametrizes the internal container model with name and data.\n\n        **Note**:\n            1. If the data with name already exists, it replaces the data.\n            2. While replacing, the old mask is reused when `reuse_mask=True`\n            3. If `reuse_mask=True`, then the replacing data needs to have the same shape as that of old data.\n            4. By default, the config of the replaced data is used as config for the replacing data, unless something\n               is specified in the config dictionary.\n        '
        assert type(data) in SUPPORTED_TYPES, 'specified data type not supported at the moment'
        local_args = copy.deepcopy(self.defaults)
        local_args.update(config)
        weight = self._extract_weight(data)
        mask = local_args.get('mask', torch.ones_like(weight))
        param_class = local_args.get('parametrization', utils.FakeSparsity)
        if name in self.state:
            warnings.warn('Replacing existing data of the same name. - Did you mean a different name?')
            old_args = self.data_groups[name]
            local_args = copy.deepcopy(old_args)
            local_args.update(config)
            if reuse_mask:
                current_data = self.get_data(name=name)
                assert weight.shape == current_data.shape, 'to retain the old mask, the shape of the new data must be the same as the previous one'
                mask = self.get_mask(name=name)
            self._delete_data(name=name)
        self._container.register_buffer(name=name, tensor=weight)
        parametrize.register_parametrization(self._container, name, param_class(mask))
        self.state[name]['mask'] = mask
        self.data_groups[name] = local_args
        return getattr(self._container, name)

    def get_data(self, name: str, return_original: bool=True):
        if False:
            for i in range(10):
                print('nop')
        'Returns weight tensor (or data)\n        Args:\n            - name: name of the data to be returned\n            - return_original returns weight tensor without applying parametrization if True\n                else - returns the sparsified version (parametrized)\n        '
        if name not in self.data_groups:
            raise ValueError('data with specified name does not exist')
        if return_original:
            if not parametrize.is_parametrized(self._container, name):
                raise ValueError('mask squashed - original mask value does not exist')
            data = getattr(self._container.parametrizations, name).original
            return data
        else:
            return getattr(self._container, name)

    def _convert_mask(self, states, sparse_coo=True):
        if False:
            i = 10
            return i + 15
        'Converts the mask to sparse coo or dense tensors depending on the `sparse_coo` argument.\n        '
        states = copy.deepcopy(states)
        for state in states.values():
            if sparse_coo:
                state['mask'] = state['mask'].to_sparse_coo()
            else:
                state['mask'] = state['mask'].to_dense()
        return states

    def state_dict(self):
        if False:
            i = 10
            return i + 15
        'Returns the state of the optimizer as a :class:`dict`.\n\n        It contains:\n        * state - contains name -> mask mapping.\n        * data_groups - a list containing all sparsity configuration groups\n            with the key name specifying the name of the data\n        * container_state_dict - the state dictionary of the internal\n            container model used for sparsification\n        '
        state = self._convert_mask(self.state)
        return {'state': state, 'data_groups': self.data_groups, '_container': self._container.state_dict()}

    def _load_container_from_state(self, states, data_groups, container_state_dict):
        if False:
            i = 10
            return i + 15
        'This restores the state of the container specifically based on the data present in state and data_groups\n        If the data was parametrized, then the data would be added to the container and then parametrized,\n        else it would just add the attribute the container.\n        '
        for (name, state) in states.items():
            config_name = data_groups.get(name, None)
            if config_name is None:
                raise RuntimeError(f'Error loading {name}')
            parametrized_name = f'parametrizations.{name}.original'
            parametrized = False
            data = container_state_dict.get(name, None)
            if name in container_state_dict:
                data = container_state_dict.get(name)
            elif parametrized_name in container_state_dict:
                data = container_state_dict.get(parametrized_name)
                parametrized = True
            else:
                raise RuntimeError(f'Error loading {name}')
            self._container.register_buffer(name=name, tensor=data)
            if parametrized:
                mask = state.get('mask', torch.ones_like(data))
                param_class = data_groups.get('parametrization', utils.FakeSparsity)
                parametrize.register_parametrization(self._container, name, param_class(mask))

    def load_state_dict(self, state_dict, strict=True):
        if False:
            for i in range(10):
                print('nop')
        'The load_state_dict() restores the state of the sparsifier based on the state_dict\n\n        Args:\n        * state_dict - the dictionary that to which the current sparsifier needs to be restored to\n        * strict - If True - the sparsifier is reset and is restored exactly to the state in state_dict.\n            If False - the current sparsifier is not reset before loading the state_dict i.e. data added\n            before loading the state_dict is not erased.\n        '
        states = copy.deepcopy(state_dict['state'])
        data_groups = copy.deepcopy(state_dict['data_groups'])
        container_state_dict = copy.deepcopy(state_dict['_container'])
        states = self._convert_mask(states, sparse_coo=False)
        if strict:
            self._container = _Container()
        self._load_container_from_state(states, data_groups, container_state_dict)
        if not strict:
            states.update(self.state)
            data_groups.update(self.data_groups)
        self.__setstate__({'state': states, 'data_groups': data_groups})

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        if '_container' in state:
            container_dict = state.pop('_container')
            self._container = _Container()
            state['state'] = self._convert_mask(state['state'], sparse_coo=False)
            self._load_container_from_state(state['state'], state['data_groups'], container_dict)
        self.__dict__.update(state)

    def __getstate__(self):
        if False:
            return 10
        state = self._convert_mask(self.state)
        return {'defaults': self.defaults, 'state': state, 'data_groups': self.data_groups, '_container': self._container.state_dict()}

    def __repr__(self):
        if False:
            print('Hello World!')
        format_string = self.__class__.__name__ + ' ('
        for (name, sparse_args) in self.data_groups.items():
            format_string += '\n'
            format_string += '\tData Group\n'
            format_string += f'\t    name: {name}\n'
            for key in sorted(sparse_args.keys()):
                if key == 'data':
                    continue
                format_string += f'\t    {key}: {sparse_args[key]}\n'
        format_string += ')'
        return format_string

    def get_mask(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        if name not in self.state:
            raise ValueError('data with specified name does not exist')
        return self.state[name]['mask']

    def squash_mask(self, *args, leave_parametrized=True, names=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Squashes the sparse masks into the appropriate tensors. Also, accepts list of strings\n        to squash mask for. If none, squashes mask for all the keys\n        kwargs:\n            * names: list of strings to squash mask for\n            * sparsified: if true - applies the mask before squashing\n                          if false - does not apply the mask before squashing\n        '
        if names is None:
            names = list(self.data_groups.keys())
        for name in names:
            parametrize.remove_parametrizations(self._container, name, leave_parametrized=leave_parametrized)

    def step(self):
        if False:
            return 10
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for (name, config) in self.data_groups.items():
                data = self.get_data(name)
                self.update_mask(name, data, **config)

    @abc.abstractmethod
    def update_mask(self, name, data, **kwargs):
        if False:
            return 10
        pass

    def _delete_data(self, name):
        if False:
            i = 10
            return i + 15
        'Detaches some data from the sparsifier.\n\n        Args:\n            name (str)\n                Name of the data to be removed from the sparsifier\n\n        Note:\n            Currently private. Kind of used as a helper function when replacing data of the same name\n        '
        self.squash_mask(names=[name], leave_parametrized=False)
        delattr(self._container, name)
        self.state.pop(name)
        self.data_groups.pop(name)
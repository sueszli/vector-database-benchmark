from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
__all__ = ['ActivationSparsifier']

class ActivationSparsifier:
    """
    The Activation sparsifier class aims to sparsify/prune activations in a neural
    network. The idea is to attach the sparsifier to a layer (or layers) and it
    zeroes out the activations based on the mask_fn (or sparsification function)
    input by the user.
    The mask_fn is applied once all the inputs are aggregated and reduced i.e.
    mask = mask_fn(reduce_fn(aggregate_fn(activations)))

    Note::
        The sparsification mask is computed on the input **before it goes through the attached layer**.

    Args:
        model (nn.Module):
            The model whose layers will be sparsified. The layers that needs to be
            sparsified should be added separately using the register_layer() function
        aggregate_fn (Optional, Callable):
            default aggregate_fn that is used if not specified while registering the layer.
            specifies how inputs should be aggregated over time.
            The aggregate_fn should usually take 2 torch tensors and return the aggregated tensor.
            Example
                def add_agg_fn(tensor1, tensor2):  return tensor1 + tensor2
                reduce_fn (Optional, Callable):
                    default reduce_fn that is used if not specified while registering the layer.
                    reduce_fn will be called on the aggregated tensor i.e. the tensor obtained after
                    calling agg_fn() on all inputs.
                    Example
                def mean_reduce_fn(agg_tensor):    return agg_tensor.mean(dim=0)
                mask_fn (Optional, Callable):
                    default mask_fn that is used to create the sparsification mask using the tensor obtained after
                    calling the reduce_fn(). This is used by default if a custom one is passed in the
                    register_layer().
                    Note that the mask_fn() definition should contain the sparse arguments that is passed in sparse_config
                    arguments.
                features (Optional, list):
                    default selected features to sparsify.
                    If this is non-empty, then the mask_fn will be applied for each feature of the input.
                    For example,
                mask = [mask_fn(reduce_fn(aggregated_fn(input[feature])) for feature in features]
                feature_dim (Optional, int):
                    default dimension of input features. Again, features along this dim will be chosen
                    for sparsification.
                sparse_config (Dict):
                    Default configuration for the mask_fn. This config will be passed
                    with the mask_fn()

    Example:
        >>> # xdoctest: +SKIP
        >>> model = SomeModel()
        >>> act_sparsifier = ActivationSparsifier(...)  # init activation sparsifier
        >>> # Initialize aggregate_fn
        >>> def agg_fn(x, y):
        >>>     return x + y
        >>>
        >>> # Initialize reduce_fn
        >>> def reduce_fn(x):
        >>>     return torch.mean(x, dim=0)
        >>>
        >>> # Initialize mask_fn
        >>> def mask_fn(data):
        >>>     return torch.eye(data.shape).to(data.device)
        >>>
        >>>
        >>> act_sparsifier.register_layer(model.some_layer, aggregate_fn=agg_fn, reduce_fn=reduce_fn, mask_fn=mask_fn)
        >>>
        >>> # start training process
        >>> for _ in [...]:
        >>>     # epoch starts
        >>>         # model.forward(), compute_loss() and model.backwards()
        >>>     # epoch ends
        >>>     act_sparsifier.step()
        >>> # end training process
        >>> sparsifier.squash_mask()
    """

    def __init__(self, model: nn.Module, aggregate_fn=None, reduce_fn=None, mask_fn=None, features=None, feature_dim=None, **sparse_config):
        if False:
            return 10
        self.model = model
        self.defaults: Dict[str, Any] = defaultdict()
        self.defaults['sparse_config'] = sparse_config
        self.defaults['aggregate_fn'] = aggregate_fn
        self.defaults['reduce_fn'] = reduce_fn
        self.defaults['mask_fn'] = mask_fn
        self.defaults['features'] = features
        self.defaults['feature_dim'] = feature_dim
        self.data_groups: Dict[str, Dict] = defaultdict(dict)
        self.state: Dict[str, Any] = defaultdict(dict)

    @staticmethod
    def _safe_rail_checks(args):
        if False:
            while True:
                i = 10
        'Makes sure that some of the functions and attributes are not passed incorrectly\n        '
        (features, feature_dim) = (args['features'], args['feature_dim'])
        if features is not None:
            assert feature_dim is not None, 'need feature dim to select features'
        fn_keys = ['aggregate_fn', 'reduce_fn', 'mask_fn']
        for key in fn_keys:
            fn = args[key]
            assert callable(fn), 'function should be callable'

    def _aggregate_hook(self, name):
        if False:
            return 10
        'Returns hook that computes aggregate of activations passing through.\n        '
        feature_dim = self.data_groups[name]['feature_dim']
        features = self.data_groups[name]['features']
        agg_fn = self.data_groups[name]['aggregate_fn']

        def hook(module, input) -> None:
            if False:
                for i in range(10):
                    print('nop')
            input_data = input[0]
            data = self.data_groups[name].get('data')
            if features is None:
                if data is None:
                    data = torch.zeros_like(input_data)
                    self.state[name]['mask'] = torch.ones_like(input_data)
                out_data = agg_fn(data, input_data)
            else:
                if data is None:
                    out_data = [0 for _ in range(0, len(features))]
                    self.state[name]['mask'] = [0 for _ in range(0, len(features))]
                else:
                    out_data = data
                for feature_idx in range(len(features)):
                    feature_tensor = torch.Tensor([features[feature_idx]]).long().to(input_data.device)
                    data_feature = torch.index_select(input_data, feature_dim, feature_tensor)
                    if data is None:
                        curr_data = torch.zeros_like(data_feature)
                        self.state[name]['mask'][feature_idx] = torch.ones_like(data_feature)
                    else:
                        curr_data = data[feature_idx]
                    out_data[feature_idx] = agg_fn(curr_data, data_feature)
            self.data_groups[name]['data'] = out_data
        return hook

    def register_layer(self, layer: nn.Module, aggregate_fn=None, reduce_fn=None, mask_fn=None, features=None, feature_dim=None, **sparse_config):
        if False:
            i = 10
            return i + 15
        '\n        Registers a layer for sparsification. The layer should be part of self.model.\n        Specifically, registers a pre-forward hook to the layer. The hook will apply the aggregate_fn\n        and store the aggregated activations that is input over each step.\n\n        Note::\n            - There is no need to pass in the name of the layer as it is automatically computed as per\n              the fqn convention.\n\n            - All the functions (fn) passed as argument will be called at a dim, feature level.\n        '
        name = module_to_fqn(self.model, layer)
        assert name is not None, 'layer not found in the model'
        if name in self.data_groups:
            warnings.warn('layer already attached to the sparsifier, deregistering the layer and registering with new config')
            self.unregister_layer(name=name)
        local_args = copy.deepcopy(self.defaults)
        update_dict = {'aggregate_fn': aggregate_fn, 'reduce_fn': reduce_fn, 'mask_fn': mask_fn, 'features': features, 'feature_dim': feature_dim, 'layer': layer}
        local_args.update(((arg, val) for (arg, val) in update_dict.items() if val is not None))
        local_args['sparse_config'].update(sparse_config)
        self._safe_rail_checks(local_args)
        self.data_groups[name] = local_args
        agg_hook = layer.register_forward_pre_hook(self._aggregate_hook(name=name))
        self.state[name]['mask'] = None
        self.data_groups[name]['hook'] = agg_hook
        self.data_groups[name]['hook_state'] = 'aggregate'

    def get_mask(self, name: Optional[str]=None, layer: Optional[nn.Module]=None):
        if False:
            while True:
                i = 10
        '\n        Returns mask associated to the layer.\n\n        The mask is\n            - a torch tensor is features for that layer is None.\n            - a list of torch tensors for each feature, otherwise\n\n        Note::\n            The shape of the mask is unknown until model.forward() is applied.\n            Hence, if get_mask() is called before model.forward(), an\n            error will be raised.\n        '
        assert name is not None or layer is not None, 'Need at least name or layer obj to retrieve mask'
        if name is None:
            assert layer is not None
            name = module_to_fqn(self.model, layer)
            assert name is not None, 'layer not found in the specified model'
        if name not in self.state:
            raise ValueError('Error: layer with the given name not found')
        mask = self.state[name].get('mask', None)
        if mask is None:
            raise ValueError('Error: shape unknown, call layer() routine at least once to infer mask')
        return mask

    def unregister_layer(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Detaches the sparsifier from the layer\n        '
        self.data_groups[name]['hook'].remove()
        self.state.pop(name)
        self.data_groups.pop(name)

    def step(self):
        if False:
            print('Hello World!')
        'Internally calls the update_mask() function for each layer\n        '
        with torch.no_grad():
            for (name, configs) in self.data_groups.items():
                data = configs['data']
                self.update_mask(name, data, configs)
                self.data_groups[name].pop('data')

    def update_mask(self, name, data, configs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called for each registered layer and does the following-\n            1. apply reduce_fn on the aggregated activations\n            2. use mask_fn to compute the sparsification mask\n\n        Note:\n            the reduce_fn and mask_fn is called for each feature, dim over the data\n        '
        mask = self.get_mask(name)
        sparse_config = configs['sparse_config']
        features = configs['features']
        reduce_fn = configs['reduce_fn']
        mask_fn = configs['mask_fn']
        if features is None:
            data = reduce_fn(data)
            mask.data = mask_fn(data, **sparse_config)
        else:
            for feature_idx in range(len(features)):
                data_feature = reduce_fn(data[feature_idx])
                mask[feature_idx].data = mask_fn(data_feature, **sparse_config)

    def _sparsify_hook(self, name):
        if False:
            return 10
        'Returns hook that applies sparsification mask to input entering the attached layer\n        '
        mask = self.get_mask(name)
        features = self.data_groups[name]['features']
        feature_dim = self.data_groups[name]['feature_dim']

        def hook(module, input):
            if False:
                print('Hello World!')
            input_data = input[0]
            if features is None:
                return input_data * mask
            else:
                for feature_idx in range(0, len(features)):
                    feature = torch.Tensor([features[feature_idx]]).long().to(input_data.device)
                    sparsified = torch.index_select(input_data, feature_dim, feature) * mask[feature_idx]
                    input_data.index_copy_(feature_dim, feature, sparsified)
                return input_data
        return hook

    def squash_mask(self, attach_sparsify_hook=True, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Unregisters aggregate hook that was applied earlier and registers sparsification hooks if\n        attach_sparsify_hook = True.\n        '
        for (name, configs) in self.data_groups.items():
            configs['hook'].remove()
            configs.pop('hook')
            self.data_groups[name]['hook_state'] = 'None'
            if attach_sparsify_hook:
                configs['hook'] = configs['layer'].register_forward_pre_hook(self._sparsify_hook(name))
            configs['hook_state'] = 'sparsify'

    def _get_serializable_data_groups(self):
        if False:
            print('Hello World!')
        'Exclude hook and layer from the config keys before serializing\n\n        TODO: Might have to treat functions (reduce_fn, mask_fn etc) in a different manner while serializing.\n              For time-being, functions are treated the same way as other attributes\n        '
        data_groups: Dict[str, Any] = defaultdict()
        for (name, config) in self.data_groups.items():
            new_config = {key: value for (key, value) in config.items() if key not in ['hook', 'layer']}
            data_groups[name] = new_config
        return data_groups

    def _convert_mask(self, states_dict, sparse_coo=True):
        if False:
            i = 10
            return i + 15
        'Converts the mask to sparse coo or dense depending on the `sparse_coo` argument.\n        If `sparse_coo=True`, then the mask is stored as sparse coo else dense tensor\n        '
        states = copy.deepcopy(states_dict)
        for state in states.values():
            if state['mask'] is not None:
                if isinstance(state['mask'], List):
                    for idx in range(len(state['mask'])):
                        if sparse_coo:
                            state['mask'][idx] = state['mask'][idx].to_sparse_coo()
                        else:
                            state['mask'][idx] = state['mask'][idx].to_dense()
                elif sparse_coo:
                    state['mask'] = state['mask'].to_sparse_coo()
                else:
                    state['mask'] = state['mask'].to_dense()
        return states

    def state_dict(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the state of the sparsifier as a :class:`dict`.\n\n        It contains:\n        * state - contains name -> mask mapping.\n        * data_groups - a dictionary containing all config information for each\n            layer\n        * defaults - the default config while creating the constructor\n        '
        data_groups = self._get_serializable_data_groups()
        state = self._convert_mask(self.state)
        return {'state': state, 'data_groups': data_groups, 'defaults': self.defaults}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if False:
            return 10
        'The load_state_dict() restores the state of the sparsifier based on the state_dict\n\n        Args:\n        * state_dict - the dictionary that to which the current sparsifier needs to be restored to\n        '
        state = state_dict['state']
        (data_groups, defaults) = (state_dict['data_groups'], state_dict['defaults'])
        self.__set_state__({'state': state, 'data_groups': data_groups, 'defaults': defaults})

    def __get_state__(self) -> Dict[str, Any]:
        if False:
            return 10
        data_groups = self._get_serializable_data_groups()
        state = self._convert_mask(self.state)
        return {'defaults': self.defaults, 'state': state, 'data_groups': data_groups}

    def __set_state__(self, state: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        state['state'] = self._convert_mask(state['state'], sparse_coo=False)
        self.__dict__.update(state)
        for (name, config) in self.data_groups.items():
            layer = fqn_to_module(self.model, name)
            assert layer is not None
            if 'hook_state' in config and config['hook_state'] == 'aggregate':
                hook = layer.register_forward_pre_hook(self._aggregate_hook(name))
            elif 'hook_state' in config and config['hook_state'] == 'sparsify':
                hook = layer.register_forward_pre_hook(self._sparsify_hook(name))
            config['layer'] = layer
            config['hook'] = hook

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        format_string = self.__class__.__name__ + ' ('
        for (name, config) in self.data_groups.items():
            format_string += '\n'
            format_string += '\tData Group\n'
            format_string += f'\t    name: {name}\n'
            for key in sorted(config.keys()):
                if key in ['data', 'hook', 'reduce_fn', 'mask_fn', 'aggregate_fn']:
                    continue
                format_string += f'\t    {key}: {config[key]}\n'
        format_string += ')'
        return format_string
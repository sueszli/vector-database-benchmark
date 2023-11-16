from typing import Any, Tuple, Callable, Optional, List, Dict, Union
from abc import ABC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal
from ding.torch_utils import get_tensor_data, zeros_like
from ding.rl_utils import create_noise_generator
from ding.utils.data import default_collate

class IModelWrapper(ABC):
    """
    Overview:
        The basic interface class of model wrappers. Model wrapper is a wrapper class of torch.nn.Module model, which         is used to add some extra operations for the wrapped model, such as hidden state maintain for RNN-base model,         argmax action selection for discrete action space, etc.
    Interfaces:
        ``__init__``, ``__getattr__``, ``info``, ``reset``, ``forward``.
    """

    def __init__(self, model: nn.Module) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Initialize model and other necessary member variabls in the model wrapper.\n        '
        self._model = model

    def __getattr__(self, key: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Get original attrbutes of torch.nn.Module model, such as variables and methods defined in model.\n        Arguments:\n            - key (:obj:`str`): The string key to query.\n        Returns:\n            - ret (:obj:`Any`): The queried attribute.\n        '
        return getattr(self._model, key)

    def info(self, attr_name: str) -> str:
        if False:
            return 10
        '\n        Overview:\n            Get some string information of the indicated ``attr_name``, which is used for debug wrappers.\n            This method will recursively search for the indicated ``attr_name``.\n        Arguments:\n            - attr_name (:obj:`str`): The string key to query information.\n        Returns:\n            - info_string (:obj:`str`): The information string of the indicated ``attr_name``.\n        '
        if attr_name in dir(self):
            if isinstance(self._model, IModelWrapper):
                return '{} {}'.format(self.__class__.__name__, self._model.info(attr_name))
            elif attr_name in dir(self._model):
                return '{} {}'.format(self.__class__.__name__, self._model.__class__.__name__)
            else:
                return '{}'.format(self.__class__.__name__)
        elif isinstance(self._model, IModelWrapper):
            return '{}'.format(self._model.info(attr_name))
        else:
            return '{}'.format(self._model.__class__.__name__)

    def reset(self, data_id: List[int]=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview\n            Basic interface, reset some stateful varaibles in the model wrapper, such as hidden state of RNN.\n            Here we do nothing and just implement this interface method.\n            Other derived model wrappers can override this method to add some extra operations.\n        Arguments:\n            - data_id (:obj:`List[int]`): The data id list to reset. If None, reset all data. In practice,                 model wrappers often needs to maintain some stateful variables for each data trajectory,                 so we leave this ``data_id`` argument to reset the stateful variables of the indicated data.\n        '
        pass

    def forward(self, *args, **kwargs) -> Any:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Basic interface, call the wrapped model's forward method. Other derived model wrappers can override this             method to add some extra operations.\n        "
        return self._model.forward(*args, **kwargs)

class BaseModelWrapper(IModelWrapper):
    """
    Overview:
        Placeholder class for the model wrapper. This class is used to wrap the model without any extra operations,         including a empty ``reset`` method and a ``forward`` method which directly call the wrapped model's forward.
        To keep the consistency of the model wrapper interface, we use this class to wrap the model without specific         operations in the implementation of DI-engine's policy.
    """
    pass

class HiddenStateWrapper(IModelWrapper):
    """
    Overview:
        Maintain the hidden state for RNN-base model. Each sample in a batch has its own state.
    Interfaces:
        ``__init__``, ``reset``, ``forward``.
    """

    def __init__(self, model: Any, state_num: int, save_prev_state: bool=False, init_fn: Callable=lambda : None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Maintain the hidden state for RNN-base model. Each sample in a batch has its own state.             Init the maintain state and state function; Then wrap the ``model.forward`` method with auto             saved data ['prev_state'] input, and create the ``model.reset`` method.\n        Arguments:\n            - model(:obj:`Any`): Wrapped model class, should contain forward method.\n            - state_num (:obj:`int`): Number of states to process.\n            - save_prev_state (:obj:`bool`): Whether to output the prev state in output.\n            - init_fn (:obj:`Callable`): The function which is used to init every hidden state when init and reset,                 default return None for hidden states.\n\n        .. note::\n            1. This helper must deal with an actual batch with some parts of samples, e.g: 6 samples of state_num 8.\n            2. This helper must deal with the single sample state reset.\n        "
        super().__init__(model)
        self._state_num = state_num
        self._state = {i: init_fn() for i in range(state_num)}
        self._save_prev_state = save_prev_state
        self._init_fn = init_fn

    def forward(self, data, **kwargs):
        if False:
            print('Hello World!')
        state_id = kwargs.pop('data_id', None)
        valid_id = kwargs.pop('valid_id', None)
        (data, state_info) = self.before_forward(data, state_id)
        output = self._model.forward(data, **kwargs)
        h = output.pop('next_state', None)
        if h is not None:
            self.after_forward(h, state_info, valid_id)
        if self._save_prev_state:
            prev_state = get_tensor_data(data['prev_state'])
            for i in range(len(prev_state)):
                if prev_state[i] is None:
                    prev_state[i] = zeros_like(h[0])
            output['prev_state'] = prev_state
        return output

    def reset(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        state = kwargs.pop('state', None)
        state_id = kwargs.get('data_id', None)
        self.reset_state(state, state_id)
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def reset_state(self, state: Optional[list]=None, state_id: Optional[list]=None) -> None:
        if False:
            print('Hello World!')
        if state_id is None:
            state_id = [i for i in range(self._state_num)]
        if state is None:
            state = [self._init_fn() for i in range(len(state_id))]
        assert len(state) == len(state_id), '{}/{}'.format(len(state), len(state_id))
        for (idx, s) in zip(state_id, state):
            self._state[idx] = s

    def before_forward(self, data: dict, state_id: Optional[list]) -> Tuple[dict, dict]:
        if False:
            while True:
                i = 10
        if state_id is None:
            state_id = [i for i in range(self._state_num)]
        state_info = {idx: self._state[idx] for idx in state_id}
        data['prev_state'] = list(state_info.values())
        return (data, state_info)

    def after_forward(self, h: Any, state_info: dict, valid_id: Optional[list]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert len(h) == len(state_info), '{}/{}'.format(len(h), len(state_info))
        for (i, idx) in enumerate(state_info.keys()):
            if valid_id is None:
                self._state[idx] = h[i]
            elif idx in valid_id:
                self._state[idx] = h[i]

class TransformerInputWrapper(IModelWrapper):

    def __init__(self, model: Any, seq_len: int, init_fn: Callable=lambda : None) -> None:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Given N the length of the sequences received by a Transformer model, maintain the last N-1 input\n            observations. In this way we can provide at each step all the observations needed by Transformer to\n            compute its output. We need this because some methods such as 'collect' and 'evaluate' only provide the\n            model 1 observation per step and don't have memory of past observations, but Transformer needs a sequence\n            of N observations. The wrapper method ``forward`` will save the input observation in a FIFO memory of\n            length N and the method ``reset`` will reset the memory. The empty memory spaces will be initialized\n            with 'init_fn' or zero by calling the method ``reset_input``. Since different env can terminate at\n            different steps, the method ``reset_memory_entry`` only initializes the memory of specific environments in\n            the batch size.\n        Arguments:\n            - model (:obj:`Any`): Wrapped model class, should contain forward method.\n            - seq_len (:obj:`int`): Number of past observations to remember.\n            - init_fn (:obj:`Callable`): The function which is used to init every memory locations when init and reset.\n        "
        super().__init__(model)
        self.seq_len = seq_len
        self._init_fn = init_fn
        self.obs_memory = None
        self.init_obs = None
        self.bs = None
        self.memory_idx = []

    def forward(self, input_obs: torch.Tensor, only_last_logit: bool=True, data_id: List=None, **kwargs) -> Dict[str, torch.Tensor]:
        if False:
            print('Hello World!')
        "\n        Arguments:\n            - input_obs (:obj:`torch.Tensor`): Input observation without sequence shape: ``(bs, *obs_shape)``.\n            - only_last_logit (:obj:`bool`): if True 'logit' only contains the output corresponding to the current                 observation (shape: bs, embedding_dim), otherwise logit has shape (seq_len, bs, embedding_dim).\n            - data_id (:obj:`List`): id of the envs that are currently running. Memory update and logits return has                 only effect for those environments. If `None` it is considered that all envs are running.\n        Returns:\n            - Dictionary containing the input_sequence 'input_seq' stored in memory and the transformer output 'logit'.\n        "
        if self.obs_memory is None:
            self.reset_input(torch.zeros_like(input_obs))
        if data_id is None:
            data_id = list(range(self.bs))
        assert self.obs_memory.shape[0] == self.seq_len
        for (i, b) in enumerate(data_id):
            if self.memory_idx[b] == self.seq_len:
                self.obs_memory[:, b] = torch.roll(self.obs_memory[:, b], -1, 0)
                self.obs_memory[self.memory_idx[b] - 1, b] = input_obs[i]
            if self.memory_idx[b] < self.seq_len:
                self.obs_memory[self.memory_idx[b], b] = input_obs[i]
                if self.memory_idx != self.seq_len:
                    self.memory_idx[b] += 1
        out = self._model.forward(self.obs_memory, **kwargs)
        out['input_seq'] = self.obs_memory
        if only_last_logit:
            out['logit'] = [out['logit'][self.memory_idx[b] - 1][b] for b in range(self.bs) if b in data_id]
            out['logit'] = default_collate(out['logit'])
        return out

    def reset_input(self, input_obs: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Initialize the whole memory\n        '
        init_obs = torch.zeros_like(input_obs)
        self.init_obs = init_obs
        self.obs_memory = []
        for i in range(self.seq_len):
            self.obs_memory.append(init_obs.clone() if init_obs is not None else self._init_fn())
        self.obs_memory = default_collate(self.obs_memory)
        self.bs = self.init_obs.shape[0]
        self.memory_idx = [0 for _ in range(self.bs)]

    def reset(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        state_id = kwargs.get('data_id', None)
        input_obs = kwargs.get('input_obs', None)
        if input_obs is not None:
            self.reset_input(input_obs)
        if state_id is not None:
            self.reset_memory_entry(state_id)
        if input_obs is None and state_id is None:
            self.obs_memory = None
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def reset_memory_entry(self, state_id: Optional[list]=None) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Reset specific batch of the memory, batch ids are specified in 'state_id'\n        "
        assert self.init_obs is not None, 'Call method "reset_memory" first'
        for _id in state_id:
            self.memory_idx[_id] = 0
            self.obs_memory[:, _id] = self.init_obs[_id]

class TransformerSegmentWrapper(IModelWrapper):

    def __init__(self, model: Any, seq_len: int) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Given T the length of a trajectory and N the length of the sequences received by a Transformer model,\n            split T in sequences of N elements and forward each sequence one by one. If T % N != 0, the last sequence\n            will be zero-padded. Usually used during Transformer training phase.\n        Arguments:\n            - model (:obj:`Any`): Wrapped model class, should contain forward method.\n            - seq_len (:obj:`int`): N, length of a sequence.\n        '
        super().__init__(model)
        self.seq_len = seq_len

    def forward(self, obs: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        if False:
            print('Hello World!')
        "\n        Arguments:\n            - data (:obj:`dict`): Dict type data, including at least                 ['main_obs', 'target_obs', 'action', 'reward', 'done', 'weight']\n        Returns:\n            - List containing a dict of the model output for each sequence.\n        "
        sequences = list(torch.split(obs, self.seq_len, dim=0))
        if sequences[-1].shape[0] < self.seq_len:
            last = sequences[-1].clone()
            diff = self.seq_len - last.shape[0]
            sequences[-1] = F.pad(input=last, pad=(0, 0, 0, 0, 0, diff), mode='constant', value=0)
        outputs = []
        for (i, seq) in enumerate(sequences):
            out = self._model.forward(seq, **kwargs)
            outputs.append(out)
        out = {}
        for k in outputs[0].keys():
            out_k = [o[k] for o in outputs]
            out_k = torch.cat(out_k, dim=0)
            out[k] = out_k
        return out

class TransformerMemoryWrapper(IModelWrapper):

    def __init__(self, model: Any, batch_size: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Stores a copy of the Transformer memory in order to be reused across different phases. To make it more\n             clear, suppose the training pipeline is divided into 3 phases: evaluate, collect, learn. The goal of the\n             wrapper is to maintain the content of the memory at the end of each phase and reuse it when the same phase\n             is executed again. In this way, it prevents different phases to interferer each other memory.\n        Arguments:\n            - model (:obj:`Any`): Wrapped model class, should contain forward method.\n            - batch_size (:obj:`int`): Memory batch size.\n        '
        super().__init__(model)
        self._model.reset_memory(batch_size=batch_size)
        self.memory = self._model.get_memory()
        self.mem_shape = self.memory.shape

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        if False:
            return 10
        "\n        Arguments:\n            - data (:obj:`dict`): Dict type data, including at least                 ['main_obs', 'target_obs', 'action', 'reward', 'done', 'weight']\n        Returns:\n            - Output of the forward method.\n        "
        self._model.reset_memory(state=self.memory)
        out = self._model.forward(*args, **kwargs)
        self.memory = self._model.get_memory()
        return out

    def reset(self, *args, **kwargs):
        if False:
            print('Hello World!')
        state_id = kwargs.get('data_id', None)
        if state_id is None:
            self.memory = torch.zeros(self.mem_shape)
        else:
            self.reset_memory_entry(state_id)
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def reset_memory_entry(self, state_id: Optional[list]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Reset specific batch of the memory, batch ids are specified in 'state_id'\n        "
        for _id in state_id:
            self.memory[:, :, _id] = torch.zeros(self.mem_shape[-1])

    def show_memory_occupancy(self, layer=0) -> None:
        if False:
            i = 10
            return i + 15
        memory = self.memory
        memory_shape = memory.shape
        print('Layer {}-------------------------------------------'.format(layer))
        for b in range(memory_shape[-2]):
            print('b{}: '.format(b), end='')
            for m in range(memory_shape[1]):
                if sum(abs(memory[layer][m][b].flatten())) != 0:
                    print(1, end='')
                else:
                    print(0, end='')
            print()

def sample_action(logit=None, prob=None):
    if False:
        for i in range(10):
            print('nop')
    if prob is None:
        prob = torch.softmax(logit, dim=-1)
    shape = prob.shape
    prob += 1e-08
    prob = prob.view(-1, shape[-1])
    action = torch.multinomial(prob, 1).squeeze(-1)
    action = action.view(*shape[:-1])
    return action

class ArgmaxSampleWrapper(IModelWrapper):
    """
    Overview:
        Used to help the model to sample argmax action.
    Interfaces:
        ``forward``.
    """

    def forward(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Employ model forward computation graph, and use the output logit to greedily select max action (argmax).\n        '
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(100000000.0 * (1 - m)) for (l, m) in zip(logit, mask)]
        action = [l.argmax(dim=-1) for l in logit]
        if len(action) == 1:
            (action, logit) = (action[0], logit[0])
        output['action'] = action
        return output

class CombinationArgmaxSampleWrapper(IModelWrapper):
    """
    Overview:
        Used to help the model to sample combination argmax action.
    Interfaces:
        ``forward``.
    """

    def forward(self, shot_number, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        output = self._model.forward(*args, **kwargs)
        act = []
        mask = torch.zeros_like(output['logit'])
        for ii in range(shot_number):
            masked_logit = output['logit'] + mask
            actions = masked_logit.argmax(dim=-1)
            act.append(actions)
            for jj in range(actions.shape[0]):
                mask[jj][actions[jj]] = -100000000.0
        act = torch.stack(act, dim=1)
        output['action'] = act
        return output

class CombinationMultinomialSampleWrapper(IModelWrapper):
    """
    Overview:
        Used to help the model to sample combination multinomial action.
    Interfaces:
        ``forward``.
    """

    def forward(self, shot_number, *args, **kwargs):
        if False:
            return 10
        output = self._model.forward(*args, **kwargs)
        act = []
        mask = torch.zeros_like(output['logit'])
        for ii in range(shot_number):
            dist = torch.distributions.Categorical(logits=output['logit'] + mask)
            actions = dist.sample()
            act.append(actions)
            for jj in range(actions.shape[0]):
                mask[jj][actions[jj]] = -100000000.0
        act = torch.stack(act, dim=1)
        output['action'] = act
        return output

class HybridArgmaxSampleWrapper(IModelWrapper):
    """
    Overview:
        Used to help the model to sample argmax action in hybrid action space,
        i.e.{'action_type': discrete, 'action_args', continuous}
    Interfaces:
        ``forward``.
    """

    def forward(self, *args, **kwargs):
        if False:
            return 10
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        if 'logit' not in output:
            return output
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(100000000.0 * (1 - m)) for (l, m) in zip(logit, mask)]
        action = [l.argmax(dim=-1) for l in logit]
        if len(action) == 1:
            (action, logit) = (action[0], logit[0])
        output = {'action': {'action_type': action, 'action_args': output['action_args']}, 'logit': logit}
        return output

class MultinomialSampleWrapper(IModelWrapper):
    """
    Overview:
        Used to help the model get the corresponding action from the output['logits']self.
    Interfaces:
        ``forward``.
    """

    def forward(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'alpha' in kwargs.keys():
            alpha = kwargs.pop('alpha')
        else:
            alpha = None
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(100000000.0 * (1 - m)) for (l, m) in zip(logit, mask)]
        if alpha is None:
            action = [sample_action(logit=l) for l in logit]
        else:
            action = [sample_action(logit=l / alpha) for l in logit]
        if len(action) == 1:
            (action, logit) = (action[0], logit[0])
        output['action'] = action
        return output

class EpsGreedySampleWrapper(IModelWrapper):
    """
    Overview:
        Epsilon greedy sampler used in collector_model to help balance exploratin and exploitation.
        The type of eps can vary from different algorithms, such as:
        - float (i.e. python native scalar): for almost normal case
        - Dict[str, float]: for algorithm NGU
    Interfaces:
        ``forward``.
    """

    def forward(self, *args, **kwargs):
        if False:
            print('Hello World!')
        eps = kwargs.pop('eps')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(100000000.0 * (1 - m)) for (l, m) in zip(logit, mask)]
        else:
            mask = None
        action = []
        if isinstance(eps, dict):
            for (i, l) in enumerate(logit[0]):
                eps_tmp = eps[i]
                if np.random.random() > eps_tmp:
                    action.append(l.argmax(dim=-1))
                elif mask is not None:
                    action.append(sample_action(prob=mask[0][i].float().unsqueeze(0)).to(logit[0].device).squeeze(0))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]).to(logit[0].device))
            action = torch.stack(action, dim=-1)
        else:
            for (i, l) in enumerate(logit):
                if np.random.random() > eps:
                    action.append(l.argmax(dim=-1))
                elif mask is not None:
                    action.append(sample_action(prob=mask[i].float()))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
            if len(action) == 1:
                (action, logit) = (action[0], logit[0])
        output['action'] = action
        return output

class EpsGreedyMultinomialSampleWrapper(IModelWrapper):
    """
    Overview:
        Epsilon greedy sampler coupled with multinomial sample used in collector_model
        to help balance exploration and exploitation.
    Interfaces:
        ``forward``.
    """

    def forward(self, *args, **kwargs):
        if False:
            print('Hello World!')
        eps = kwargs.pop('eps')
        if 'alpha' in kwargs.keys():
            alpha = kwargs.pop('alpha')
        else:
            alpha = None
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(100000000.0 * (1 - m)) for (l, m) in zip(logit, mask)]
        else:
            mask = None
        action = []
        for (i, l) in enumerate(logit):
            if np.random.random() > eps:
                if alpha is None:
                    action = [sample_action(logit=l) for l in logit]
                else:
                    action = [sample_action(logit=l / alpha) for l in logit]
            elif mask:
                action.append(sample_action(prob=mask[i].float()))
            else:
                action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            (action, logit) = (action[0], logit[0])
        output['action'] = action
        return output

class HybridEpsGreedySampleWrapper(IModelWrapper):
    """
    Overview:
        Epsilon greedy sampler used in collector_model to help balance exploration and exploitation.
        In hybrid action space, i.e.{'action_type': discrete, 'action_args', continuous}
    Interfaces:
        ``forward``.
    """

    def forward(self, *args, **kwargs):
        if False:
            print('Hello World!')
        eps = kwargs.pop('eps')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(100000000.0 * (1 - m)) for (l, m) in zip(logit, mask)]
        else:
            mask = None
        action = []
        for (i, l) in enumerate(logit):
            if np.random.random() > eps:
                action.append(l.argmax(dim=-1))
            elif mask:
                action.append(sample_action(prob=mask[i].float()))
            else:
                action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            (action, logit) = (action[0], logit[0])
        output = {'action': {'action_type': action, 'action_args': output['action_args']}, 'logit': logit}
        return output

class HybridEpsGreedyMultinomialSampleWrapper(IModelWrapper):
    """
    Overview:
        Epsilon greedy sampler coupled with multinomial sample used in collector_model
        to help balance exploration and exploitation.
        In hybrid action space, i.e.{'action_type': discrete, 'action_args', continuous}
    Interfaces:
        ``forward``.
    """

    def forward(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        eps = kwargs.pop('eps')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        if 'logit' not in output:
            return output
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(100000000.0 * (1 - m)) for (l, m) in zip(logit, mask)]
        else:
            mask = None
        action = []
        for (i, l) in enumerate(logit):
            if np.random.random() > eps:
                action = [sample_action(logit=l) for l in logit]
            elif mask:
                action.append(sample_action(prob=mask[i].float()))
            else:
                action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            (action, logit) = (action[0], logit[0])
        output = {'action': {'action_type': action, 'action_args': output['action_args']}, 'logit': logit}
        return output

class HybridReparamMultinomialSampleWrapper(IModelWrapper):
    """
    Overview:
        Reparameterization sampler coupled with multinomial sample used in collector_model
        to help balance exploration and exploitation.
        In hybrid action space, i.e.{'action_type': discrete, 'action_args', continuous}
    Interfaces:
        forward
    """

    def forward(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        logit = output['logit']
        action_type_logit = logit['action_type']
        prob = torch.softmax(action_type_logit, dim=-1)
        pi_action = Categorical(prob)
        action_type = pi_action.sample()
        (mu, sigma) = (logit['action_args']['mu'], logit['action_args']['sigma'])
        dist = Independent(Normal(mu, sigma), 1)
        action_args = dist.sample()
        action = {'action_type': action_type, 'action_args': action_args}
        output['action'] = action
        return output

class HybridDeterministicArgmaxSampleWrapper(IModelWrapper):
    """
    Overview:
        Deterministic sampler coupled with argmax sample used in eval_model.
        In hybrid action space, i.e.{'action_type': discrete, 'action_args', continuous}
    Interfaces:
        forward
    """

    def forward(self, *args, **kwargs):
        if False:
            print('Hello World!')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        logit = output['logit']
        action_type_logit = logit['action_type']
        action_type = action_type_logit.argmax(dim=-1)
        mu = logit['action_args']['mu']
        action_args = mu
        action = {'action_type': action_type, 'action_args': action_args}
        output['action'] = action
        return output

class DeterministicSampleWrapper(IModelWrapper):
    """
    Overview:
        Deterministic sampler (just use mu directly) used in eval_model.
    Interfaces:
        forward
    """

    def forward(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        output['action'] = output['logit']['mu']
        return output

class ReparamSampleWrapper(IModelWrapper):
    """
    Overview:
        Reparameterization gaussian sampler used in collector_model.
    Interfaces:
        forward
    """

    def forward(self, *args, **kwargs):
        if False:
            print('Hello World!')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        (mu, sigma) = (output['logit']['mu'], output['logit']['sigma'])
        dist = Independent(Normal(mu, sigma), 1)
        output['action'] = dist.sample()
        return output

class ActionNoiseWrapper(IModelWrapper):
    """
    Overview:
        Add noise to collector's action output; Do clips on both generated noise and action after adding noise.
    Interfaces:
        ``__init__``, ``forward``.
    Arguments:
        - model (:obj:`Any`): Wrapped model class. Should contain ``forward`` method.
        - noise_type (:obj:`str`): The type of noise that should be generated, support ['gauss', 'ou'].
        - noise_kwargs (:obj:`dict`): Keyword args that should be used in noise init. Depends on ``noise_type``.
        - noise_range (:obj:`Optional[dict]`): Range of noise, used for clipping.
        - action_range (:obj:`Optional[dict]`): Range of action + noise, used for clip, default clip to [-1, 1].
    """

    def __init__(self, model: Any, noise_type: str='gauss', noise_kwargs: dict={}, noise_range: Optional[dict]=None, action_range: Optional[dict]={'min': -1, 'max': 1}) -> None:
        if False:
            return 10
        super().__init__(model)
        self.noise_generator = create_noise_generator(noise_type, noise_kwargs)
        self.noise_range = noise_range
        self.action_range = action_range

    def forward(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'sigma' in kwargs:
            sigma = kwargs.pop('sigma')
            if sigma is not None:
                self.noise_generator.sigma = sigma
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), 'model output must be dict, but find {}'.format(type(output))
        if 'action' in output or 'action_args' in output:
            key = 'action' if 'action' in output else 'action_args'
            action = output[key]
            assert isinstance(action, torch.Tensor)
            action = self.add_noise(action)
            output[key] = action
        return output

    def add_noise(self, action: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        "\n        Overview:\n            Generate noise and clip noise if needed. Add noise to action and clip action if needed.\n        Arguments:\n            - action (:obj:`torch.Tensor`): Model's action output.\n        Returns:\n            - noised_action (:obj:`torch.Tensor`): Action processed after adding noise and clipping.\n        "
        noise = self.noise_generator(action.shape, action.device)
        if self.noise_range is not None:
            noise = noise.clamp(self.noise_range['min'], self.noise_range['max'])
        action += noise
        if self.action_range is not None:
            action = action.clamp(self.action_range['min'], self.action_range['max'])
        return action

class TargetNetworkWrapper(IModelWrapper):
    """
    Overview:
        Maintain and update the target network
    Interfaces:
        update, reset
    """

    def __init__(self, model: Any, update_type: str, update_kwargs: dict):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model)
        assert update_type in ['momentum', 'assign']
        self._update_type = update_type
        self._update_kwargs = update_kwargs
        self._update_count = 0

    def reset(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        target_update_count = kwargs.pop('target_update_count', None)
        self.reset_state(target_update_count)
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def update(self, state_dict: dict, direct: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Update the target network state dict\n\n        Arguments:\n            - state_dict (:obj:`dict`): the state_dict from learner model\n            - direct (:obj:`bool`): whether to update the target network directly, \\\n                if true then will simply call the load_state_dict method of the model\n        '
        if direct:
            self._model.load_state_dict(state_dict, strict=True)
            self._update_count = 0
        elif self._update_type == 'assign':
            if (self._update_count + 1) % self._update_kwargs['freq'] == 0:
                self._model.load_state_dict(state_dict, strict=True)
            self._update_count += 1
        elif self._update_type == 'momentum':
            theta = self._update_kwargs['theta']
            for (name, p) in self._model.named_parameters():
                p.data = (1 - theta) * p.data + theta * state_dict[name]

    def reset_state(self, target_update_count: int=None) -> None:
        if False:
            return 10
        '\n        Overview:\n            Reset the update_count\n        Arguments:\n            target_update_count (:obj:`int`): reset target update count value.\n        '
        if target_update_count is not None:
            self._update_count = target_update_count

class TeacherNetworkWrapper(IModelWrapper):
    """
    Overview:
        Set the teacher Network. Set the model's model.teacher_cfg to the input teacher_cfg
    """

    def __init__(self, model, teacher_cfg):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model)
        self._model._teacher_cfg = teacher_cfg
        raise NotImplementedError
wrapper_name_map = {'base': BaseModelWrapper, 'hidden_state': HiddenStateWrapper, 'argmax_sample': ArgmaxSampleWrapper, 'hybrid_argmax_sample': HybridArgmaxSampleWrapper, 'eps_greedy_sample': EpsGreedySampleWrapper, 'eps_greedy_multinomial_sample': EpsGreedyMultinomialSampleWrapper, 'deterministic_sample': DeterministicSampleWrapper, 'reparam_sample': ReparamSampleWrapper, 'hybrid_eps_greedy_sample': HybridEpsGreedySampleWrapper, 'hybrid_eps_greedy_multinomial_sample': HybridEpsGreedyMultinomialSampleWrapper, 'hybrid_reparam_multinomial_sample': HybridReparamMultinomialSampleWrapper, 'hybrid_deterministic_argmax_sample': HybridDeterministicArgmaxSampleWrapper, 'multinomial_sample': MultinomialSampleWrapper, 'action_noise': ActionNoiseWrapper, 'transformer_input': TransformerInputWrapper, 'transformer_segment': TransformerSegmentWrapper, 'transformer_memory': TransformerMemoryWrapper, 'target': TargetNetworkWrapper, 'teacher': TeacherNetworkWrapper, 'combination_argmax_sample': CombinationArgmaxSampleWrapper, 'combination_multinomial_sample': CombinationMultinomialSampleWrapper}

def model_wrap(model: Union[nn.Module, IModelWrapper], wrapper_name: str=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Wrap the model with the specified wrapper and return the wrappered model.\n    Arguments:\n        - model (:obj:`Any`): The model to be wrapped.\n        - wrapper_name (:obj:`str`): The name of the wrapper to be used.\n\n    .. note::\n        The arguments of the wrapper should be passed in as kwargs.\n    '
    if wrapper_name in wrapper_name_map:
        if not isinstance(model, IModelWrapper):
            model = wrapper_name_map['base'](model)
        model = wrapper_name_map[wrapper_name](model, **kwargs)
    else:
        raise TypeError('not support model_wrapper type: {}'.format(wrapper_name))
    return model

def register_wrapper(name: str, wrapper_type: type) -> None:
    if False:
        return 10
    '\n    Overview:\n        Register new wrapper to ``wrapper_name_map``. When user implements a new wrapper, they must call this function         to complete the registration. Then the wrapper can be called by ``model_wrap``.\n    Arguments:\n        - name (:obj:`str`): The name of the new wrapper to be registered.\n        - wrapper_type (:obj:`type`): The wrapper class needs to be added in ``wrapper_name_map``. This argument             should be the subclass of ``IModelWrapper``.\n    '
    assert isinstance(name, str)
    assert issubclass(wrapper_type, IModelWrapper)
    wrapper_name_map[name] = wrapper_type
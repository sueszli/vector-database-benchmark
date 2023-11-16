from typing import Optional, List, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod
from collections import namedtuple
from easydict import EasyDict
import copy
import torch
from ding.model import create_model
from ding.utils import import_module, allreduce, broadcast, get_rank, allreduce_async, synchronize, deep_merge_dicts, POLICY_REGISTRY

class Policy(ABC):
    """
    Overview:
        The basic class of Reinforcement Learning (RL) and Imitation Learning (IL) policy in DI-engine.
    Property:
        ``cfg``, ``learn_mode``, ``collect_mode``, ``eval_mode``
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Get the default config of policy. This method is used to create the default config of policy.\n        Returns:\n            - cfg (:obj:`EasyDict`): The default config of corresponding policy. For the derived policy class,                 it will recursively merge the default config of base class and its own default config.\n\n        .. tip::\n            This method will deepcopy the ``config`` attribute of the class and return the result. So users don't need             to worry about the modification of the returned config.\n        "
        if cls == Policy:
            raise RuntimeError("Basic class Policy doesn't have completed default_config")
        base_cls = cls.__base__
        if base_cls == Policy:
            base_policy_cfg = EasyDict(copy.deepcopy(Policy.config))
        else:
            base_policy_cfg = copy.deepcopy(base_cls.default_config())
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg = deep_merge_dicts(base_policy_cfg, cfg)
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    learn_function = namedtuple('learn_function', ['forward', 'reset', 'info', 'monitor_vars', 'get_attribute', 'set_attribute', 'state_dict', 'load_state_dict'])
    collect_function = namedtuple('collect_function', ['forward', 'process_transition', 'get_train_sample', 'reset', 'get_attribute', 'set_attribute', 'state_dict', 'load_state_dict'])
    eval_function = namedtuple('eval_function', ['forward', 'reset', 'get_attribute', 'set_attribute', 'state_dict', 'load_state_dict'])
    total_field = set(['learn', 'collect', 'eval'])
    config = dict(on_policy=False, cuda=False, multi_gpu=False, bp_update_sync=True, traj_len_inf=False, model=dict())

    def __init__(self, cfg: EasyDict, model: Optional[torch.nn.Module]=None, enable_field: Optional[List[str]]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Initialize policy instance according to input configures and model. This method will initialize differnent             fields in policy, including ``learn``, ``collect``, ``eval``. The ``learn`` field is used to train the             policy, the ``collect`` field is used to collect data for training, and the ``eval`` field is used to             evaluate the policy. The ``enable_field`` is used to specify which field to initialize, if it is None,             then all fields will be initialized.\n        Arguments:\n            - cfg (:obj:`EasyDict`): The final merged config used to initialize policy. For the default config,                 see the ``config`` attribute and its comments of policy class.\n            - model (:obj:`torch.nn.Module`): The neural network model used to initialize policy. If it                 is None, then the model will be created according to ``default_model`` method and ``cfg.model`` field.                 Otherwise, the model will be set to the ``model`` instance created by outside caller.\n            - enable_field (:obj:`Optional[List[str]]`): The field list to initialize. If it is None, then all fields                 will be initialized. Otherwise, only the fields in ``enable_field`` will be initialized, which is                 beneficial to save resources.\n\n        .. note::\n            For the derived policy class, it should implement the ``_init_learn``, ``_init_collect``, ``_init_eval``             method to initialize the corresponding field.\n        '
        self._cfg = cfg
        self._on_policy = self._cfg.on_policy
        if enable_field is None:
            self._enable_field = self.total_field
        else:
            self._enable_field = enable_field
        assert set(self._enable_field).issubset(self.total_field), self._enable_field
        if len(set(self._enable_field).intersection(set(['learn', 'collect', 'eval']))) > 0:
            model = self._create_model(cfg, model)
            self._cuda = cfg.cuda and torch.cuda.is_available()
            if len(set(self._enable_field).intersection(set(['learn']))) > 0:
                multi_gpu = self._cfg.multi_gpu
                self._rank = get_rank() if multi_gpu else 0
                if self._cuda:
                    model.cuda()
                if multi_gpu:
                    bp_update_sync = self._cfg.bp_update_sync
                    self._bp_update_sync = bp_update_sync
                    self._init_multi_gpu_setting(model, bp_update_sync)
            else:
                self._rank = 0
                if self._cuda:
                    model.cuda()
            self._model = model
            self._device = 'cuda:{}'.format(self._rank % torch.cuda.device_count()) if self._cuda else 'cpu'
        else:
            self._cuda = False
            self._rank = 0
            self._device = 'cpu'
        for field in self._enable_field:
            getattr(self, '_init_' + field)()

    def _init_multi_gpu_setting(self, model: torch.nn.Module, bp_update_sync: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Initialize multi-gpu data parallel training setting, including broadcast model parameters at the beginning             of the training, and prepare the hook function to allreduce the gradients of model parameters.\n        Arguments:\n            - model (:obj:`torch.nn.Module`): The neural network model to be trained.\n            - bp_update_sync (:obj:`bool`): Whether to synchronize update the model parameters after allreduce the                 gradients of model parameters. Async update can be parallel in different network layers like pipeline                 so that it can save time.\n        '
        for (name, param) in model.state_dict().items():
            assert isinstance(param.data, torch.Tensor), type(param.data)
            broadcast(param.data, 0)
        for (name, param) in model.named_parameters():
            setattr(param, 'grad', torch.zeros_like(param))
        if not bp_update_sync:

            def make_hook(name, p):
                if False:
                    i = 10
                    return i + 15

                def hook(*ignore):
                    if False:
                        while True:
                            i = 10
                    allreduce_async(name, p.grad.data)
                return hook
            for (i, (name, p)) in enumerate(model.named_parameters()):
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(make_hook(name, p))

    def _create_model(self, cfg: EasyDict, model: Optional[torch.nn.Module]=None) -> torch.nn.Module:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Create or validate the neural network model according to input configures and model. If the input model is             None, then the model will be created according to ``default_model`` method and ``cfg.model`` field.             Otherwise, the model will be verified as an instance of ``torch.nn.Module`` and set to the ``model``             instance created by outside caller.\n        Arguments:\n            - cfg (:obj:`EasyDict`): The final merged config used to initialize policy.\n            - model (:obj:`torch.nn.Module`): The neural network model used to initialize policy. User can refer to                 the default model defined in corresponding policy to customize its own model.\n        Returns:\n            - model (:obj:`torch.nn.Module`): The created neural network model. The different modes of policy will                 add distinct wrappers and plugins to the model, which is used to train, collect and evaluate.\n        Raises:\n            - RuntimeError: If the input model is not None and is not an instance of ``torch.nn.Module``.\n        '
        if model is None:
            model_cfg = cfg.model
            if 'type' not in model_cfg:
                (m_type, import_names) = self.default_model()
                model_cfg.type = m_type
                model_cfg.import_names = import_names
            return create_model(model_cfg)
        elif isinstance(model, torch.nn.Module):
            return model
        else:
            raise RuntimeError('invalid model: {}'.format(type(model)))

    @property
    def cfg(self) -> EasyDict:
        if False:
            for i in range(10):
                print('nop')
        return self._cfg

    @abstractmethod
    def _init_learn(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Initialize the learn mode of policy, including related attributes and modules. This method will be             called in ``__init__`` method if ``learn`` field is in ``enable_field``. Almost different policies have             its own learn mode, so this method must be overrided in subclass.\n\n        .. note::\n            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn``             and ``_load_state_dict_learn`` methods.\n\n        .. note::\n            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.\n\n        .. note::\n            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them             with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.\n        "
        raise NotImplementedError

    @abstractmethod
    def _init_collect(self) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Initialize the collect mode of policy, including related attributes and modules. This method will be             called in ``__init__`` method if ``collect`` field is in ``enable_field``. Almost different policies have             its own collect mode, so this method must be overrided in subclass.\n\n        .. note::\n            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_collect``             and ``_load_state_dict_collect`` methods.\n\n        .. note::\n            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them             with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.\n        "
        raise NotImplementedError

    @abstractmethod
    def _init_eval(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Initialize the eval mode of policy, including related attributes and modules. This method will be             called in ``__init__`` method if ``eval`` field is in ``enable_field``. Almost different policies have             its own eval mode, so this method must be overrided in subclass.\n\n        .. note::\n            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_eval``             and ``_load_state_dict_eval`` methods.\n\n        .. note::\n            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them             with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.\n        "
        raise NotImplementedError

    @property
    def learn_mode(self) -> 'Policy.learn_function':
        if False:
            return 10
        '\n        Overview:\n            Return the interfaces of learn mode of policy, which is used to train the model. Here we use namedtuple             to define immutable interfaces and restrict the usage of policy in different mode. Moreover, derived             subclass can override the interfaces to customize its own learn mode.\n        Returns:\n            - interfaces (:obj:`Policy.learn_function`): The interfaces of learn mode of policy, it is a namedtuple                 whose values of distinct fields are different internal methods.\n        Examples:\n            >>> policy = Policy(cfg, model)\n            >>> policy_learn = policy.learn_mode\n            >>> train_output = policy_learn.forward(data)\n            >>> state_dict = policy_learn.state_dict()\n        '
        return Policy.learn_function(self._forward_learn, self._reset_learn, self.__repr__, self._monitor_vars_learn, self._get_attribute, self._set_attribute, self._state_dict_learn, self._load_state_dict_learn)

    @property
    def collect_mode(self) -> 'Policy.collect_function':
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Return the interfaces of collect mode of policy, which is used to train the model. Here we use namedtuple             to define immutable interfaces and restrict the usage of policy in different mode. Moreover, derived             subclass can override the interfaces to customize its own collect mode.\n        Returns:\n            - interfaces (:obj:`Policy.collect_function`): The interfaces of collect mode of policy, it is a                 namedtuple whose values of distinct fields are different internal methods.\n        Examples:\n            >>> policy = Policy(cfg, model)\n            >>> policy_collect = policy.collect_mode\n            >>> obs = env_manager.ready_obs\n            >>> inference_output = policy_collect.forward(obs)\n            >>> next_obs, rew, done, info = env_manager.step(inference_output.action)\n        '
        return Policy.collect_function(self._forward_collect, self._process_transition, self._get_train_sample, self._reset_collect, self._get_attribute, self._set_attribute, self._state_dict_collect, self._load_state_dict_collect)

    @property
    def eval_mode(self) -> 'Policy.eval_function':
        if False:
            return 10
        '\n        Overview:\n            Return the interfaces of eval mode of policy, which is used to train the model. Here we use namedtuple             to define immutable interfaces and restrict the usage of policy in different mode. Moreover, derived             subclass can override the interfaces to customize its own eval mode.\n        Returns:\n            - interfaces (:obj:`Policy.eval_function`): The interfaces of eval mode of policy, it is a namedtuple                 whose values of distinct fields are different internal methods.\n        Examples:\n            >>> policy = Policy(cfg, model)\n            >>> policy_eval = policy.eval_mode\n            >>> obs = env_manager.ready_obs\n            >>> inference_output = policy_eval.forward(obs)\n            >>> next_obs, rew, done, info = env_manager.step(inference_output.action)\n        '
        return Policy.eval_function(self._forward_eval, self._reset_eval, self._get_attribute, self._set_attribute, self._state_dict_eval, self._load_state_dict_eval)

    def _set_attribute(self, name: str, value: Any) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            In order to control the access of the policy attributes, we expose different modes to outside rather than             directly use the policy instance. And we also provide a method to set the attribute of the policy in             different modes. And the new attribute will named as ``_{name}``.\n        Arguments:\n            - name (:obj:`str`): The name of the attribute.\n            - value (:obj:`Any`): The value of the attribute.\n        '
        setattr(self, '_' + name, value)

    def _get_attribute(self, name: str) -> Any:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            In order to control the access of the policy attributes, we expose different modes to outside rather than             directly use the policy instance. And we also provide a method to get the attribute of the policy in             different modes.\n        Arguments:\n            - name (:obj:`str`): The name of the attribute.\n        Returns:\n            - value (:obj:`Any`): The value of the attribute.\n\n        .. note::\n            DI-engine's policy will first try to access `_get_{name}` method, and then try to access `_{name}`             attribute. If both of them are not found, it will raise a ``NotImplementedError``.\n        "
        if hasattr(self, '_get_' + name):
            return getattr(self, '_get_' + name)()
        elif hasattr(self, '_' + name):
            return getattr(self, '_' + name)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        if False:
            return 10
        '\n        Overview:\n            Get the string representation of the policy.\n        Returns:\n            - repr (:obj:`str`): The string representation of the policy.\n        '
        return 'DI-engine DRL Policy\n{}'.format(repr(self._model))

    def sync_gradients(self, model: torch.nn.Module) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Synchronize (allreduce) gradients of model parameters in data-parallel multi-gpu training.\n        Arguments:\n            - model (:obj:`torch.nn.Module`): The model to synchronize gradients.\n\n        .. note::\n            This method is only used in multi-gpu training, and it shoule be called after ``backward`` method and             before ``step`` method. The user can also use ``bp_update_sync`` config to control whether to synchronize             gradients allreduce and optimizer updates.\n        '
        if self._bp_update_sync:
            for (name, param) in model.named_parameters():
                if param.requires_grad:
                    allreduce(param.grad.data)
        else:
            synchronize()

    def default_model(self) -> Tuple[str, List[str]]:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will             automatically call this method to get the default model setting and create model.\n        Returns:\n            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.\n\n        .. note::\n            The user can define and use customized network model but must obey the same inferface definition indicated             by import_names path. For example about DQN, its registered name is ``dqn`` and the import_names is             ``ding.model.template.q_learning.DQN``\n        "
        raise NotImplementedError

    @abstractmethod
    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Overview:\n            Policy forward function of learn mode (training policy and updating parameters). Forward means             that the policy inputs some training batch data from the replay buffer and then returns the output             result, including various training information such as loss value, policy entropy, q value, priority,             and so on. This method is left to be implemented by the subclass, and more arguments can be added in             ``data`` item if necessary.\n        Arguments:\n            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of                 training samples. For each element in list, the key of the dict is the name of data items and the                 value is the corresponding data. Usually, in the ``_forward_learn`` method, data should be stacked in                 the batch dimension by some utility functions such as ``default_preprocess_learn``.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): The training information of policy forward, including some metrics for                 monitoring training such as loss, priority, q value, policy entropy, and some data for next step                 training such as priority. Note the output data item should be Python native scalar rather than                 PyTorch tensor, which is convenient for the outside to use.\n        '
        raise NotImplementedError

    def _reset_learn(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            return 10
        '\n        Overview:\n            Reset some stateful variables for learn mode when necessary, such as the hidden state of RNN or the             memory bank of some special algortihms. If ``data_id`` is None, it means to reset all the stateful             varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example,             different trajectories in ``data_id`` will have different hidden state in RNN.\n        Arguments:\n            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables                 specified by ``data_id``.\n\n        .. note::\n            This method is not mandatory to be implemented. The sub-class can overwrite this method if necessary.\n        '
        pass

    def _monitor_vars_learn(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such             as text logger, tensorboard logger, will use these keys to save the corresponding data.\n        Returns:\n            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.\n\n        .. tip::\n            The default implementation is ``['cur_lr', 'total_loss']``. Other derived classes can overwrite this             method to add their own keys if necessary.\n        "
        return ['cur_lr', 'total_loss']

    def _state_dict_learn(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Return the state_dict of learn mode, usually including model and optimizer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.\n        '
        return {'model': self._learn_model.state_dict(), 'optimizer': self._optimizer.state_dict()}

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Load the state_dict variable into policy learn mode.\n        Arguments:\n            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.\n\n        .. tip::\n            If you want to only load some parts of model, you can simply set the ``strict`` argument in             load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more             complicated operation.\n        '
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _get_batch_size(self) -> Union[int, Dict[str, int]]:
        if False:
            i = 10
            return i + 15
        if 'batch_size' in self._cfg:
            return self._cfg.batch_size
        else:
            return self._cfg.learn.batch_size

    @abstractmethod
    def _forward_collect(self, data: Dict[int, Any], **kwargs) -> Dict[int, Any]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means             that the policy gets some necessary data (mainly observation) from the envs and then returns the output             data, such as the action to interact with the envs, or the action logits to calculate the loss in learn             mode. This method is left to be implemented by the subclass, and more arguments can be added in ``kwargs``             part if necessary.\n        Arguments:\n            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The                 key of the dict is environment id and the value is the corresponding data of the env.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and                 other necessary data for learn mode defined in ``self._process_transition`` method. The key of the                 dict is the same as the input data, i.e. environment id.\n        '
        raise NotImplementedError

    @abstractmethod
    def _process_transition(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]], policy_output: Dict[str, torch.Tensor], timestep: namedtuple) -> Dict[str, torch.Tensor]:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Process and pack one timestep transition data into a dict, such as <s, a, r, s', done>. Some policies             need to do some special process and pack its own necessary attributes (e.g. hidden state and logit),             so this method is left to be implemented by the subclass.\n        Arguments:\n            - obs (:obj:`Union[torch.Tensor, Dict[str, torch.Tensor]]`): The observation of the current timestep.\n            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation                 as input. Usually, it contains the action and the logit of the action.\n            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method,                 except all the elements have been transformed into tensor data. Usually, it contains the next obs,                 reward, done, info, etc.\n        Returns:\n            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.\n        "
        raise NotImplementedError

    @abstractmethod
    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that             can be used for training directly. A train sample can be a processed transition (DQN with nstep TD)             or some multi-timestep transitions (DRQN). This method is usually used in collectors to execute necessary             RL data preprocessing before training, which can help learner amortize revelant time consumption.             In addition, you can also implement this method as an identity function and do the data processing             in ``self._forward_learn`` method.\n        Arguments:\n            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is                 the same format as the return value of ``self._process_transition`` method.\n        Returns:\n            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format                 as input transitions, but may contain more data for training, such as nstep reward, advantage, etc.\n\n        .. note::\n            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version.             And the user can customize the this data processing procecure by overriding this two methods and collector             itself\n        '
        raise NotImplementedError

    def _reset_collect(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Reset some stateful variables for collect mode when necessary, such as the hidden state of RNN or the             memory bank of some special algortihms. If ``data_id`` is None, it means to reset all the stateful             varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example,             different environments/episodes in collecting in ``data_id`` will have different hidden state in RNN.\n        Arguments:\n            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables                 specified by ``data_id``.\n\n        .. note::\n            This method is not mandatory to be implemented. The sub-class can overwrite this method if necessary.\n        '
        pass

    def _state_dict_collect(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Return the state_dict of collect mode, only including model in usual, which is necessary for distributed             training scenarios to auto-recover collectors.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy collect state, for saving and restoring.\n\n        .. tip::\n            Not all the scenarios need to auto-recover collectors, sometimes, we can directly shutdown the crashed             collector and renew a new one.\n        '
        return {'model': self._collect_model.state_dict()}

    def _load_state_dict_collect(self, state_dict: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Load the state_dict variable into policy collect mode, such as load pretrained state_dict, auto-recover             checkpoint, or model replica from learner in distributed training scenarios.\n        Arguments:\n            - state_dict (:obj:`Dict[str, Any]`): The dict of policy collect state saved before.\n\n        .. tip::\n            If you want to only load some parts of model, you can simply set the ``strict`` argument in             load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more             complicated operation.\n        '
        self._collect_model.load_state_dict(state_dict['model'], strict=True)

    def _get_n_sample(self) -> Union[int, None]:
        if False:
            i = 10
            return i + 15
        if 'n_sample' in self._cfg:
            return self._cfg.n_sample
        else:
            return self._cfg.collect.get('n_sample', None)

    def _get_n_episode(self) -> Union[int, None]:
        if False:
            return 10
        if 'n_episode' in self._cfg:
            return self._cfg.n_episode
        else:
            return self._cfg.collect.get('n_episode', None)

    @abstractmethod
    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Policy forward function of eval mode (evaluation policy performance, such as interacting with envs or             computing metrics on validation dataset). Forward means that the policy gets some necessary data (mainly             observation) from the envs and then returns the output data, such as the action to interact with the envs.             This method is left to be implemented by the subclass.\n        Arguments:\n            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The                 key of the dict is environment id and the value is the corresponding data of the env.\n        Returns:\n            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The                 key of the dict is the same as the input data, i.e. environment id.\n        '
        raise NotImplementedError

    def _reset_eval(self, data_id: Optional[List[int]]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Reset some stateful variables for eval mode when necessary, such as the hidden state of RNN or the             memory bank of some special algortihms. If ``data_id`` is None, it means to reset all the stateful             varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example,             different environments/episodes in evaluation in ``data_id`` will have different hidden state in RNN.\n        Arguments:\n            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables                 specified by ``data_id``.\n\n        .. note::\n            This method is not mandatory to be implemented. The sub-class can overwrite this method if necessary.\n        '
        pass

    def _state_dict_eval(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Return the state_dict of eval mode, only including model in usual, which is necessary for distributed             training scenarios to auto-recover evaluators.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy eval state, for saving and restoring.\n\n        .. tip::\n            Not all the scenarios need to auto-recover evaluators, sometimes, we can directly shutdown the crashed             evaluator and renew a new one.\n        '
        return {'model': self._eval_model.state_dict()}

    def _load_state_dict_eval(self, state_dict: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Load the state_dict variable into policy eval mode, such as load auto-recover             checkpoint, or model replica from learner in distributed training scenarios.\n        Arguments:\n            - state_dict (:obj:`Dict[str, Any]`): The dict of policy eval state saved before.\n\n        .. tip::\n            If you want to only load some parts of model, you can simply set the ``strict`` argument in             load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more             complicated operation.\n        '
        self._eval_model.load_state_dict(state_dict['model'], strict=True)

class CommandModePolicy(Policy):
    """
    Overview:
        Policy with command mode, which can be used in old version of DI-engine pipeline: ``serial_pipeline``.         ``CommandModePolicy`` uses ``_get_setting_learn``, ``_get_setting_collect``, ``_get_setting_eval`` methods         to exchange information between different workers.

    Interface:
        ``_init_command``, ``_get_setting_learn``, ``_get_setting_collect``, ``_get_setting_eval``
    Property:
        ``command_mode``
    """
    command_function = namedtuple('command_function', ['get_setting_learn', 'get_setting_collect', 'get_setting_eval'])
    total_field = set(['learn', 'collect', 'eval', 'command'])

    @property
    def command_mode(self) -> 'Policy.command_function':
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Return the interfaces of command mode of policy, which is used to train the model. Here we use namedtuple             to define immutable interfaces and restrict the usage of policy in different mode. Moreover, derived             subclass can override the interfaces to customize its own command mode.\n        Returns:\n            - interfaces (:obj:`Policy.command_function`): The interfaces of command mode, it is a namedtuple                 whose values of distinct fields are different internal methods.\n        Examples:\n            >>> policy = CommandModePolicy(cfg, model)\n            >>> policy_command = policy.command_mode\n            >>> settings = policy_command.get_setting_learn(command_info)\n        '
        return CommandModePolicy.command_function(self._get_setting_learn, self._get_setting_collect, self._get_setting_eval)

    @abstractmethod
    def _init_command(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Initialize the command mode of policy, including related attributes and modules. This method will be             called in ``__init__`` method if ``command`` field is in ``enable_field``. Almost different policies have             its own command mode, so this method must be overrided in subclass.\n\n        .. note::\n            If you want to set some spacial member variables in ``_init_command`` method, you'd better name them             with prefix ``_command_`` to avoid conflict with other modes, such as ``self._command_attr1``.\n        "
        raise NotImplementedError

    @abstractmethod
    def _get_setting_learn(self, command_info: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Accoding to ``command_info``, i.e., global training information (e.g. training iteration, collected env             step, evaluation results, etc.), return the setting of learn mode, which contains dynamically changed             hyperparameters for learn mode, such as ``batch_size``, ``learning_rate``, etc.\n        Arguments:\n            - command_info (:obj:`Dict[str, Any]`): The global training information, which is defined in ``commander``.\n        Returns:\n            - setting (:obj:`Dict[str, Any]`): The latest setting of learn mode, which is usually used as extra                 arguments of the ``policy._forward_learn`` method.\n        '
        raise NotImplementedError

    @abstractmethod
    def _get_setting_collect(self, command_info: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Accoding to ``command_info``, i.e., global training information (e.g. training iteration, collected env             step, evaluation results, etc.), return the setting of collect mode, which contains dynamically changed             hyperparameters for collect mode, such as ``eps``, ``temperature``, etc.\n        Arguments:\n            - command_info (:obj:`Dict[str, Any]`): The global training information, which is defined in ``commander``.\n        Returns:\n            - setting (:obj:`Dict[str, Any]`): The latest setting of collect mode, which is usually used as extra                 arguments of the ``policy._forward_collect`` method.\n        '
        raise NotImplementedError

    @abstractmethod
    def _get_setting_eval(self, command_info: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Accoding to ``command_info``, i.e., global training information (e.g. training iteration, collected env             step, evaluation results, etc.), return the setting of eval mode, which contains dynamically changed             hyperparameters for eval mode, such as ``temperature``, etc.\n        Arguments:\n            - command_info (:obj:`Dict[str, Any]`): The global training information, which is defined in ``commander``.\n        Returns:\n            - setting (:obj:`Dict[str, Any]`): The latest setting of eval mode, which is usually used as extra                 arguments of the ``policy._forward_eval`` method.\n        '
        raise NotImplementedError

def create_policy(cfg: EasyDict, **kwargs) -> Policy:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        Create a policy instance according to ``cfg`` and other kwargs.\n    Arguments:\n        - cfg (:obj:`EasyDict`): Final merged policy config.\n    ArgumentsKeys:\n        - type (:obj:`str`): Policy type set in ``POLICY_REGISTRY.register`` method , such as ``dqn`` .\n        - import_names (:obj:`List[str]`): A list of module names (paths) to import before creating policy, such             as ``ding.policy.dqn`` .\n    Returns:\n        - policy (:obj:`Policy`): The created policy instance.\n\n    .. tip::\n        ``kwargs`` contains other arguments that need to be passed to the policy constructor. You can refer to         the ``__init__`` method of the corresponding policy class for details.\n\n    .. note::\n        For more details about how to merge config, please refer to the system document of DI-engine         (`en link <../03_system/config.html>`_).\n    '
    import_module(cfg.get('import_names', []))
    return POLICY_REGISTRY.build(cfg.type, cfg=cfg, **kwargs)

def get_policy_cls(cfg: EasyDict) -> type:
    if False:
        return 10
    '\n    Overview:\n        Get policy class according to ``cfg``, which is used to access related class variables/methods.\n    Arguments:\n        - cfg (:obj:`EasyDict`): Final merged policy config.\n    ArgumentsKeys:\n        - type (:obj:`str`): Policy type set in ``POLICY_REGISTRY.register`` method , such as ``dqn`` .\n        - import_names (:obj:`List[str]`): A list of module names (paths) to import before creating policy, such             as ``ding.policy.dqn`` .\n    Returns:\n        - policy (:obj:`type`): The policy class.\n    '
    import_module(cfg.get('import_names', []))
    return POLICY_REGISTRY.get(cfg.type)
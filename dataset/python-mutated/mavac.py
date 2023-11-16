from typing import Union, Dict, Optional
import torch
import torch.nn as nn
from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead

@MODEL_REGISTRY.register('mavac')
class MAVAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of algorithms related to (state) Value Actor-Critic (VAC) for         multi-agent, such as MAPPO(https://arxiv.org/abs/2103.01955). This model now supports discrete and         continuous action space. The MAVAC is composed of four parts: ``actor_encoder``, ``critic_encoder``,         ``actor_head`` and ``critic_head``. Encoders are used to extract the feature from various observation.         Heads are used to predict corresponding value or action logit.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``, ``compute_actor_critic``.
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(self, agent_obs_shape: Union[int, SequenceType], global_obs_shape: Union[int, SequenceType], action_shape: Union[int, SequenceType], agent_num: int, actor_head_hidden_size: int=256, actor_head_layer_num: int=2, critic_head_hidden_size: int=512, critic_head_layer_num: int=1, action_space: str='discrete', activation: Optional[nn.Module]=nn.ReLU(), norm_type: Optional[str]=None, sigma_type: Optional[str]='independent', bound_type: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Init the MAVAC Model according to arguments.\n        Arguments:\n            - agent_obs_shape (:obj:`Union[int, SequenceType]`): Observation's space for single agent,                 such as 8 or [4, 84, 84].\n            - global_obs_shape (:obj:`Union[int, SequenceType]`): Global observation's space, such as 8 or [4, 84, 84].\n            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape for single agent, such as 6                 or [2, 3, 3].\n            - agent_num (:obj:`int`): This parameter is temporarily reserved. This parameter may be required for                 subsequent changes to the model\n            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of ``actor_head`` network, defaults                 to 256, it must match the last element of ``agent_obs_shape``.\n            - actor_head_layer_num (:obj:`int`): The num of layers used in the ``actor_head`` network to compute action.\n            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of ``critic_head`` network, defaults                 to 512, it must match the last element of ``global_obs_shape``.\n            - critic_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output for                 critic's nn.\n            - action_space (:obj:`Union[int, SequenceType]`): The type of different action spaces, including                 ['discrete', 'continuous'], then will instantiate corresponding head, including ``DiscreteHead``                 and ``ReparameterizationHead``.\n            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` the after                 ``layer_fn``, if ``None`` then default set to ``nn.ReLU()``.\n            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see                 ``ding.torch_utils.fc_block`` for more details. you can choose one of ['BN', 'IN', 'SyncBN', 'LN'].\n            - sigma_type (:obj:`Optional[str]`): The type of sigma in continuous action space, see                 ``ding.torch_utils.network.dreamer.ReparameterizationHead`` for more details, in MAPPO, it defaults                 to ``independent``, which means state-independent sigma parameters.\n            - bound_type (:obj:`Optional[str]`): The type of action bound methods in continuous action space, defaults                 to ``None``, which means no bound.\n        "
        super(MAVAC, self).__init__()
        agent_obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape: int = squeeze(action_shape)
        (self.global_obs_shape, self.agent_obs_shape, self.action_shape) = (global_obs_shape, agent_obs_shape, action_shape)
        self.action_space = action_space
        self.actor_encoder = nn.Identity()
        self.critic_encoder = nn.Identity()
        self.critic_head = nn.Sequential(nn.Linear(global_obs_shape, critic_head_hidden_size), activation, RegressionHead(critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type))
        assert self.action_space in ['discrete', 'continuous'], self.action_space
        if self.action_space == 'discrete':
            self.actor_head = nn.Sequential(nn.Linear(agent_obs_shape, actor_head_hidden_size), activation, DiscreteHead(actor_head_hidden_size, action_shape, actor_head_layer_num, activation=activation, norm_type=norm_type))
        elif self.action_space == 'continuous':
            self.actor_head = nn.Sequential(nn.Linear(agent_obs_shape, actor_head_hidden_size), activation, ReparameterizationHead(actor_head_hidden_size, action_shape, actor_head_layer_num, sigma_type=sigma_type, activation=activation, norm_type=norm_type, bound_type=bound_type))
        self.actor = [self.actor_encoder, self.actor_head]
        self.critic = [self.critic_encoder, self.critic_head]
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            MAVAC forward computation graph, input observation tensor to predict state value or action logit.             ``mode`` includes ``compute_actor``, ``compute_critic``, ``compute_actor_critic``.\n            Different ``mode`` will forward with different network modules to get different outputs and save             computation.\n        Arguments:\n            - inputs (:obj:`Dict`): The input dict including observation and related info,                 whose key-values vary from different ``mode``.\n            - mode (:obj:`str`): The forward mode, all the modes are defined in the beginning of this class.\n        Returns:\n            - outputs (:obj:`Dict`): The output dict of MAVAC's forward computation graph, whose key-values vary from                 different ``mode``.\n\n        Examples (Actor):\n            >>> model = MAVAC(agent_obs_shape=64, global_obs_shape=128, action_shape=14)\n            >>> inputs = {\n                    'agent_state': torch.randn(10, 8, 64),\n                    'global_state': torch.randn(10, 8, 128),\n                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))\n                }\n            >>> actor_outputs = model(inputs,'compute_actor')\n            >>> assert actor_outputs['logit'].shape == torch.Size([10, 8, 14])\n\n        Examples (Critic):\n            >>> model = MAVAC(agent_obs_shape=64, global_obs_shape=128, action_shape=14)\n            >>> inputs = {\n                    'agent_state': torch.randn(10, 8, 64),\n                    'global_state': torch.randn(10, 8, 128),\n                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))\n                }\n            >>> critic_outputs = model(inputs,'compute_critic')\n            >>> assert actor_outputs['value'].shape == torch.Size([10, 8])\n\n        Examples (Actor-Critic):\n            >>> model = MAVAC(64, 64)\n            >>> inputs = {\n                    'agent_state': torch.randn(10, 8, 64),\n                    'global_state': torch.randn(10, 8, 128),\n                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))\n                }\n            >>> outputs = model(inputs,'compute_actor_critic')\n            >>> assert outputs['value'].shape == torch.Size([10, 8, 14])\n            >>> assert outputs['logit'].shape == torch.Size([10, 8])\n\n        "
        assert mode in self.mode, 'not support forward mode: {}/{}'.format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: Dict) -> Dict:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            MAVAC forward computation graph for actor part,             predicting action logit with agent observation tensor in ``x``.\n        Arguments:\n            - x (:obj:`Dict`): Input data dict with keys ['agent_state', 'action_mask'(optional)].\n                - agent_state: (:obj:`torch.Tensor`): Each agent local state(obs).\n                - action_mask(optional): (:obj:`torch.Tensor`): When ``action_space`` is discrete, action_mask needs                     to be provided to mask illegal actions.\n        Returns:\n            - outputs (:obj:`Dict`): The output dict of the forward computation graph for actor, including ``logit``.\n        ReturnsKeys:\n            - logit (:obj:`torch.Tensor`): The predicted action logit tensor, for discrete action space, it will be                 the same dimension real-value ranged tensor of possible action choices, and for continuous action                 space, it will be the mu and sigma of the Gaussian distribution, and the number of mu and sigma is the                 same as the number of continuous actions.\n        Shapes:\n            - logit (:obj:`torch.FloatTensor`): :math:`(B, M, N)`, where B is batch size and N is ``action_shape``               and M is ``agent_num``.\n\n        Examples:\n            >>> model = MAVAC(agent_obs_shape=64, global_obs_shape=128, action_shape=14)\n            >>> inputs = {\n                    'agent_state': torch.randn(10, 8, 64),\n                    'global_state': torch.randn(10, 8, 128),\n                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))\n                }\n            >>> actor_outputs = model(inputs,'compute_actor')\n            >>> assert actor_outputs['logit'].shape == torch.Size([10, 8, 14])\n\n        "
        if self.action_space == 'discrete':
            action_mask = x['action_mask']
            x = x['agent_state']
            x = self.actor_encoder(x)
            x = self.actor_head(x)
            logit = x['logit']
            logit[action_mask == 0.0] = -99999999
        elif self.action_space == 'continuous':
            x = x['agent_state']
            x = self.actor_encoder(x)
            x = self.actor_head(x)
            logit = x
        return {'logit': logit}

    def compute_critic(self, x: Dict) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            MAVAC forward computation graph for critic part.             Predict state value with global observation tensor in ``x``.\n        Arguments:\n            - x (:obj:`Dict`): Input data dict with keys ['global_state'].\n                - global_state: (:obj:`torch.Tensor`): Global state(obs).\n        Returns:\n            - outputs (:obj:`Dict`): The output dict of MAVAC's forward computation graph for critic,                 including ``value``.\n        ReturnsKeys:\n            - value (:obj:`torch.Tensor`): The predicted state value tensor.\n        Shapes:\n            - value (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch size and M is ``agent_num``.\n\n        Examples:\n            >>> model = MAVAC(agent_obs_shape=64, global_obs_shape=128, action_shape=14)\n            >>> inputs = {\n                    'agent_state': torch.randn(10, 8, 64),\n                    'global_state': torch.randn(10, 8, 128),\n                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))\n                }\n            >>> critic_outputs = model(inputs,'compute_critic')\n            >>> assert critic_outputs['value'].shape == torch.Size([10, 8])\n        "
        x = self.critic_encoder(x['global_state'])
        x = self.critic_head(x)
        return {'value': x['pred']}

    def compute_actor_critic(self, x: Dict) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            MAVAC forward computation graph for both actor and critic part, input observation to predict action             logit and state value.\n        Arguments:\n            - x (:obj:`Dict`): The input dict contains ``agent_state``, ``global_state`` and other related info.\n        Returns:\n            - outputs (:obj:`Dict`): The output dict of MAVAC's forward computation graph for both actor and critic,                 including ``logit`` and ``value``.\n        ReturnsKeys:\n            - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.\n            - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.\n        Shapes:\n            - logit (:obj:`torch.FloatTensor`): :math:`(B, M, N)`, where B is batch size and N is ``action_shape``               and M is ``agent_num``.\n            - value (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch sizeand M is ``agent_num``.\n\n        Examples:\n            >>> model = MAVAC(64, 64)\n            >>> inputs = {\n                    'agent_state': torch.randn(10, 8, 64),\n                    'global_state': torch.randn(10, 8, 128),\n                    'action_mask': torch.randint(0, 2, size=(10, 8, 14))\n                }\n            >>> outputs = model(inputs,'compute_actor_critic')\n            >>> assert outputs['value'].shape == torch.Size([10, 8])\n            >>> assert outputs['logit'].shape == torch.Size([10, 8, 14])\n        "
        logit = self.compute_actor(x)['logit']
        value = self.compute_critic(x)['value']
        return {'logit': logit, 'value': value}
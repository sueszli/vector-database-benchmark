from typing import Union, Dict, Optional
import torch
import torch.nn as nn
from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import RegressionHead, ReparameterizationHead, DistributionHead

@MODEL_REGISTRY.register('qac_dist')
class QACDIST(nn.Module):
    """
    Overview:
        The QAC model with distributional Q-value.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(self, obs_shape: Union[int, SequenceType], action_shape: Union[int, SequenceType], action_space: str='regression', critic_head_type: str='categorical', actor_head_hidden_size: int=64, actor_head_layer_num: int=1, critic_head_hidden_size: int=64, critic_head_layer_num: int=1, activation: Optional[nn.Module]=nn.ReLU(), norm_type: Optional[str]=None, v_min: Optional[float]=-10, v_max: Optional[float]=10, n_atom: Optional[int]=51) -> None:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Init the QAC Distributional Model according to arguments.\n        Arguments:\n            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.\n            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.\n            - action_space (:obj:`str`): Whether choose ``regression`` or ``reparameterization``.\n            - critic_head_type (:obj:`str`): Only ``categorical``.\n            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.\n            - actor_head_layer_num (:obj:`int`):\n                The num of layers used in the network to compute Q value output for actor's nn.\n            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.\n            - critic_head_layer_num (:obj:`int`):\n                The num of layers used in the network to compute Q value output for critic's nn.\n            - activation (:obj:`Optional[nn.Module]`):\n                The type of activation function to use in ``MLP`` the after ``layer_fn``,\n                if ``None`` then default set to ``nn.ReLU()``\n            - norm_type (:obj:`Optional[str]`):\n                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details.\n            - v_min (:obj:`int`): Value of the smallest atom\n            - v_max (:obj:`int`): Value of the largest atom\n            - n_atom (:obj:`int`): Number of atoms in the support\n        "
        super(QACDIST, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape: int = squeeze(action_shape)
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization']
        if self.action_space == 'regression':
            self.actor = nn.Sequential(nn.Linear(obs_shape, actor_head_hidden_size), activation, RegressionHead(actor_head_hidden_size, action_shape, actor_head_layer_num, final_tanh=True, activation=activation, norm_type=norm_type))
        elif self.action_space == 'reparameterization':
            self.actor = nn.Sequential(nn.Linear(obs_shape, actor_head_hidden_size), activation, ReparameterizationHead(actor_head_hidden_size, action_shape, actor_head_layer_num, sigma_type='conditioned', activation=activation, norm_type=norm_type))
        self.critic_head_type = critic_head_type
        assert self.critic_head_type in ['categorical'], self.critic_head_type
        if self.critic_head_type == 'categorical':
            self.critic = nn.Sequential(nn.Linear(obs_shape + action_shape, critic_head_hidden_size), activation, DistributionHead(critic_head_hidden_size, 1, critic_head_layer_num, n_atom=n_atom, v_min=v_min, v_max=v_max, activation=activation, norm_type=norm_type))

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        if False:
            return 10
        "\n        Overview:\n            Use observation and action tensor to predict output.\n            Parameter updates with QACDIST's MLPs forward setup.\n        Arguments:\n            Forward with ``'compute_actor'``:\n                - inputs (:obj:`torch.Tensor`):\n                    The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.\n                    Whether ``actor_head_hidden_size`` or ``critic_head_hidden_size`` depend on ``mode``.\n\n            Forward with ``'compute_critic'``, inputs (`Dict`) Necessary Keys:\n                - ``obs``, ``action`` encoded tensors.\n\n            - mode (:obj:`str`): Name of the forward mode.\n        Returns:\n            - outputs (:obj:`Dict`): Outputs of network forward.\n\n                Forward with ``'compute_actor'``, Necessary Keys (either):\n                    - action (:obj:`torch.Tensor`): Action tensor with same size as input ``x``.\n                    - logit (:obj:`torch.Tensor`):\n                        Logit tensor encoding ``mu`` and ``sigma``, both with same size as input ``x``.\n\n                Forward with ``'compute_critic'``, Necessary Keys:\n                    - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.\n                    - distribution (:obj:`torch.Tensor`): Q value distribution tensor.\n        Actor Shapes:\n            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``\n            - action (:obj:`torch.Tensor`): :math:`(B, N0)`\n            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.\n\n        Critic Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``\n            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is``action_shape``\n            - q_value (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``\n            - distribution (:obj:`torch.FloatTensor`): :math:`(B, 1, N3)`, where B is batch size and N3 is ``num_atom``\n\n        Actor Examples:\n            >>> # Regression mode\n            >>> model = QACDIST(64, 64, 'regression')\n            >>> inputs = torch.randn(4, 64)\n            >>> actor_outputs = model(inputs,'compute_actor')\n            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])\n            >>> # Reparameterization Mode\n            >>> model = QACDIST(64, 64, 'reparameterization')\n            >>> inputs = torch.randn(4, 64)\n            >>> actor_outputs = model(inputs,'compute_actor')\n            >>> actor_outputs['logit'][0].shape # mu\n            >>> torch.Size([4, 64])\n            >>> actor_outputs['logit'][1].shape # sigma\n            >>> torch.Size([4, 64])\n\n        Critic Examples:\n            >>> # Categorical mode\n            >>> inputs = {'obs': torch.randn(4,N), 'action': torch.randn(4,1)}\n            >>> model = QACDIST(obs_shape=(N, ),action_shape=1,action_space='regression',             ...                 critic_head_type='categorical', n_atoms=51)\n            >>> q_value = model(inputs, mode='compute_critic') # q value\n            >>> assert q_value['q_value'].shape == torch.Size([4, 1])\n            >>> assert q_value['distribution'].shape == torch.Size([4, 1, 51])\n        "
        assert mode in self.mode, 'not support forward mode: {}/{}'.format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: torch.Tensor) -> Dict:
        if False:
            return 10
        "\n        Overview:\n            Use encoded embedding tensor to predict output.\n            Execute parameter updates with ``'compute_actor'`` mode\n            Use encoded embedding tensor to predict output.\n        Arguments:\n            - inputs (:obj:`torch.Tensor`):\n                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.\n                ``hidden_size = actor_head_hidden_size``\n            - mode (:obj:`str`): Name of the forward mode.\n        Returns:\n            - outputs (:obj:`Dict`): Outputs of forward pass encoder and head.\n\n        ReturnsKeys (either):\n            - action (:obj:`torch.Tensor`): Continuous action tensor with same size as ``action_shape``.\n            - logit (:obj:`torch.Tensor`):\n                Logit tensor encoding ``mu`` and ``sigma``, both with same size as input ``x``.\n        Shapes:\n            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``\n            - action (:obj:`torch.Tensor`): :math:`(B, N0)`\n            - logit (:obj:`list`): 2 elements, mu and sigma, each is the shape of :math:`(B, N0)`.\n            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, B is batch size.\n        Examples:\n            >>> # Regression mode\n            >>> model = QACDIST(64, 64, 'regression')\n            >>> inputs = torch.randn(4, 64)\n            >>> actor_outputs = model(inputs,'compute_actor')\n            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])\n            >>> # Reparameterization Mode\n            >>> model = QACDIST(64, 64, 'reparameterization')\n            >>> inputs = torch.randn(4, 64)\n            >>> actor_outputs = model(inputs,'compute_actor')\n            >>> actor_outputs['logit'][0].shape # mu\n            >>> torch.Size([4, 64])\n            >>> actor_outputs['logit'][1].shape # sigma\n            >>> torch.Size([4, 64])\n        "
        x = self.actor(inputs)
        if self.action_space == 'regression':
            return {'action': x['pred']}
        elif self.action_space == 'reparameterization':
            return {'logit': [x['mu'], x['sigma']]}

    def compute_critic(self, inputs: Dict) -> Dict:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Execute parameter updates with ``'compute_critic'`` mode\n            Use encoded embedding tensor to predict output.\n        Arguments:\n            - ``obs``, ``action`` encoded tensors.\n            - mode (:obj:`str`): Name of the forward mode.\n        Returns:\n            - outputs (:obj:`Dict`): Q-value output and distribution.\n\n        ReturnKeys:\n            - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.\n            - distribution (:obj:`torch.Tensor`): Q value distribution tensor.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``\n            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is``action_shape``\n            - q_value (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``\n            - distribution (:obj:`torch.FloatTensor`): :math:`(B, 1, N3)`, where B is batch size and N3 is ``num_atom``\n\n        Examples:\n            >>> # Categorical mode\n            >>> inputs = {'obs': torch.randn(4,N), 'action': torch.randn(4,1)}\n            >>> model = QACDIST(obs_shape=(N, ),action_shape=1,action_space='regression',             ...                 critic_head_type='categorical', n_atoms=51)\n            >>> q_value = model(inputs, mode='compute_critic') # q value\n            >>> assert q_value['q_value'].shape == torch.Size([4, 1])\n            >>> assert q_value['distribution'].shape == torch.Size([4, 1, 51])\n        "
        (obs, action) = (inputs['obs'], inputs['action'])
        assert len(obs.shape) == 2
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        x = torch.cat([obs, action], dim=1)
        x = self.critic(x)
        return {'q_value': x['logit'], 'distribution': x['distribution']}
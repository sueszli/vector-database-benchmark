import torch
import torch.nn as nn
from typing import Union, Optional, Dict
import numpy as np
from ding.model.common.head import DiscreteHead, RegressionHead, ReparameterizationHead
from ding.utils import SequenceType, squeeze
from ding.model.common.encoder import FCEncoder, ConvEncoder
from torch.distributions import Independent, Normal

class InverseDynamicsModel(nn.Module):
    """
    InverseDynamicsModel: infering missing action information from state transition.
    input and output: given pair of observation, return action (s0,s1 --> a0 if n=2)
    """

    def __init__(self, obs_shape: Union[int, SequenceType], action_shape: Union[int, SequenceType], encoder_hidden_size_list: SequenceType=[60, 80, 100, 40], action_space: str='regression', activation: Optional[nn.Module]=nn.LeakyReLU(), norm_type: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Init the Inverse Dynamics (encoder + head) Model according to input arguments.\n        Arguments:\n            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].\n            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].\n            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \\\n                the last element must match ``head_hidden_size``.\n            - action_space (:obj:`String`): Action space, such as 'regression', 'reparameterization', 'discrete'.\n            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \\\n                if ``None`` then default set it to ``nn.LeakyReLU()`` refer to https://arxiv.org/abs/1805.01954\n            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \\\n                ``ding.torch_utils.fc_block`` for more details.\n        "
        super(InverseDynamicsModel, self).__init__()
        (obs_shape, action_shape) = (squeeze(obs_shape), squeeze(action_shape))
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape * 2, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        elif len(obs_shape) == 3:
            obs_shape = (obs_shape[0] * 2, *obs_shape[1:])
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError('not support obs_shape for pre-defined encoder: {}, please customize your own Model'.format(obs_shape))
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization', 'discrete'], 'not supported action_space: {}'.format(self.action_space)
        if self.action_space == 'regression':
            self.header = RegressionHead(encoder_hidden_size_list[-1], action_shape, final_tanh=False, activation=activation, norm_type=norm_type)
        elif self.action_space == 'reparameterization':
            self.header = ReparameterizationHead(encoder_hidden_size_list[-1], action_shape, sigma_type='conditioned', activation=activation, norm_type=norm_type)
        elif self.action_space == 'discrete':
            self.header = DiscreteHead(encoder_hidden_size_list[-1], action_shape, activation=activation, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> Dict:
        if False:
            return 10
        if self.action_space == 'regression':
            x = self.encoder(x)
            x = self.header(x)
            return {'action': x['pred']}
        elif self.action_space == 'reparameterization':
            x = self.encoder(x)
            x = self.header(x)
            (mu, sigma) = (x['mu'], x['sigma'])
            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            action = torch.tanh(pred)
            return {'logit': [mu, sigma], 'action': action}
        elif self.action_space == 'discrete':
            x = self.encoder(x)
            x = self.header(x)
            return x

    def predict_action(self, x: torch.Tensor) -> Dict:
        if False:
            print('Hello World!')
        if self.action_space == 'discrete':
            res = nn.Softmax(dim=-1)
            action = torch.argmax(res(self.forward(x)['logit']), -1)
            return {'action': action}
        else:
            return self.forward(x)

    def train(self, training_set: dict, n_epoch: int, learning_rate: float, weight_decay: float):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Train idm model, given pair of states return action (s_t,s_t+1,a_t)\n\n        Arguments:\n            - training_set (:obj:`dict`):states transition\n            - n_epoch (:obj:`int`): number of epoches\n            - learning_rate (:obj:`float`): learning rate for optimizer\n            - weight_decay (:obj:`float`): weight decay for optimizer\n        '
        if self.action_space == 'discrete':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_list = []
        for itr in range(n_epoch):
            data = training_set['obs']
            y = training_set['action']
            if self.action_space == 'discrete':
                y_pred = self.forward(data)['logit']
            else:
                y_pred = self.forward(data)['action']
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss = np.mean(loss_list)
        return loss
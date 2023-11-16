from typing import List, Dict, Any
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from ding.utils import REWARD_MODEL_REGISTRY
from ding.utils.data import default_collate
from .base_reward_model import BaseRewardModel

class GuidedCostNN(nn.Module):

    def __init__(self, input_size, hidden_size=128, output_size=1):
        if False:
            i = 10
            return i + 15
        super(GuidedCostNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.net(x)

@REWARD_MODEL_REGISTRY.register('guided_cost')
class GuidedCostRewardModel(BaseRewardModel):
    """
    Overview:
        Policy class of Guided cost algorithm. (https://arxiv.org/pdf/1603.00448.pdf)
    Interface:
        ``estimate``, ``train``, ``collect_data``, ``clear_date``,         ``__init__``,  ``state_dict``, ``load_state_dict``, ``learn``        ``state_dict_reward_model``, ``load_state_dict_reward_model``
    Config:
        == ====================  ========   =============  ========================================  ================
        ID Symbol                Type       Default Value  Description                               Other(Shape)
        == ====================  ========   =============  ========================================  ================
        1  ``type``              str         guided_cost   | Reward model register name, refer        |
                                                           | to registry ``REWARD_MODEL_REGISTRY``    |
        2  | ``continuous``      bool        True          | Whether action is continuous             |
        3  | ``learning_rate``   float       0.001         | learning rate for optimizer              |
        4  | ``update_per_``     int         100           | Number of updates per collect            |
           | ``collect``                                   |                                          |
        5  | ``batch_size``      int         64            | Training batch size                      |
        6  | ``hidden_size``     int         128           | Linear model hidden size                 |
        7  | ``action_shape``    int         1             | Action space shape                       |
        8  | ``log_every_n``     int         50            | add loss to log every n iteration        |
           | ``_train``                                    |                                          |
        9  | ``store_model_``    int         100           | save model every n iteration             |
           | ``every_n_train``                                                                        |
        == ====================  ========   =============  ========================================  ================

    """
    config = dict(type='guided_cost', learning_rate=0.001, action_shape=1, continuous=True, batch_size=64, hidden_size=128, update_per_collect=100, log_every_n_train=50, store_model_every_n_train=100)

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:
        if False:
            for i in range(10):
                print('nop')
        super(GuidedCostRewardModel, self).__init__()
        self.cfg = config
        self.action_shape = self.cfg.action_shape
        assert device == 'cpu' or device.startswith('cuda')
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = GuidedCostNN(config.input_size, config.hidden_size)
        self.reward_model.to(self.device)
        self.opt = optim.Adam(self.reward_model.parameters(), lr=config.learning_rate)

    def train(self, expert_demo: torch.Tensor, samp: torch.Tensor, iter, step):
        if False:
            while True:
                i = 10
        device_0 = expert_demo[0]['obs'].device
        device_1 = samp[0]['obs'].device
        for i in range(len(expert_demo)):
            expert_demo[i]['prob'] = torch.FloatTensor([1]).to(device_0)
        if self.cfg.continuous:
            for i in range(len(samp)):
                (mu, sigma) = samp[i]['logit']
                dist = Independent(Normal(mu, sigma), 1)
                next_action = samp[i]['action']
                log_prob = dist.log_prob(next_action)
                samp[i]['prob'] = torch.exp(log_prob).unsqueeze(0).to(device_1)
        else:
            for i in range(len(samp)):
                probs = F.softmax(samp[i]['logit'], dim=-1)
                prob = probs[samp[i]['action']]
                samp[i]['prob'] = prob.to(device_1)
        samp.extend(expert_demo)
        expert_demo = default_collate(expert_demo)
        samp = default_collate(samp)
        cost_demo = self.reward_model(torch.cat([expert_demo['obs'], expert_demo['action'].float().reshape(-1, self.action_shape)], dim=-1))
        cost_samp = self.reward_model(torch.cat([samp['obs'], samp['action'].float().reshape(-1, self.action_shape)], dim=-1))
        prob = samp['prob'].unsqueeze(-1)
        loss_IOC = torch.mean(cost_demo) + torch.log(torch.mean(torch.exp(-cost_samp) / (prob + 1e-07)))
        self.opt.zero_grad()
        loss_IOC.backward()
        self.opt.step()
        if iter % self.cfg.log_every_n_train == 0:
            self.tb_logger.add_scalar('reward_model/loss_iter', loss_IOC, iter)
            self.tb_logger.add_scalar('reward_model/loss_step', loss_IOC, step)

    def estimate(self, data: list) -> List[Dict]:
        if False:
            return 10
        train_data_augmented = data
        for i in range(len(train_data_augmented)):
            with torch.no_grad():
                reward = self.reward_model(torch.cat([train_data_augmented[i]['obs'], train_data_augmented[i]['action'].float()]).unsqueeze(0)).squeeze(0)
                train_data_augmented[i]['reward'] = -reward
        return train_data_augmented

    def collect_data(self, data) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Collecting training data, not implemented if reward model (i.e. online_net) is only trained ones,                 if online_net is trained continuously, there should be some implementations in collect_data method\n        '
        pass

    def clear_data(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Collecting clearing data, not implemented if reward model (i.e. online_net) is only trained ones,                 if online_net is trained continuously, there should be some implementations in clear_data method\n        '
        pass

    def state_dict_reward_model(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return {'model': self.reward_model.state_dict(), 'optimizer': self.opt.state_dict()}

    def load_state_dict_reward_model(self, state_dict: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        self.reward_model.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['optimizer'])
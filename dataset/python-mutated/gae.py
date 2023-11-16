from collections import namedtuple
import torch
from ding.hpc_rl import hpc_wrapper
gae_data = namedtuple('gae_data', ['value', 'next_value', 'reward', 'done', 'traj_flag'])

def shape_fn_gae(args, kwargs):
    if False:
        return 10
    '\n    Overview:\n        Return shape of gae for hpc\n    Returns:\n        shape: [T, B]\n    '
    if len(args) <= 0:
        tmp = kwargs['data'].reward.shape
    else:
        tmp = args[0].reward.shape
    return tmp

@hpc_wrapper(shape_fn=shape_fn_gae, namedtuple_data=True, include_args=[0, 1, 2], include_kwargs=['data', 'gamma', 'lambda_'])
def gae(data: namedtuple, gamma: float=0.99, lambda_: float=0.97) -> torch.FloatTensor:
    if False:
        print('Hello World!')
    "\n    Overview:\n        Implementation of Generalized Advantage Estimator (arXiv:1506.02438)\n    Arguments:\n        - data (:obj:`namedtuple`): gae input data with fields ['value', 'reward'], which contains some episodes or             trajectories data.\n        - gamma (:obj:`float`): the future discount factor, should be in [0, 1], defaults to 0.99.\n        - lambda (:obj:`float`): the gae parameter lambda, should be in [0, 1], defaults to 0.97, when lambda -> 0,             it induces bias, but when lambda -> 1, it has high variance due to the sum of terms.\n    Returns:\n        - adv (:obj:`torch.FloatTensor`): the calculated advantage\n    Shapes:\n        - value (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is trajectory length and B is batch size\n        - next_value (:obj:`torch.FloatTensor`): :math:`(T, B)`\n        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`\n        - adv (:obj:`torch.FloatTensor`): :math:`(T, B)`\n    Examples:\n        >>> value = torch.randn(2, 3)\n        >>> next_value = torch.randn(2, 3)\n        >>> reward = torch.randn(2, 3)\n        >>> data = gae_data(value, next_value, reward, None, None)\n        >>> adv = gae(data)\n    "
    (value, next_value, reward, done, traj_flag) = data
    if done is None:
        done = torch.zeros_like(reward, device=reward.device)
    if traj_flag is None:
        traj_flag = done
    done = done.float()
    traj_flag = traj_flag.float()
    if len(value.shape) == len(reward.shape) + 1:
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        traj_flag = traj_flag.unsqueeze(-1)
    next_value *= 1 - done
    delta = reward + gamma * next_value - value
    factor = gamma * lambda_ * (1 - traj_flag)
    adv = torch.zeros_like(value)
    gae_item = torch.zeros_like(value[0])
    for t in reversed(range(reward.shape[0])):
        gae_item = delta[t] + factor[t] * gae_item
        adv[t] = gae_item
    return adv
import torch
import torch.nn.functional as F
from collections import namedtuple
from ding.rl_utils.isw import compute_importance_weights

def compute_q_retraces(q_values: torch.Tensor, v_pred: torch.Tensor, rewards: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor, ratio: torch.Tensor, gamma: float=0.9) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    "\n    Shapes:\n        - q_values (:obj:`torch.Tensor`): :math:`(T + 1, B, N)`, where T is unroll_len, B is batch size, N is discrete             action dim.\n        - v_pred (:obj:`torch.Tensor`): :math:`(T + 1, B, 1)`\n        - rewards (:obj:`torch.Tensor`): :math:`(T, B)`\n        - actions (:obj:`torch.Tensor`): :math:`(T, B)`\n        - weights (:obj:`torch.Tensor`): :math:`(T, B)`\n        - ratio (:obj:`torch.Tensor`): :math:`(T, B, N)`\n        - q_retraces (:obj:`torch.Tensor`): :math:`(T + 1, B, 1)`\n    Examples:\n        >>> T=2\n        >>> B=3\n        >>> N=4\n        >>> q_values=torch.randn(T+1, B, N)\n        >>> v_pred=torch.randn(T+1, B, 1)\n        >>> rewards=torch.randn(T, B)\n        >>> actions=torch.randint(0, N, (T, B))\n        >>> weights=torch.ones(T, B)\n        >>> ratio=torch.randn(T, B, N)\n        >>> q_retraces = compute_q_retraces(q_values, v_pred, rewards, actions, weights, ratio)\n\n    .. note::\n        q_retrace operation doesn't need to compute gradient, just executes forward computation.\n    "
    T = q_values.size()[0] - 1
    rewards = rewards.unsqueeze(-1)
    actions = actions.unsqueeze(-1)
    weights = weights.unsqueeze(-1)
    q_retraces = torch.zeros_like(v_pred)
    tmp_retraces = v_pred[-1]
    q_retraces[-1] = v_pred[-1]
    q_gather = torch.zeros_like(v_pred)
    q_gather[0:-1] = q_values[0:-1].gather(-1, actions)
    ratio_gather = ratio.gather(-1, actions)
    for idx in reversed(range(T)):
        q_retraces[idx] = rewards[idx] + gamma * weights[idx] * tmp_retraces
        tmp_retraces = ratio_gather[idx].clamp(max=1.0) * (q_retraces[idx] - q_gather[idx]) + v_pred[idx]
    return q_retraces
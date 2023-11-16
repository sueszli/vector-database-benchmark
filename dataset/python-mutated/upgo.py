import torch
import torch.nn.functional as F
from ding.hpc_rl import hpc_wrapper
from .td import generalized_lambda_returns

def tb_cross_entropy(logit, label, mask=None):
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Compute the cross entropy loss for label and logit, with mask support\n    Arguments:\n        - logit (:obj:`torch.Tensor`): the logit tensor, of size [T, B, N] or [T, B, N, N2]\n        - label (:obj:`torch.Tensor`): the label tensor, of size [T, B] or [T, B, N2]\n        - mask (:obj:`torch.Tensor` or :obj:`None`): the mask tensor, of size [T, B] or [T, B, N2]\n    Returns:\n        - ce (:obj:`torch.Tensor`): the computed cross entropy, of size [T, B]\n    Examples:\n        >>> T, B, N, N2 = 4, 8, 5, 7\n        >>> logit = torch.randn(T, B, N, N2).softmax(-1).requires_grad_(True)\n        >>> action = logit.argmax(-1).detach()\n        >>> ce = tb_cross_entropy(logit, action)\n    '
    assert len(label.shape) >= 2
    (T, B) = label.shape[:2]
    if len(label.shape) > 2:
        assert len(label.shape) == 3
        (s, n) = logit.shape[-2:]
        logit = logit.reshape(-1, n)
        label = label.reshape(-1)
        ce = -F.cross_entropy(logit, label, reduction='none')
        ce = ce.view(T * B, -1)
        if mask is not None:
            ce *= mask.reshape(-1, s)
        ce = ce.sum(dim=1)
        ce = ce.reshape(T, B)
    else:
        label = label.reshape(-1)
        logit = logit.reshape(-1, logit.shape[-1])
        ce = -F.cross_entropy(logit, label, reduction='none')
        ce = ce.reshape(T, B, -1)
        ce = ce.mean(dim=2)
    return ce

def upgo_returns(rewards: torch.Tensor, bootstrap_values: torch.Tensor) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        Computing UPGO return targets. Also notice there is no special handling for the terminal state.\n    Arguments:\n        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, \\\n            of size [T_traj, batchsize]\n        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T, \\\n            of size [T_traj+1, batchsize]\n    Returns:\n        - ret (:obj:`torch.Tensor`): Computed lambda return value for each state from 0 to T-1, \\\n            of size [T_traj, batchsize]\n    Examples:\n        >>> T, B, N, N2 = 4, 8, 5, 7\n        >>> rewards = torch.randn(T, B)\n        >>> bootstrap_values = torch.randn(T + 1, B).requires_grad_(True)\n        >>> returns = upgo_returns(rewards, bootstrap_values)\n    '
    lambdas = rewards + bootstrap_values[1:] >= bootstrap_values[:-1]
    lambdas = torch.cat([lambdas[1:], torch.ones_like(lambdas[-1:])], dim=0)
    return generalized_lambda_returns(bootstrap_values, rewards, 1.0, lambdas)

@hpc_wrapper(shape_fn=lambda args: args[0].shape, namedtuple_data=True, include_args=5, include_kwargs=['target_output', 'rhos', 'action', 'rewards', 'bootstrap_values'])
def upgo_loss(target_output: torch.Tensor, rhos: torch.Tensor, action: torch.Tensor, rewards: torch.Tensor, bootstrap_values: torch.Tensor, mask=None) -> torch.Tensor:
    if False:
        return 10
    '\n    Overview:\n        Computing UPGO loss given constant gamma and lambda. There is no special handling for terminal state value,\n        if the last state in trajectory is the terminal, just pass a 0 as bootstrap_terminal_value.\n    Arguments:\n        - target_output (:obj:`torch.Tensor`): the output computed by the target policy network, \\\n            of size [T_traj, batchsize, n_output]\n        - rhos (:obj:`torch.Tensor`): the importance sampling ratio, of size [T_traj, batchsize]\n        - action (:obj:`torch.Tensor`): the action taken, of size [T_traj, batchsize]\n        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]\n        - bootstrap_values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T, \\\n            of size [T_traj+1, batchsize]\n    Returns:\n        - loss (:obj:`torch.Tensor`): Computed importance sampled UPGO loss, averaged over the samples, of size []\n    Examples:\n        >>> T, B, N, N2 = 4, 8, 5, 7\n        >>> rhos = torch.randn(T, B)\n        >>> loss = upgo_loss(logit, rhos, action, rewards, bootstrap_values)\n    '
    with torch.no_grad():
        returns = upgo_returns(rewards, bootstrap_values)
        advantages = rhos * (returns - bootstrap_values[:-1])
    metric = tb_cross_entropy(target_output, action, mask)
    assert metric.shape == action.shape[:2]
    losses = advantages * metric
    return -losses.mean()
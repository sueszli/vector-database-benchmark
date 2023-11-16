import numpy as np
from ray.rllib.utils.framework import try_import_torch
(torch, nn) = try_import_torch()

class VDNMixer(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        if False:
            i = 10
            return i + 15
        return torch.sum(agent_qs, dim=2, keepdim=True)

class QMixer(nn.Module):

    def __init__(self, n_agents, state_shape, mixing_embed_dim):
        if False:
            return 10
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim
        self.state_dim = int(np.prod(state_shape))
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        if False:
            return 10
        'Forward pass for the mixer.\n\n        Args:\n            agent_qs: Tensor of shape [B, T, n_agents, n_actions]\n            states: Tensor of shape [B, T, state_dim]\n        '
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = nn.functional.elu(torch.bmm(agent_qs, w1) + b1)
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)
        y = torch.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1, 1)
        return q_tot
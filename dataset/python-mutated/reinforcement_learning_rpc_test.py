import numpy as np
from itertools import count
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
TOTAL_EPISODE_STEP = 5000
GAMMA = 0.1
SEED = 543

def _call_method(method, rref, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    a helper function to call a method on the given RRef\n    '
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    if False:
        return 10
    '\n    a helper function to run method on the owner of rref and fetch back the\n    result using RPC\n    '
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

class Policy(nn.Module):
    """
    Borrowing the ``Policy`` class from the Reinforcement Learning example.
    Copying the code to make these two examples independent.
    See https://github.com/pytorch/examples/tree/master/reinforcement_learning
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class DummyEnv:
    """
    A dummy environment that implements the required subset of the OpenAI gym
    interface. It exists only to avoid a dependency on gym for running the
    tests in this file. It is designed to run for a set max number of iterations,
    returning random states and rewards at each step.
    """

    def __init__(self, state_dim=4, num_iters=10, reward_threshold=475.0):
        if False:
            i = 10
            return i + 15
        self.state_dim = state_dim
        self.num_iters = num_iters
        self.iter = 0
        self.reward_threshold = reward_threshold

    def seed(self, manual_seed):
        if False:
            while True:
                i = 10
        torch.manual_seed(manual_seed)

    def reset(self):
        if False:
            print('Hello World!')
        self.iter = 0
        return torch.randn(self.state_dim)

    def step(self, action):
        if False:
            i = 10
            return i + 15
        self.iter += 1
        state = torch.randn(self.state_dim)
        reward = torch.rand(1).item() * self.reward_threshold
        done = self.iter >= self.num_iters
        info = {}
        return (state, reward, done, info)

class Observer:
    """
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.id = rpc.get_worker_info().id
        self.env = DummyEnv()
        self.env.seed(SEED)

    def run_episode(self, agent_rref, n_steps):
        if False:
            i = 10
            return i + 15
        '\n        Run one episode of n_steps.\n        Arguments:\n            agent_rref (RRef): an RRef referencing the agent object.\n            n_steps (int): number of steps in this episode\n        '
        (state, ep_reward) = (self.env.reset(), 0)
        for step in range(n_steps):
            action = _remote_method(Agent.select_action, agent_rref, self.id, state)
            (state, reward, done, _) = self.env.step(action)
            _remote_method(Agent.report_reward, agent_rref, self.id, reward)
            if done:
                break

class Agent:

    def __init__(self, world_size):
        if False:
            i = 10
            return i + 15
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = DummyEnv().reward_threshold
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(worker_name(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

    def select_action(self, ob_id, state):
        if False:
            i = 10
            return i + 15
        '\n        This function is mostly borrowed from the Reinforcement Learning example.\n        See https://github.com/pytorch/examples/tree/master/reinforcement_learning\n        The main difference is that instead of keeping all probs in one list,\n        the agent keeps probs in a dictionary, one key per observer.\n\n        NB: no need to enforce thread-safety here as GIL will serialize\n        executions.\n        '
        probs = self.policy(state.unsqueeze(0))
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        if False:
            print('Hello World!')
        '\n        Observers call this function to report rewards.\n        '
        self.rewards[ob_id].append(reward)

    def run_episode(self, n_steps=0):
        if False:
            print('Hello World!')
        '\n        Run one episode. The agent will tell each observer to run n_steps.\n        '
        futs = []
        for ob_rref in self.ob_rrefs:
            futs.append(rpc_async(ob_rref.owner(), _call_method, args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps)))
        for fut in futs:
            fut.wait()

    def finish_episode(self):
        if False:
            return 10
        '\n        This function is mostly borrowed from the Reinforcement Learning example.\n        See https://github.com/pytorch/examples/tree/master/reinforcement_learning\n        The main difference is that it joins all probs and rewards from\n        different observers into one list, and uses the minimum observer rewards\n        as the reward of the current episode.\n        '
        (R, probs, rewards) = (0, [], [])
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []
        (policy_loss, returns) = ([], [])
        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for (log_prob, R) in zip(probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        return min_reward

def run_agent(agent, n_steps):
    if False:
        i = 10
        return i + 15
    for i_episode in count(1):
        agent.run_episode(n_steps=n_steps)
        last_reward = agent.finish_episode()
        if agent.running_reward > agent.reward_threshold:
            print(f'Solved! Running reward is now {agent.running_reward}!')
            break

class ReinforcementLearningRpcTest(RpcAgentTestFixture):

    @dist_init(setup_rpc=False)
    def test_rl_rpc(self):
        if False:
            while True:
                i = 10
        if self.rank == 0:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
            agent = Agent(self.world_size)
            run_agent(agent, n_steps=int(TOTAL_EPISODE_STEP / (self.world_size - 1)))
            self.assertGreater(agent.running_reward, 0.0)
        else:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        rpc.shutdown()
import argparse
import os
import random
import time
from distutils.util import strtobool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip('.py'), help='the name of this experiment')
    parser.add_argument('--seed', type=int, default=1, help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default='cleanRL', help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--env-id', type=str, default='Hopper-v4', help='the id of the environment')
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='total timesteps of the experiments')
    parser.add_argument('--buffer-size', type=int, default=int(1000000.0), help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor gamma')
    parser.add_argument('--tau', type=float, default=0.005, help='target smoothing coefficient (default: 0.005)')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size of sample from the reply memory')
    parser.add_argument('--learning-starts', type=int, default=5000.0, help='timestep to start learning')
    parser.add_argument('--policy-lr', type=float, default=0.0003, help='the learning rate of the policy network optimizer')
    parser.add_argument('--q-lr', type=float, default=0.001, help='the learning rate of the Q network network optimizer')
    parser.add_argument('--policy-frequency', type=int, default=2, help='the frequency of training policy (delayed)')
    parser.add_argument('--target-network-frequency', type=int, default=1, help='the frequency of updates for the target nerworks')
    parser.add_argument('--noise-clip', type=float, default=0.5, help='noise clip parameter of the Target Policy Smoothing Regularization')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy regularization coefficient.')
    parser.add_argument('--autotune', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='automatic tuning of the entropy coefficient')
    args = parser.parse_args()
    return args

def make_env(env_id, seed, idx, capture_video, run_name):
    if False:
        while True:
            i = 10

    def thunk():
        if False:
            for i in range(10):
                print('nop')
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

class SoftQNetwork(nn.Module):

    def __init__(self, env):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        if False:
            i = 10
            return i + 15
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):

    def __init__(self, env):
        if False:
            return 10
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.register_buffer('action_scale', torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        if False:
            return 10
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return (mean, log_std)

    def get_action(self, x):
        if False:
            i = 10
            return i + 15
        (mean, log_std) = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-06)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return (action, log_prob, mean)
if __name__ == '__main__':
    import stable_baselines3 as sb3
    if sb3.__version__ < '2.0':
        raise ValueError('Ongoing migration: run the following command to install the new dependencies:\npoetry run pip install "stable_baselines3==2.0.0a1"\n')
    args = parse_args()
    run_name = f'{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}'
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text('hyperparameters', '|param|value|\n|-|-|\n%s' % '\n'.join([f'|{key}|{value}|' for (key, value) in vars(args).items()]))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), 'only continuous action space is supported'
    max_action = float(envs.single_action_space.high[0])
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device, handle_timeout_termination=False)
    start_time = time.time()
    (obs, _) = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            (actions, _, _) = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
        (next_obs, rewards, terminations, truncations, infos) = envs.step(actions)
        if 'final_info' in infos:
            for info in infos['final_info']:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar('charts/episodic_return', info['episode']['r'], global_step)
                writer.add_scalar('charts/episodic_length', info['episode']['l'], global_step)
                break
        real_next_obs = next_obs.copy()
        for (idx, trunc) in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos['final_observation'][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                (next_state_actions, next_state_log_pi, _) = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    (pi, log_pi, _) = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if args.autotune:
                        with torch.no_grad():
                            (_, log_pi, _) = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
            if global_step % args.target_network_frequency == 0:
                for (param, target_param) in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for (param, target_param) in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            if global_step % 100 == 0:
                writer.add_scalar('losses/qf1_values', qf1_a_values.mean().item(), global_step)
                writer.add_scalar('losses/qf2_values', qf2_a_values.mean().item(), global_step)
                writer.add_scalar('losses/qf1_loss', qf1_loss.item(), global_step)
                writer.add_scalar('losses/qf2_loss', qf2_loss.item(), global_step)
                writer.add_scalar('losses/qf_loss', qf_loss.item() / 2.0, global_step)
                writer.add_scalar('losses/actor_loss', actor_loss.item(), global_step)
                writer.add_scalar('losses/alpha', alpha, global_step)
                print('SPS:', int(global_step / (time.time() - start_time)))
                writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar('losses/alpha_loss', alpha_loss.item(), global_step)
    envs.close()
    writer.close()
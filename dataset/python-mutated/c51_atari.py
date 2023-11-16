import argparse
import os
import random
import time
from distutils.util import strtobool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip('.py'), help='the name of this experiment')
    parser.add_argument('--seed', type=int, default=1, help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default='cleanRL', help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--save-model', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='whether to save model into the `runs/{run_name}` folder')
    parser.add_argument('--upload-model', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='whether to upload the saved model to huggingface')
    parser.add_argument('--hf-entity', type=str, default='', help='the user or org name of the model repository from the Hugging Face Hub')
    parser.add_argument('--env-id', type=str, default='BreakoutNoFrameskip-v4', help='the id of the environment')
    parser.add_argument('--total-timesteps', type=int, default=10000000, help='total timesteps of the experiments')
    parser.add_argument('--learning-rate', type=float, default=0.00025, help='the learning rate of the optimizer')
    parser.add_argument('--num-envs', type=int, default=1, help='the number of parallel game environments')
    parser.add_argument('--n-atoms', type=int, default=51, help='the number of atoms')
    parser.add_argument('--v-min', type=float, default=-10, help='the return lower bound')
    parser.add_argument('--v-max', type=float, default=10, help='the return upper bound')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=10000, help='the timesteps it takes to update the target network')
    parser.add_argument('--batch-size', type=int, default=32, help='the batch size of sample from the reply memory')
    parser.add_argument('--start-e', type=float, default=1, help='the starting epsilon for exploration')
    parser.add_argument('--end-e', type=float, default=0.01, help='the ending epsilon for exploration')
    parser.add_argument('--exploration-fraction', type=float, default=0.1, help='the fraction of `total-timesteps` it takes from start-e to go end-e')
    parser.add_argument('--learning-starts', type=int, default=80000, help='timestep to start learning')
    parser.add_argument('--train-frequency', type=int, default=4, help='the frequency of training')
    args = parser.parse_args()
    assert args.num_envs == 1, 'vectorized envs are not supported at the moment'
    return args

def make_env(env_id, seed, idx, capture_video, run_name):
    if False:
        return 10

    def thunk():
        if False:
            return 10
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk

class QNetwork(nn.Module):

    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer('atoms', torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n
        self.network = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, self.n * n_atoms))

    def get_action(self, x, action=None):
        if False:
            while True:
                i = 10
        logits = self.network(x / 255.0)
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return (action, pmfs[torch.arange(len(x)), action])

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    if False:
        i = 10
        return i + 15
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
if __name__ == '__main__':
    import stable_baselines3 as sb3
    if sb3.__version__ < '2.0':
        raise ValueError('Ongoing migration: run the following command to install the new dependencies:\n\npoetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" \n')
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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), 'only discrete action space is supported'
    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())
    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device, optimize_memory_usage=True, handle_timeout_termination=False)
    start_time = time.time()
    (obs, _) = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            (actions, pmf) = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()
        (next_obs, rewards, terminations, truncations, infos) = envs.step(actions)
        if 'final_info' in infos:
            for info in infos['final_info']:
                if 'episode' not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar('charts/episodic_return', info['episode']['r'], global_step)
                writer.add_scalar('charts/episodic_length', info['episode']['l'], global_step)
                writer.add_scalar('charts/epsilon', epsilon, global_step)
                break
        real_next_obs = next_obs.copy()
        for (idx, trunc) in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos['final_observation'][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    (_, next_pmfs) = target_network.get_action(data.next_observations)
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)
                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
                (_, old_pmfs) = q_network.get_action(data.observations, data.actions.flatten())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-05, max=1 - 1e-05).log()).sum(-1)).mean()
                if global_step % 100 == 0:
                    writer.add_scalar('losses/loss', loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar('losses/q_values', old_val.mean().item(), global_step)
                    print('SPS:', int(global_step / (time.time() - start_time)))
                    writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
    if args.save_model:
        model_path = f'runs/{run_name}/{args.exp_name}.cleanrl_model'
        model_data = {'model_weights': q_network.state_dict(), 'args': vars(args)}
        torch.save(model_data, model_path)
        print(f'model saved to {model_path}')
        from cleanrl_utils.evals.c51_eval import evaluate
        episodic_returns = evaluate(model_path, make_env, args.env_id, eval_episodes=10, run_name=f'{run_name}-eval', Model=QNetwork, device=device, epsilon=0.05)
        for (idx, episodic_return) in enumerate(episodic_returns):
            writer.add_scalar('eval/episodic_return', episodic_return, idx)
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub
            repo_name = f'{args.env_id}-{args.exp_name}-seed{args.seed}'
            repo_id = f'{args.hf_entity}/{repo_name}' if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, 'C51', f'runs/{run_name}', f'videos/{run_name}-eval')
    envs.close()
    writer.close()
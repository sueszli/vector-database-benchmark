import argparse
import os
import random
import time
from distutils.util import strtobool
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip('.py'), help='the name of this experiment')
    parser.add_argument('--seed', type=int, default=1, help='seed of the experiment')
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
        while True:
            i = 10

    def thunk():
        if False:
            i = 10
            return i + 15
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
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        if False:
            return 10
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        x = nn.softmax(x, axis=-1)
        return x

class TrainState(TrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray

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
    key = jax.random.PRNGKey(args.seed)
    (key, q_key) = jax.random.split(key, 2)
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), 'only discrete action space is supported'
    (obs, _) = envs.reset(seed=args.seed)
    q_network = QNetwork(action_dim=envs.single_action_space.n, n_atoms=args.n_atoms)
    q_state = TrainState.create(apply_fn=q_network.apply, params=q_network.init(q_key, obs), target_params=q_network.init(q_key, obs), atoms=jnp.asarray(np.linspace(args.v_min, args.v_max, num=args.n_atoms)), tx=optax.adam(learning_rate=args.learning_rate, eps=0.01 / args.batch_size))
    q_network.apply = jax.jit(q_network.apply)
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))
    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, 'cpu', optimize_memory_usage=True, handle_timeout_termination=False)

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        if False:
            return 10
        next_pmfs = q_network.apply(q_state.target_params, next_observations)
        next_vals = (next_pmfs * q_state.atoms).sum(axis=-1)
        next_action = jnp.argmax(next_vals, axis=-1)
        next_pmfs = next_pmfs[np.arange(next_pmfs.shape[0]), next_action]
        next_atoms = rewards + args.gamma * q_state.atoms * (1 - dones)
        delta_z = q_state.atoms[1] - q_state.atoms[0]
        tz = jnp.clip(next_atoms, a_min=args.v_min, a_max=args.v_max)
        b = (tz - args.v_min) / delta_z
        l = jnp.clip(jnp.floor(b), a_min=0, a_max=args.n_atoms - 1)
        u = jnp.clip(jnp.ceil(b), a_min=0, a_max=args.n_atoms - 1)
        d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs
        target_pmfs = jnp.zeros_like(next_pmfs)

        def project_to_bins(i, val):
            if False:
                return 10
            val = val.at[i, l[i].astype(jnp.int32)].add(d_m_l[i])
            val = val.at[i, u[i].astype(jnp.int32)].add(d_m_u[i])
            return val
        target_pmfs = jax.lax.fori_loop(0, target_pmfs.shape[0], project_to_bins, target_pmfs)

        def loss(q_params, observations, actions, target_pmfs):
            if False:
                print('Hello World!')
            pmfs = q_network.apply(q_params, observations)
            old_pmfs = pmfs[np.arange(pmfs.shape[0]), actions.squeeze()]
            old_pmfs_l = jnp.clip(old_pmfs, a_min=1e-05, a_max=1 - 1e-05)
            loss = (-(target_pmfs * jnp.log(old_pmfs_l)).sum(-1)).mean()
            return (loss, (old_pmfs * q_state.atoms).sum(-1))
        ((loss_value, old_values), grads) = jax.value_and_grad(loss, has_aux=True)(q_state.params, observations, actions, target_pmfs)
        q_state = q_state.apply_gradients(grads=grads)
        return (loss_value, old_values, q_state)

    @jax.jit
    def get_action(q_state, obs):
        if False:
            for i in range(10):
                print('nop')
        pmfs = q_network.apply(q_state.params, obs)
        q_vals = (pmfs * q_state.atoms).sum(axis=-1)
        actions = q_vals.argmax(axis=-1)
        return actions
    start_time = time.time()
    (obs, _) = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = get_action(q_state, obs)
            actions = jax.device_get(actions)
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
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            (loss, old_val, q_state) = update(q_state, data.observations.numpy(), data.actions.numpy(), data.next_observations.numpy(), data.rewards.numpy(), data.dones.numpy())
            if global_step % 100 == 0:
                writer.add_scalar('losses/loss', jax.device_get(loss), global_step)
                writer.add_scalar('losses/q_values', jax.device_get(old_val.mean()), global_step)
                print('SPS:', int(global_step / (time.time() - start_time)))
                writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))
    if args.save_model:
        model_path = f'runs/{run_name}/{args.exp_name}.cleanrl_model'
        model_data = {'model_weights': q_state.params, 'args': vars(args)}
        with open(model_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(model_data))
        print(f'model saved to {model_path}')
        from cleanrl_utils.evals.c51_jax_eval import evaluate
        episodic_returns = evaluate(model_path, make_env, args.env_id, eval_episodes=10, run_name=f'{run_name}-eval', Model=QNetwork, epsilon=0.05)
        for (idx, episodic_return) in enumerate(episodic_returns):
            writer.add_scalar('eval/episodic_return', episodic_return, idx)
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub
            repo_name = f'{args.env_id}-{args.exp_name}-seed{args.seed}'
            repo_id = f'{args.hf_entity}/{repo_name}' if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, 'C51', f'runs/{run_name}', f'videos/{run_name}-eval')
    envs.close()
    writer.close()
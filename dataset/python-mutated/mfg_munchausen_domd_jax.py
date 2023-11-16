"""Run deep online mirror descent algorithm with Munchausen DQN agents."""
import os
from typing import Sequence
from absl import flags
import jax
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.mfg import utils
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import munchausen_deep_mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.utils import app
from open_spiel.python.utils import metrics
FLAGS = flags.FLAGS
flags.DEFINE_string('game_name', 'mfg_crowd_modelling_2d', 'Name of the game.')
_ENV_SETTING = flags.DEFINE_string('env_setting', 'crowd_modelling_2d_four_rooms', 'Name of the game settings. If None, the game name will be used.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'Number of transitions to sample at each learning step.')
_LEARN_EVERY = flags.DEFINE_integer('learn_every', 64, 'Number of steps between learning updates.')
_NUM_EPISODES_PER_ITERATION = flags.DEFINE_integer('num_episodes_per_iteration', 1000, 'Number of training eepisodes for each iteration.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations.')
_EPSILON_DECAY_DURATION = flags.DEFINE_integer('epsilon_decay_duration', 100000, 'Number of game steps over which epsilon is decayed.')
flags.DEFINE_float('epsilon_power', 1, 'Power for the epsilon decay.')
flags.DEFINE_float('epsilon_start', 0.1, 'Starting exploration parameter.')
flags.DEFINE_float('epsilon_end', 0.1, 'Final exploration parameter.')
_DISCOUNT_FACTOR = flags.DEFINE_float('discount_factor', 1.0, 'Discount factor for future rewards.')
_RESET_REPLAY_BUFFER_ON_UPDATE = flags.DEFINE_bool('reset_replay_buffer_on_update', False, 'Reset the replay buffer when the softmax policy is updated.')
flags.DEFINE_integer('seed', 42, 'Training seed.')
_EVAL_EVERY = flags.DEFINE_integer('eval_every', 200, 'Episode frequency at which the agents are evaluated.')
_HIDDEN_LAYERS_SIZES = flags.DEFINE_list('hidden_layers_sizes', ['128', '128'], 'Number of hidden units in the avg-net and Q-net.')
_UPDATE_TARGET_NETWORK_EVERY = flags.DEFINE_integer('update_target_network_every', 200, 'Number of steps between DQN target network updates.')
_REPLAY_BUFFER_CAPACITY = flags.DEFINE_integer('replay_buffer_capacity', 40000, 'Size of the replay buffer.')
_MIN_BUFFER_SIZE_TO_LEARN = flags.DEFINE_integer('min_buffer_size_to_learn', 1000, 'Number of samples in buffer before learning begins.')
flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam'], 'Optimizer.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for inner rl agent.')
flags.DEFINE_enum('loss', 'mse', ['mse', 'huber'], 'Loss function.')
flags.DEFINE_float('huber_loss_parameter', 1.0, 'Parameter for Huber loss.')
flags.DEFINE_float('gradient_clipping', None, 'Value to clip the gradient to.')
flags.DEFINE_float('tau', 10, 'Temperature parameter in Munchausen target.')
flags.DEFINE_float('alpha', 0.99, 'Alpha parameter in Munchausen target.')
_WITH_MUNCHAUSEN = flags.DEFINE_bool('with_munchausen', True, 'If true, target uses Munchausen penalty terms.')
flags.DEFINE_bool('use_checkpoints', False, 'Save/load neural network weights.')
_CHECKPOINT_DIR = flags.DEFINE_string('checkpoint_dir', '/tmp/dqn_test', 'Directory to save/load the agent.')
_LOGDIR = flags.DEFINE_string('logdir', None, 'Logging dir to use for TF summary files. If None, the metrics will only be logged to stderr.')
_LOG_DISTRIBUTION = flags.DEFINE_bool('log_distribution', False, 'Enables logging of the distribution.')

def main(argv: Sequence[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    game = factory.create_game_with_setting(FLAGS.game_name, _ENV_SETTING.value)
    num_players = game.num_players()
    uniform_policy = policy.UniformRandomPolicy(game)
    uniform_dist = distribution.DistributionPolicy(game, uniform_policy)
    envs = [rl_environment.Environment(game, mfg_distribution=uniform_dist, mfg_population=p, observation_type=rl_environment.ObservationType.OBSERVATION) for p in range(num_players)]
    env = envs[0]
    info_state_size = env.observation_spec()['info_state'][0]
    num_actions = env.action_spec()['num_actions']
    kwargs = {'alpha': FLAGS.alpha, 'batch_size': _BATCH_SIZE.value, 'discount_factor': _DISCOUNT_FACTOR.value, 'epsilon_decay_duration': _EPSILON_DECAY_DURATION.value, 'epsilon_end': FLAGS.epsilon_end, 'epsilon_power': FLAGS.epsilon_power, 'epsilon_start': FLAGS.epsilon_start, 'gradient_clipping': FLAGS.gradient_clipping, 'hidden_layers_sizes': [int(l) for l in _HIDDEN_LAYERS_SIZES.value], 'huber_loss_parameter': FLAGS.huber_loss_parameter, 'learn_every': _LEARN_EVERY.value, 'learning_rate': FLAGS.learning_rate, 'loss': FLAGS.loss, 'min_buffer_size_to_learn': _MIN_BUFFER_SIZE_TO_LEARN.value, 'optimizer': FLAGS.optimizer, 'replay_buffer_capacity': _REPLAY_BUFFER_CAPACITY.value, 'reset_replay_buffer_on_update': _RESET_REPLAY_BUFFER_ON_UPDATE.value, 'seed': FLAGS.seed, 'tau': FLAGS.tau, 'update_target_network_every': _UPDATE_TARGET_NETWORK_EVERY.value, 'with_munchausen': _WITH_MUNCHAUSEN.value}
    agents = [munchausen_deep_mirror_descent.MunchausenDQN(p, info_state_size, num_actions, **kwargs) for p in range(num_players)]
    just_logging = _LOGDIR.value is None or jax.host_id() > 0
    writer = metrics.create_default_writer(logdir=_LOGDIR.value, just_logging=just_logging)
    writer.write_hparams(kwargs)

    def logging_fn(it, episode, vals):
        if False:
            for i in range(10):
                print('nop')
        writer.write_scalars(it * num_episodes_per_iteration + episode, vals)
    num_episodes_per_iteration = _NUM_EPISODES_PER_ITERATION.value
    md = munchausen_deep_mirror_descent.DeepOnlineMirrorDescent(game, envs, agents, eval_every=_EVAL_EVERY.value, num_episodes_per_iteration=num_episodes_per_iteration, logging_fn=logging_fn)

    def log_metrics(it):
        if False:
            print('Hello World!')
        'Logs the training metrics for each iteration.'
        initial_states = game.new_initial_states()
        pi_value = policy_value.PolicyValue(game, md.distribution, md.policy)
        m = {f'best_response/{state}': pi_value.eval_state(state) for state in initial_states}
        nash_conv_md = nash_conv.NashConv(game, md.policy).nash_conv()
        m['nash_conv_md'] = nash_conv_md
        if _LOG_DISTRIBUTION.value and _LOGDIR.value:
            filename = os.path.join(_LOGDIR.value, f'distribution_{it}.pkl')
            utils.save_parametric_distribution(md.distribution, filename)
        logging_fn(it, 0, m)
    log_metrics(0)
    for it in range(1, FLAGS.num_iterations + 1):
        md.iteration()
        log_metrics(it)
    writer.flush()
if __name__ == '__main__':
    jax.config.parse_flags_with_absl()
    app.run(main)
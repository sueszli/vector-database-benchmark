"""Tabular Q-Learning on Lewis Signaling Game."""
import copy
from absl import app
from absl import flags
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_states', 3, 'Number of states and actions')
flags.DEFINE_integer('num_messages', 3, 'Number of messages')
flags.DEFINE_string('payoffs', '1, 0, 0, 0, 1, 0, 0, 0, 1', "Payoffs to use ('random' for random [0, 1) payoffs)")
flags.DEFINE_bool('centralized', False, 'Set to use centralized learning')
flags.DEFINE_integer('num_episodes', 2000, 'Number of train episodes')
flags.DEFINE_float('step_size', 0.1, 'Step size for updates')
flags.DEFINE_float('eps_init', 1.0, 'Initial value of epsilon')
flags.DEFINE_float('eps_final', 0.0, 'Final value of epsilon')
flags.DEFINE_integer('eps_decay_steps', 1900, 'Number of episodes to decay epsilon')
flags.DEFINE_integer('num_runs', 100, 'Number of repetitions')
flags.DEFINE_integer('log_interval', 10, 'Number of episodes between each logging')
flags.DEFINE_bool('plot', False, 'Set to plot the graphs')
flags.DEFINE_bool('compare', False, 'Set to compare centralized vs decentralized')

def run_experiment(num_players, env, payoffs, centralized):
    if False:
        print('Hello World!')
    'Run the experiments.'
    num_states = FLAGS.num_states
    num_messages = FLAGS.num_messages
    num_actions = env.action_spec()['num_actions']
    num_runs = FLAGS.num_runs
    training_episodes = FLAGS.num_episodes
    log_interval = FLAGS.log_interval
    rewards = np.zeros((num_runs, training_episodes // log_interval))
    opts = np.zeros((num_runs, training_episodes // log_interval))
    converge_point = np.zeros((num_states, num_states))
    percent_opt = 0
    for i in range(num_runs):
        eps_schedule = rl_tools.LinearSchedule(FLAGS.eps_init, FLAGS.eps_final, FLAGS.eps_decay_steps * 2)
        agents = [tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions, step_size=FLAGS.step_size, epsilon_schedule=eps_schedule, centralized=centralized) for idx in range(num_players)]
        for cur_episode in range(training_episodes):
            time_step = env.reset()
            cur_state = time_step.observations['info_state'][0][3:].index(1)
            while not time_step.last():
                player_id = time_step.observations['current_player']
                agent_output = agents[player_id].step(time_step)
                time_step = env.step([agent_output.action])
            for agent in agents:
                agent.step(time_step)
            reward = time_step.rewards[0]
            max_reward = payoffs[cur_state].max()
            cur_idx = (i, cur_episode // log_interval)
            rewards[cur_idx] += reward / log_interval
            opts[cur_idx] += np.isclose(reward, max_reward) / log_interval
        base_info_state0 = [1.0, 0.0, 0.0] + [0.0] * num_states
        base_info_state1 = [0.0, 1.0, 0.0] + [0.0] * num_states
        if centralized:
            base_info_state0 = [base_info_state0, base_info_state0.copy()]
            base_info_state1 = [base_info_state1, base_info_state1.copy()]
        for s in range(num_states):
            info_state0 = copy.deepcopy(base_info_state0)
            if centralized:
                info_state0[0][3 + s] = 1.0
            else:
                info_state0[3 + s] = 1.0
            (m, _) = agents[0]._epsilon_greedy(str(info_state0), np.arange(num_messages), 0)
            info_state1 = copy.deepcopy(base_info_state1)
            if centralized:
                info_state1[0][3 + s] = 1.0
                info_state1[1][3 + m] = 1.0
            else:
                info_state1[3 + m] = 1.0
            (a, _) = agents[1]._epsilon_greedy(str(info_state1), np.arange(num_states), 0)
            converge_point[s, a] += 1
            best_act = payoffs[s].argmax()
            percent_opt += int(a == best_act) / num_runs / num_states
    return (rewards, opts, converge_point, percent_opt)

def main(_):
    if False:
        i = 10
        return i + 15
    game = 'lewis_signaling'
    num_players = 2
    num_states = FLAGS.num_states
    num_messages = FLAGS.num_messages
    if FLAGS.payoffs == 'random':
        payoffs = np.random.random((num_states, num_states))
        payoffs_str = ','.join([str(x) for x in payoffs.flatten()])
    elif FLAGS.payoffs == 'climbing':
        payoffs = np.array([[11, -30, 0], [-30, 7, 6], [0, 0, 5]]) / 30
        payoffs_str = ','.join([str(x) for x in payoffs.flatten()])
    else:
        payoffs_str = FLAGS.payoffs
        try:
            payoffs_list = [float(x) for x in payoffs_str.split(',')]
            payoffs = np.array(payoffs_list).reshape((num_states, num_states))
        except ValueError:
            raise ValueError('There should be {} (states * actions) elements in payoff. Found {} elements'.format(num_states * num_states, len(payoffs_list))) from None
    env_configs = {'num_states': num_states, 'num_messages': num_messages, 'payoffs': payoffs_str}
    env = rl_environment.Environment(game, **env_configs)
    if FLAGS.compare:
        rewards_list = []
        opts_list = []
        converge_point_list = []
        percent_opt_list = []
        for centralized in [True, False]:
            (rewards, opts, converge_point, percent_opt) = run_experiment(num_players, env, payoffs, centralized)
            rewards_list += [rewards]
            opts_list += [opts]
            converge_point_list += [converge_point]
            percent_opt_list += [percent_opt]
    else:
        (rewards, opts, converge_point, percent_opt) = run_experiment(num_players, env, payoffs, FLAGS.centralized)
        rewards_list = [rewards]
        opts_list = [opts]
        converge_point_list = [converge_point]
        percent_opt_list = [percent_opt]
    if FLAGS.plot:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from scipy import stats
        params = {'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11}
        mpl.rcParams.update(params)

        def init_fig():
            if False:
                i = 10
                return i + 15
            (fig, ax) = plt.subplots(1, 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            return (fig, ax)

        def plot_scalars(scalars, repetition_axis=0, scalar_labels=None, title=None, ax_labels=None):
            if False:
                i = 10
                return i + 15
            'Plots scalar on ax by filling 1 standard error.\n\n      Args:\n          scalars: List of scalars to plot (mean taken over repetition\n            axis)\n          repetition_axis: Axis to take the mean over\n          scalar_labels: Labels for the scalars (for legend)\n          title: Figure title\n          ax_labels: Labels for x and y axis (list of 2 strings)\n      '
            if not all([len(s.shape) == 2 for s in scalars]):
                raise ValueError('Only 2D arrays supported for plotting')
            if scalar_labels is None:
                scalar_labels = [None] * len(scalars)
            if len(scalars) != len(scalar_labels):
                raise ValueError('Wrong number of scalar labels, expected {} but received {}'.format(len(scalars), len(scalar_labels)))
            (_, plot_axis) = init_fig()
            for (i, scalar) in enumerate(scalars):
                xs = np.arange(scalar.shape[1 - repetition_axis]) * FLAGS.log_interval
                mean = scalar.mean(axis=repetition_axis)
                sem = stats.sem(scalar, axis=repetition_axis)
                plot_axis.plot(xs, mean, label=scalar_labels[i])
                plot_axis.fill_between(xs, mean - sem, mean + sem, alpha=0.5)
            if title is not None:
                plot_axis.set_title(title)
            if ax_labels is not None:
                plot_axis.set_xlabel(ax_labels[0])
                plot_axis.set_ylabel(ax_labels[1])

        def plot_confusion_matrix(cm, cmap=plt.cm.Blues, title=None):
            if False:
                print('Hello World!')
            "Plots the confusion matrix.\n\n      Args:\n          cm (np.ndarray): Confusion matrix to plot\n          cmap: Color map to be used in matplotlib's imshow\n          title: Figure title\n\n      Returns:\n          Figure and axis on which the confusion matrix is plotted\n      "
            (fig, ax) = plt.subplots()
            ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Receiver's action", fontsize=14)
            ax.set_ylabel("Sender's state", fontsize=14)
            fmt = 'd'
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
            fig.tight_layout()
            if title is not None:
                ax.set_title(title)
            return (fig, ax)
        if FLAGS.compare:
            labels = ['Centralized', 'Decentralized']
        else:
            labels = ['Centralized'] if FLAGS.centralized else ['Decentralized']
        plot_scalars(rewards_list, scalar_labels=labels, title='Reward graph (Tabular Q-Learning)', ax_labels=['Episodes', 'Reward per episode'])
        plt.legend()
        plot_scalars(opts_list, scalar_labels=labels, title='Percentage of optimal actions (Tabular Q-Learning)', ax_labels=['Episodes', '% optimal actions'])
        plt.legend()
        for (i, cp) in enumerate(converge_point_list):
            plot_confusion_matrix(cp.astype(int), title='Final policy (Tabular {})'.format(labels[i]))
        plt.show()
    return percent_opt_list
if __name__ == '__main__':
    app.run(main)
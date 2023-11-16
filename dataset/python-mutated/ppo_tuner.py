"""
ppo_tuner.py including:
    class PPOTuner
"""
import copy
import logging
import numpy as np
from gym import spaces
from schema import Schema, Optional
import nni
from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward
from .model import Model
from .util import set_global_seeds
from .policy import build_lstm_policy
logger = logging.getLogger('ppo_tuner_AutoML')

def _constfn(val):
    if False:
        i = 10
        return i + 15
    '\n    Wrap as function\n    '

    def f(_):
        if False:
            i = 10
            return i + 15
        return val
    return f

class ModelConfig:
    """
    Configurations of the PPO model
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.observation_space = None
        self.action_space = None
        self.num_envs = 0
        self.nsteps = 0
        self.ent_coef = 0.0
        self.lr = 0.0003
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.gamma = 0.99
        self.lam = 0.95
        self.cliprange = 0.2
        self.embedding_size = None
        self.noptepochs = 4
        self.total_timesteps = 5000
        self.nminibatches = 4

class TrialsInfo:
    """
    Informations of each trial from one model inference
    """

    def __init__(self, obs, actions, values, neglogpacs, dones, last_value, inf_batch_size):
        if False:
            return 10
        self.iter = 0
        self.obs = obs
        self.actions = actions
        self.values = values
        self.neglogpacs = neglogpacs
        self.dones = dones
        self.last_value = last_value
        self.rewards = None
        self.returns = None
        self.inf_batch_size = inf_batch_size

    def get_next(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get actions of the next trial\n        '
        if self.iter >= self.inf_batch_size:
            return (None, None)
        actions = []
        for step in self.actions:
            actions.append(step[self.iter])
        self.iter += 1
        return (self.iter - 1, actions)

    def update_rewards(self, rewards, returns):
        if False:
            return 10
        '\n        After the trial is finished, reward and return of this trial is updated\n        '
        self.rewards = rewards
        self.returns = returns

    def convert_shape(self):
        if False:
            i = 10
            return i + 15
        '\n        Convert shape\n        '

        def sf01(arr):
            if False:
                return 10
            '\n            swap and then flatten axes 0 and 1\n            '
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        self.obs = sf01(self.obs)
        self.returns = sf01(self.returns)
        self.dones = sf01(self.dones)
        self.actions = sf01(self.actions)
        self.values = sf01(self.values)
        self.neglogpacs = sf01(self.neglogpacs)

class PPOModel:
    """
    PPO Model
    """

    def __init__(self, model_config, mask):
        if False:
            print('Hello World!')
        self.model_config = model_config
        self.states = None
        self.nupdates = None
        self.cur_update = 1
        self.np_mask = mask
        set_global_seeds(None)
        assert isinstance(self.model_config.lr, float)
        self.lr = _constfn(self.model_config.lr)
        assert isinstance(self.model_config.cliprange, float)
        self.cliprange = _constfn(self.model_config.cliprange)
        policy = build_lstm_policy(model_config)
        nenvs = model_config.num_envs
        self.nbatch = nbatch = nenvs * model_config.nsteps
        nbatch_train = nbatch // model_config.nminibatches
        self.nupdates = self.model_config.total_timesteps // self.nbatch
        self.model = Model(policy=policy, nbatch_act=nenvs, nbatch_train=nbatch_train, nsteps=model_config.nsteps, ent_coef=model_config.ent_coef, vf_coef=model_config.vf_coef, max_grad_norm=model_config.max_grad_norm, np_mask=self.np_mask)
        self.states = self.model.initial_state
        logger.info('=== finished PPOModel initialization')

    def inference(self, num):
        if False:
            print('Hello World!')
        '\n        Generate actions along with related info from policy network.\n        observation is the action of the last step.\n\n        Parameters\n        ----------\n        num: int\n            The number of trials to generate\n\n        Returns\n        -------\n        mb_obs : list\n            Observation of the ``num`` configurations\n        mb_actions : list\n            Actions of the ``num`` configurations\n        mb_values : list\n            Values from the value function of the ``num`` configurations\n        mb_neglogpacs : list\n            ``neglogp`` of the ``num`` configurations\n        mb_dones : list\n            To show whether the play is done, always ``True``\n        last_values : tensorflow tensor\n            The last values of the ``num`` configurations, got with session run\n        '
        (mb_obs, mb_actions, mb_values, mb_dones, mb_neglogpacs) = ([], [], [], [], [])
        first_step_ob = self.model_config.action_space.n
        obs = [first_step_ob for _ in range(num)]
        dones = [True for _ in range(num)]
        states = self.states
        for cur_step in range(self.model_config.nsteps):
            (actions, values, states, neglogpacs) = self.model.step(cur_step, obs, S=states, M=dones)
            mb_obs.append(obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(dones)
            obs[:] = actions
            if cur_step == self.model_config.nsteps - 1:
                dones = [True for _ in range(num)]
            else:
                dones = [False for _ in range(num)]
        np_obs = np.asarray(obs)
        mb_obs = np.asarray(mb_obs, dtype=np_obs.dtype)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=bool)
        last_values = self.model.value(np_obs, S=states, M=dones)
        return (mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values)

    def compute_rewards(self, trials_info, trials_result):
        if False:
            i = 10
            return i + 15
        '\n        Compute the rewards of the trials in trials_info based on trials_result,\n        and update the rewards in trials_info\n\n        Parameters\n        ----------\n        trials_info : TrialsInfo\n            Info of the generated trials\n        trials_result : list\n            Final results (e.g., acc) of the generated trials\n        '
        mb_rewards = np.asarray([trials_result for _ in trials_info.actions], dtype=np.float32)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        last_dones = np.asarray([True for _ in trials_result], dtype=bool)
        for t in reversed(range(self.model_config.nsteps)):
            if t == self.model_config.nsteps - 1:
                nextnonterminal = 1.0 - last_dones
                nextvalues = trials_info.last_value
            else:
                nextnonterminal = 1.0 - trials_info.dones[t + 1]
                nextvalues = trials_info.values[t + 1]
            delta = mb_rewards[t] + self.model_config.gamma * nextvalues * nextnonterminal - trials_info.values[t]
            lastgaelam = delta + self.model_config.gamma * self.model_config.lam * nextnonterminal * lastgaelam
            mb_advs[t] = lastgaelam
        mb_returns = mb_advs + trials_info.values
        trials_info.update_rewards(mb_rewards, mb_returns)
        trials_info.convert_shape()

    def train(self, trials_info, nenvs):
        if False:
            while True:
                i = 10
        '\n        Train the policy/value network using trials_info\n\n        Parameters\n        ----------\n        trials_info : TrialsInfo\n            Complete info of the generated trials from the previous inference\n        nenvs : int\n            The batch size of the (previous) inference\n        '
        if self.cur_update <= self.nupdates:
            frac = 1.0 - (self.cur_update - 1.0) / self.nupdates
        else:
            logger.warning('current update (self.cur_update) %d has exceeded total updates (self.nupdates) %d', self.cur_update, self.nupdates)
            frac = 1.0 - (self.nupdates - 1.0) / self.nupdates
        lrnow = self.lr(frac)
        cliprangenow = self.cliprange(frac)
        self.cur_update += 1
        states = self.states
        assert states is not None
        assert nenvs % self.model_config.nminibatches == 0
        envsperbatch = nenvs // self.model_config.nminibatches
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * self.model_config.nsteps).reshape(nenvs, self.model_config.nsteps)
        for _ in range(self.model_config.noptepochs):
            np.random.shuffle(envinds)
            for start in range(0, nenvs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                slices = (arr[mbflatinds] for arr in (trials_info.obs, trials_info.returns, trials_info.dones, trials_info.actions, trials_info.values, trials_info.neglogpacs))
                mbstates = states[mbenvinds]
                self.model.train(lrnow, cliprangenow, *slices, mbstates)

class PPOClassArgsValidator(ClassArgsValidator):

    def validate_class_args(self, **kwargs):
        if False:
            while True:
                i = 10
        Schema({'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'), Optional('trials_per_update'): self.range('trials_per_update', int, 0, 99999), Optional('epochs_per_update'): self.range('epochs_per_update', int, 0, 99999), Optional('minibatch_size'): self.range('minibatch_size', int, 0, 99999), Optional('ent_coef'): float, Optional('lr'): float, Optional('vf_coef'): float, Optional('max_grad_norm'): float, Optional('gamma'): float, Optional('lam'): float, Optional('cliprange'): float}).validate(kwargs)

class PPOTuner(Tuner):
    """
    PPOTuner, the implementation inherits the main logic of the implementation
    `ppo2 from openai <https://github.com/openai/baselines/tree/master/baselines/ppo2>`__ and is adapted for NAS scenario.
    It uses ``lstm`` for its policy network and value network, policy and value share the same network.

    Parameters
    ----------
    optimize_mode : str
        maximize or minimize
    trials_per_update : int
        Number of trials to have for each model update
    epochs_per_update : int
        Number of epochs to run for each model update
    minibatch_size : int
        Minibatch size (number of trials) for the update
    ent_coef : float
        Policy entropy coefficient in the optimization objective
    lr : float
        Learning rate of the model (lstm network), constant
    vf_coef : float
        Value function loss coefficient in the optimization objective
    max_grad_norm : float
        Gradient norm clipping coefficient
    gamma : float
        Discounting factor
    lam : float
        Advantage estimation discounting factor (lambda in the paper)
    cliprange : float
        Cliprange in the PPO algorithm, constant
    """

    def __init__(self, optimize_mode, trials_per_update=20, epochs_per_update=4, minibatch_size=4, ent_coef=0.0, lr=0.0003, vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95, cliprange=0.2):
        if False:
            while True:
                i = 10
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.model_config = ModelConfig()
        self.model = None
        self.search_space = None
        self.running_trials = {}
        self.inf_batch_size = trials_per_update
        self.first_inf = True
        self.trials_result = [None for _ in range(self.inf_batch_size)]
        self.credit = 0
        self.param_ids = []
        self.finished_trials = 0
        self.chosen_arch_template = {}
        self.actions_spaces = None
        self.actions_to_config = None
        self.full_act_space = None
        self.trials_info = None
        self.all_trials = {}
        self.model_config.num_envs = self.inf_batch_size
        self.model_config.noptepochs = epochs_per_update
        self.model_config.nminibatches = minibatch_size
        self.send_trial_callback = None
        logger.info('Finished PPOTuner initialization')

    def _process_nas_space(self, search_space):
        if False:
            while True:
                i = 10
        actions_spaces = []
        actions_to_config = []
        for (key, val) in search_space.items():
            if val['_type'] == 'layer_choice':
                actions_to_config.append((key, 'layer_choice'))
                actions_spaces.append(val['_value'])
                self.chosen_arch_template[key] = None
            elif val['_type'] == 'input_choice':
                candidates = val['_value']['candidates']
                n_chosen = val['_value']['n_chosen']
                if n_chosen not in [0, 1, [0, 1]]:
                    raise ValueError('Optional_input_size can only be 0, 1, or [0, 1], but the pecified one is %s' % n_chosen)
                if isinstance(n_chosen, list):
                    actions_to_config.append((key, 'input_choice'))
                    actions_spaces.append(['None', *candidates])
                    self.chosen_arch_template[key] = None
                elif n_chosen == 1:
                    actions_to_config.append((key, 'input_choice'))
                    actions_spaces.append(candidates)
                    self.chosen_arch_template[key] = None
                elif n_chosen == 0:
                    self.chosen_arch_template[key] = []
            else:
                raise ValueError('Unsupported search space type: %s' % val['_type'])
        dedup = {}
        for step in actions_spaces:
            for action in step:
                dedup[action] = 1
        full_act_space = [act for (act, _) in dedup.items()]
        assert len(full_act_space) == len(dedup)
        observation_space = len(full_act_space)
        nsteps = len(actions_spaces)
        return (actions_spaces, actions_to_config, full_act_space, observation_space, nsteps)

    def _generate_action_mask(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Different step could have different action space. to deal with this case, we merge all the\n        possible actions into one action space, and use mask to indicate available actions for each step\n        '
        two_masks = []
        mask = []
        for acts in self.actions_spaces:
            one_mask = [0 for _ in range(len(self.full_act_space))]
            for act in acts:
                idx = self.full_act_space.index(act)
                one_mask[idx] = 1
            mask.append(one_mask)
        two_masks.append(mask)
        mask = []
        for acts in self.actions_spaces:
            one_mask = [-np.inf for _ in range(len(self.full_act_space))]
            for act in acts:
                idx = self.full_act_space.index(act)
                one_mask[idx] = 0
            mask.append(one_mask)
        two_masks.append(mask)
        return np.asarray(two_masks, dtype=np.float32)

    def update_search_space(self, search_space):
        if False:
            print('Hello World!')
        '\n        Get search space, currently the space only includes that for NAS\n\n        Parameters\n        ----------\n        search_space : dict\n            Search space for NAS\n            the format could be referred to search space spec (https://nni.readthedocs.io/en/latest/Tutorial/SearchSpaceSpec.html).\n        '
        logger.info('update search space %s', search_space)
        assert self.search_space is None
        self.search_space = search_space
        assert self.model_config.observation_space is None
        assert self.model_config.action_space is None
        (self.actions_spaces, self.actions_to_config, self.full_act_space, obs_space, nsteps) = self._process_nas_space(search_space)
        self.model_config.observation_space = spaces.Discrete(obs_space)
        self.model_config.action_space = spaces.Discrete(obs_space)
        self.model_config.nsteps = nsteps
        mask = self._generate_action_mask()
        assert self.model is None
        self.model = PPOModel(self.model_config, mask)

    def _actions_to_config(self, actions):
        if False:
            return 10
        '\n        Given actions, to generate the corresponding trial configuration\n        '
        chosen_arch = copy.deepcopy(self.chosen_arch_template)
        for (cnt, act) in enumerate(actions):
            act_name = self.full_act_space[act]
            (_key, _type) = self.actions_to_config[cnt]
            if _type == 'input_choice':
                if act_name == 'None':
                    chosen_arch[_key] = {'_value': [], '_idx': []}
                else:
                    candidates = self.search_space[_key]['_value']['candidates']
                    idx = candidates.index(act_name)
                    chosen_arch[_key] = {'_value': [act_name], '_idx': [idx]}
            elif _type == 'layer_choice':
                idx = self.search_space[_key]['_value'].index(act_name)
                chosen_arch[_key] = {'_value': act_name, '_idx': idx}
            else:
                raise ValueError('unrecognized key: {0}'.format(_type))
        return chosen_arch

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.\n\n        Parameters\n        ----------\n        parameter_id_list : list of int\n            Unique identifiers for each set of requested hyper-parameters.\n            These will later be used in :meth:`receive_trial_result`.\n        **kwargs\n            Not used\n\n        Returns\n        -------\n        list\n            A list of newly generated configurations\n        '
        result = []
        self.send_trial_callback = kwargs['st_callback']
        for parameter_id in parameter_id_list:
            had_exception = False
            try:
                logger.debug('generating param for %s', parameter_id)
                res = self.generate_parameters(parameter_id, **kwargs)
            except nni.NoMoreTrialError:
                had_exception = True
            if not had_exception:
                result.append(res)
        return result

    def generate_parameters(self, parameter_id, **kwargs):
        if False:
            return 10
        '\n        Generate parameters, if no trial configration for now, self.credit plus 1 to send the config later\n\n        Parameters\n        ----------\n        parameter_id : int\n            Unique identifier for requested hyper-parameters.\n            This will later be used in :meth:`receive_trial_result`.\n        **kwargs\n            Not used\n\n        Returns\n        -------\n        dict\n            One newly generated configuration\n\n        '
        if self.first_inf:
            self.trials_result = [None for _ in range(self.inf_batch_size)]
            (mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values) = self.model.inference(self.inf_batch_size)
            self.trials_info = TrialsInfo(mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values, self.inf_batch_size)
            self.first_inf = False
        (trial_info_idx, actions) = self.trials_info.get_next()
        if trial_info_idx is None:
            logger.debug('Credit added by one in parameters request')
            self.credit += 1
            self.param_ids.append(parameter_id)
            raise nni.NoMoreTrialError('no more parameters now.')
        self.running_trials[parameter_id] = trial_info_idx
        new_config = self._actions_to_config(actions)
        return new_config

    def _next_round_inference(self):
        if False:
            return 10
        '\n        Run a inference to generate next batch of configurations\n        '
        logger.debug('Start next round inference...')
        self.finished_trials = 0
        self.model.compute_rewards(self.trials_info, self.trials_result)
        self.model.train(self.trials_info, self.inf_batch_size)
        self.running_trials = {}
        self.trials_result = [None for _ in range(self.inf_batch_size)]
        (mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values) = self.model.inference(self.inf_batch_size)
        self.trials_info = TrialsInfo(mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, last_values, self.inf_batch_size)
        logger.debug('Next round inference complete.')
        for _ in range(self.credit):
            (trial_info_idx, actions) = self.trials_info.get_next()
            if trial_info_idx is None:
                logger.warning('No enough trial config, trials_per_update is suggested to be larger than trialConcurrency')
                break
            assert self.param_ids
            param_id = self.param_ids.pop()
            self.running_trials[param_id] = trial_info_idx
            new_config = self._actions_to_config(actions)
            self.send_trial_callback(param_id, new_config)
            self.credit -= 1
            logger.debug('Send new trial (%d, %s) for reducing credit', param_id, new_config)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        if False:
            return 10
        "\n        Receive trial's result. if the number of finished trials equals self.inf_batch_size, start the next update to\n        train the model.\n\n        Parameters\n        ----------\n        parameter_id : int\n            Unique identifier of used hyper-parameters, same with :meth:`generate_parameters`.\n        parameters : dict\n            Hyper-parameters generated by :meth:`generate_parameters`.\n        value : dict\n            Result from trial (the return value of :func:`nni.report_final_result`).\n        "
        trial_info_idx = self.running_trials.pop(parameter_id, None)
        assert trial_info_idx is not None
        value = extract_scalar_reward(value)
        if self.optimize_mode == OptimizeMode.Minimize:
            value = -value
        self.trials_result[trial_info_idx] = value
        self.finished_trials += 1
        logger.debug('receive_trial_result, parameter_id %d, trial_info_idx %d, finished_trials %d, inf_batch_size %d', parameter_id, trial_info_idx, self.finished_trials, self.inf_batch_size)
        if self.finished_trials == self.inf_batch_size:
            logger.debug('Start next round inference in receive_trial_result')
            self._next_round_inference()

    def trial_end(self, parameter_id, success, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        To deal with trial failure. If a trial fails, it is popped out from ``self.running_trials``,\n        and the final result of this trial is assigned with the average of the finished trials.\n\n        Parameters\n        ----------\n        parameter_id : int\n            Unique identifier for hyper-parameters used by this trial.\n        success : bool\n            True if the trial successfully completed; False if failed or terminated.\n        **kwargs\n            Not used\n        '
        if not success:
            if parameter_id not in self.running_trials:
                logger.warning('The trial is failed, but self.running_trial does not have this trial')
                return
            trial_info_idx = self.running_trials.pop(parameter_id, None)
            assert trial_info_idx is not None
            values = [val for val in self.trials_result if val is not None]
            logger.warning('In trial_end, values: %s', values)
            self.trials_result[trial_info_idx] = np.mean(values) if values else 0
            self.finished_trials += 1
            if self.finished_trials == self.inf_batch_size:
                logger.debug('Start next round inference in trial_end')
                self._next_round_inference()

    def import_data(self, data):
        if False:
            print('Hello World!')
        '\n        Import additional data for tuning, not supported yet.\n\n        Parameters\n        ----------\n        data : list\n            A list of dictionarys, each of which has at least two keys, ``parameter`` and ``value``\n        '
        logger.warning('PPOTuner cannot leverage imported data.')
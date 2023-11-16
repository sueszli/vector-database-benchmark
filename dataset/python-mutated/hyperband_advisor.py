"""
hyperband_advisor.py
"""
import copy
import logging
import math
import sys
import numpy as np
from schema import Schema, Optional
import nni
from nni import ClassArgsValidator
from nni.common.hpo_utils import validate_search_space
from nni.runtime.common import multi_phase_enabled
from nni.runtime.msg_dispatcher_base import MsgDispatcherBase
from nni.runtime.tuner_command_channel import CommandType
from nni.utils import NodeType, OptimizeMode, MetricType, extract_scalar_reward
from nni import parameter_expressions
_logger = logging.getLogger(__name__)
_next_parameter_id = 0
_KEY = 'TRIAL_BUDGET'
_epsilon = 1e-06

def create_parameter_id():
    if False:
        while True:
            i = 10
    'Create an id\n\n    Returns\n    -------\n    int\n        parameter id\n    '
    global _next_parameter_id
    _next_parameter_id += 1
    return _next_parameter_id - 1

def create_bracket_parameter_id(brackets_id, brackets_curr_decay, increased_id=-1):
    if False:
        i = 10
        return i + 15
    "Create a full id for a specific bracket's hyperparameter configuration\n\n    Parameters\n    ----------\n    brackets_id: string\n        brackets id\n    brackets_curr_decay:\n        brackets curr decay\n    increased_id: int\n        increased id\n\n    Returns\n    -------\n    int\n        params id\n    "
    if increased_id == -1:
        increased_id = str(create_parameter_id())
    params_id = '_'.join([brackets_id, str(brackets_curr_decay), increased_id])
    return params_id

def json2parameter(ss_spec, random_state):
    if False:
        return 10
    'Randomly generate values for hyperparameters from hyperparameter space i.e., x.\n\n    Parameters\n    ----------\n    ss_spec:\n        hyperparameter space\n    random_state:\n        random operator to generate random values\n\n    Returns\n    -------\n    Parameter:\n        Parameters in this experiment\n    '
    if isinstance(ss_spec, dict):
        if NodeType.TYPE in ss_spec.keys():
            _type = ss_spec[NodeType.TYPE]
            _value = ss_spec[NodeType.VALUE]
            if _type == 'choice':
                _index = random_state.randint(len(_value))
                chosen_params = json2parameter(ss_spec[NodeType.VALUE][_index], random_state)
            else:
                chosen_params = getattr(parameter_expressions, _type)(*_value + [random_state])
        else:
            chosen_params = dict()
            for key in ss_spec.keys():
                chosen_params[key] = json2parameter(ss_spec[key], random_state)
    elif isinstance(ss_spec, list):
        chosen_params = list()
        for (_, subspec) in enumerate(ss_spec):
            chosen_params.append(json2parameter(subspec, random_state))
    else:
        chosen_params = copy.deepcopy(ss_spec)
    return chosen_params

class Bracket:
    """
    A bracket in Hyperband, all the information of a bracket is managed by an instance of this class

    Parameters
    ----------
    bracket_id: string
        The id of this bracket, usually be set as '{Hyperband index}-{SH iteration index}'
    s: int
        The current SH iteration index.
    s_max: int
        total number of SH iterations
    eta: float
        In each iteration, a complete run of sequential halving is executed. In it,
		after evaluating each configuration on the same subset size, only a fraction of
		1/eta of them 'advances' to the next round.
    R:
        the budget associated with each stage
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    """

    def __init__(self, bracket_id, s, s_max, eta, R, optimize_mode):
        if False:
            for i in range(10):
                print('nop')
        self.bracket_id = bracket_id
        self.s = s
        self.s_max = s_max
        self.eta = eta
        self.n = math.ceil((s_max + 1) * eta ** s / (s + 1) - _epsilon)
        self.r = R / eta ** s
        self.i = 0
        self.hyper_configs = []
        self.configs_perf = []
        self.num_configs_to_run = []
        self.num_finished_configs = []
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.no_more_trial = False

    def is_completed(self):
        if False:
            return 10
        'check whether this bracket has sent out all the hyperparameter configurations'
        return self.no_more_trial

    def get_n_r(self):
        if False:
            print('Hello World!')
        'return the values of n and r for the next round'
        return (math.floor(self.n / self.eta ** self.i + _epsilon), math.floor(self.r * self.eta ** self.i + _epsilon))

    def increase_i(self):
        if False:
            print('Hello World!')
        'i means the ith round. Increase i by 1'
        self.i += 1
        if self.i > self.s:
            self.no_more_trial = True

    def set_config_perf(self, i, parameter_id, seq, value):
        if False:
            return 10
        "update trial's latest result with its sequence number, e.g., epoch number or batch number\n\n        Parameters\n        ----------\n        i: int\n            the ith round\n        parameter_id: int\n            the id of the trial/parameter\n        seq: int\n            sequence number, e.g., epoch number or batch number\n        value: int\n            latest result with sequence number seq\n\n        Returns\n        -------\n        None\n        "
        if parameter_id in self.configs_perf[i]:
            if self.configs_perf[i][parameter_id][0] < seq:
                self.configs_perf[i][parameter_id] = [seq, value]
        else:
            self.configs_perf[i][parameter_id] = [seq, value]

    def inform_trial_end(self, i):
        if False:
            print('Hello World!')
        'If the trial is finished and the corresponding round (i.e., i) has all its trials finished,\n        it will choose the top k trials for the next round (i.e., i+1)\n\n        Parameters\n        ----------\n        i: int\n            the ith round\n        '
        global _KEY
        self.num_finished_configs[i] += 1
        _logger.debug('bracket id: %d, round: %d %d, finished: %d, all: %d', self.bracket_id, self.i, i, self.num_finished_configs[i], self.num_configs_to_run[i])
        if self.num_finished_configs[i] >= self.num_configs_to_run[i] and self.no_more_trial is False:
            assert self.i == i + 1
            this_round_perf = self.configs_perf[i]
            if self.optimize_mode is OptimizeMode.Maximize:
                sorted_perf = sorted(this_round_perf.items(), key=lambda kv: kv[1][1], reverse=True)
            else:
                sorted_perf = sorted(this_round_perf.items(), key=lambda kv: kv[1][1])
            _logger.debug('bracket %s next round %s, sorted hyper configs: %s', self.bracket_id, self.i, sorted_perf)
            (next_n, next_r) = self.get_n_r()
            _logger.debug('bracket %s next round %s, next_n=%d, next_r=%d', self.bracket_id, self.i, next_n, next_r)
            hyper_configs = dict()
            for k in range(next_n):
                params_id = sorted_perf[k][0]
                params = self.hyper_configs[i][params_id]
                params[_KEY] = next_r
                increased_id = params_id.split('_')[-1]
                new_id = create_bracket_parameter_id(self.bracket_id, self.i, increased_id)
                hyper_configs[new_id] = params
            self._record_hyper_configs(hyper_configs)
            return [[key, value] for (key, value) in hyper_configs.items()]
        return None

    def get_hyperparameter_configurations(self, num, r, searchspace_json, random_state):
        if False:
            for i in range(10):
                print('nop')
        'Randomly generate num hyperparameter configurations from search space\n\n        Parameters\n        ----------\n        num: int\n            the number of hyperparameter configurations\n\n        Returns\n        -------\n        list\n            a list of hyperparameter configurations. Format: [[key1, value1], [key2, value2], ...]\n        '
        global _KEY
        assert self.i == 0
        hyperparameter_configs = dict()
        for _ in range(num):
            params_id = create_bracket_parameter_id(self.bracket_id, self.i)
            params = json2parameter(searchspace_json, random_state)
            params[_KEY] = r
            hyperparameter_configs[params_id] = params
        self._record_hyper_configs(hyperparameter_configs)
        return [[key, value] for (key, value) in hyperparameter_configs.items()]

    def _record_hyper_configs(self, hyper_configs):
        if False:
            i = 10
            return i + 15
        'after generating one round of hyperconfigs, this function records the generated hyperconfigs,\n        creates a dict to record the performance when those hyperconifgs are running, set the number of finished configs\n        in this round to be 0, and increase the round number.\n\n        Parameters\n        ----------\n        hyper_configs: list\n            the generated hyperconfigs\n        '
        self.hyper_configs.append(hyper_configs)
        self.configs_perf.append(dict())
        self.num_finished_configs.append(0)
        self.num_configs_to_run.append(len(hyper_configs))
        self.increase_i()

class HyperbandClassArgsValidator(ClassArgsValidator):

    def validate_class_args(self, **kwargs):
        if False:
            i = 10
            return i + 15
        Schema({'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'), Optional('exec_mode'): self.choices('exec_mode', 'serial', 'parallelism'), Optional('R'): int, Optional('eta'): int}).validate(kwargs)

class Hyperband(MsgDispatcherBase):
    """
    `Hyperband <https://arxiv.org/pdf/1603.06560.pdf>`__ is a multi-fidelity hyperparameter tuning algorithm
    based on successive halving.

    The basic idea of Hyperband is to create several buckets,
    each having ``n`` randomly generated hyperparameter configurations,
    each configuration using ``r`` resources (e.g., epoch number, batch number).
    After the ``n`` configurations are finished, it chooses the top ``n/eta`` configurations
    and runs them using increased ``r*eta`` resources.
    At last, it chooses the best configuration it has found so far.
    Please refer to the paper :footcite:t:`li2017hyperband` for detailed algorithm.

    Examples
    --------

    .. code-block::

        config.tuner.name = 'Hyperband'
        config.tuner.class_args = {
            'optimize_mode': 'maximize',
            'R': 60,
            'eta': 3
        }


    Note that once you use Advisor, you are not allowed to add a Tuner and Assessor spec in the config file.
    When Hyperband is used, the dict returned by :func:`nni.get_next_parameter` one more key
    called ``TRIAL_BUDGET`` besides the hyperparameters and their values.
    **With this TRIAL_BUDGET, users can control in trial code how long a trial runs by following
    the suggested trial budget from Hyperband.** ``TRIAL_BUDGET`` is a relative number,
    users can interpret them as number of epochs, number of mini-batches, running time, etc.

    Here is a concrete example of ``R=81`` and ``eta=3``:

    .. list-table::
        :header-rows: 1
        :widths: auto

        * -
          - s=4
          - s=3
          - s=2
          - s=1
          - s=0
        * - i
          - n r
          - n r
          - n r
          - n r
          - n r
        * - 0
          - 81 1
          - 27 3
          - 9 9
          - 6 27
          - 5 81
        * - 1
          - 27 3
          - 9 9
          - 3 27
          - 2 81
          -
        * - 2
          - 9 9
          - 3 27
          - 1 81
          -
          -
        * - 3
          - 3 27
          - 1 81
          -
          -
          -
        * - 4
          - 1 81
          -
          -
          -
          -


    ``s`` means bucket, ``n`` means the number of configurations that are generated,
    the corresponding ``r`` means how many budgets these configurations run.
    ``i`` means round, for example, bucket 4 has 5 rounds, bucket 3 has 4 rounds.

    A complete example can be found :githublink:`examples/trials/mnist-advisor`.

    Parameters
    ----------
    optimize_mode: str
        Optimize mode, 'maximize' or 'minimize'.

    R: int
        The maximum amount of budget that can be allocated to a single configuration.
        Here, trial budget could mean the number of epochs, number of mini-batches, etc.,
        depending on how users interpret it.
        Each trial should use ``TRIAL_BUDGET`` to control how long it runs.

    eta: int
        The variable that controls the proportion of configurations discarded in each round of SuccessiveHalving.
        ``1/eta`` configurations will survive and rerun using more budgets in each round.

    exec_mode: str
        Execution mode, 'serial' or 'parallelism'.
        If 'parallelism', the tuner will try to use available resources to start new bucket immediately.
        If 'serial', the tuner will only start new bucket after the current bucket is done.


    Notes
    -----

    First, Hyperband an example of how to write an autoML algorithm based on MsgDispatcherBase,
    rather than based on Tuner and Assessor. Hyperband is implemented in this way
    because it integrates the functions of both Tuner and Assessor,thus, we call it Advisor.

    Second, this implementation fully leverages Hyperband's internal parallelism.
    Specifically, the next bucket is not started strictly after the current bucket.
    Instead, it starts when there are available resources. If you want to use full parallelism mode,
    set ``exec_mode`` to ``parallelism``.

    Or if you want to set ``exec_mode`` with ``serial`` according to the original algorithm.
    In this mode, the next bucket will start strictly after the current bucket.

    ``parallelism`` mode may lead to multiple unfinished buckets,
    in contrast, there is at most one unfinished bucket under ``serial`` mode.
    The advantage of ``parallelism`` mode is to make full use of resources,
    which may reduce the experiment duration multiple times.
    """

    def __init__(self, optimize_mode='maximize', R=60, eta=3, exec_mode='parallelism'):
        if False:
            while True:
                i = 10
        'B = (s_max + 1)R'
        super(Hyperband, self).__init__()
        self.R = R
        self.eta = eta
        self.brackets = dict()
        self.generated_hyper_configs = []
        self.completed_hyper_configs = []
        self.s_max = math.floor(math.log(self.R, self.eta) + _epsilon)
        self.curr_s = self.s_max
        self.curr_hb = 0
        self.exec_mode = exec_mode
        self.curr_bracket_id = None
        self.searchspace_json = None
        self.random_state = None
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.credit = 0
        self.job_id_para_id_map = dict()

    def handle_initialize(self, data):
        if False:
            return 10
        'callback for initializing the advisor\n        Parameters\n        ----------\n        data: dict\n            search space\n        '
        self.handle_update_search_space(data)
        self.send(CommandType.Initialized, '')

    def handle_request_trial_jobs(self, data):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        data: int\n            number of trial jobs\n        '
        self.credit += data
        for _ in range(self.credit):
            self._request_one_trial_job()

    def _request_one_trial_job(self):
        if False:
            i = 10
            return i + 15
        ret = self._get_one_trial_job()
        if ret is not None:
            self.send(CommandType.NewTrialJob, nni.dump(ret))
            self.credit -= 1

    def _get_one_trial_job(self):
        if False:
            return 10
        'get one trial job, i.e., one hyperparameter configuration.'
        if not self.generated_hyper_configs:
            if self.exec_mode == 'parallelism' or (self.exec_mode == 'serial' and (self.curr_bracket_id is None or self.brackets[self.curr_bracket_id].is_completed())):
                if self.curr_s < 0:
                    self.curr_s = self.s_max
                    self.curr_hb += 1
                _logger.debug('create a new bracket, self.curr_hb=%d, self.curr_s=%d', self.curr_hb, self.curr_s)
                self.curr_bracket_id = '{}-{}'.format(self.curr_hb, self.curr_s)
                self.brackets[self.curr_bracket_id] = Bracket(self.curr_bracket_id, self.curr_s, self.s_max, self.eta, self.R, self.optimize_mode)
                (next_n, next_r) = self.brackets[self.curr_bracket_id].get_n_r()
                _logger.debug('new bracket, next_n=%d, next_r=%d', next_n, next_r)
                assert self.searchspace_json is not None and self.random_state is not None
                generated_hyper_configs = self.brackets[self.curr_bracket_id].get_hyperparameter_configurations(next_n, next_r, self.searchspace_json, self.random_state)
                self.generated_hyper_configs = generated_hyper_configs.copy()
                self.curr_s -= 1
            else:
                ret = {'parameter_id': '-1_0_0', 'parameter_source': 'algorithm', 'parameters': ''}
                self.send(CommandType.NoMoreTrialJobs, nni.dump(ret))
                return None
        assert self.generated_hyper_configs
        params = self.generated_hyper_configs.pop(0)
        ret = {'parameter_id': params[0], 'parameter_source': 'algorithm', 'parameters': params[1]}
        return ret

    def handle_update_search_space(self, data):
        if False:
            return 10
        'data: JSON object, which is search space\n        '
        validate_search_space(data)
        self.searchspace_json = data
        self.random_state = np.random.RandomState()

    def _handle_trial_end(self, parameter_id):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        parameter_id: parameter id of the finished config\n        '
        (bracket_id, i, _) = parameter_id.split('_')
        hyper_configs = self.brackets[bracket_id].inform_trial_end(int(i))
        if hyper_configs is not None:
            _logger.debug('bracket %s next round %s, hyper_configs: %s', bracket_id, i, hyper_configs)
            self.generated_hyper_configs = self.generated_hyper_configs + hyper_configs
        for _ in range(self.credit):
            self._request_one_trial_job()

    def handle_trial_end(self, data):
        if False:
            i = 10
            return i + 15
        "\n        Parameters\n        ----------\n        data: dict()\n            it has three keys: trial_job_id, event, hyper_params\n            trial_job_id: the id generated by training service\n            event: the job's state\n            hyper_params: the hyperparameters (a string) generated and returned by tuner\n        "
        hyper_params = nni.load(data['hyper_params'])
        if self.is_created_in_previous_exp(hyper_params['parameter_id']):
            return
        self._handle_trial_end(hyper_params['parameter_id'])
        if data['trial_job_id'] in self.job_id_para_id_map:
            del self.job_id_para_id_map[data['trial_job_id']]

    def handle_report_metric_data(self, data):
        if False:
            i = 10
            return i + 15
        "\n        Parameters\n        ----------\n        data:\n            it is an object which has keys 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.\n\n        Raises\n        ------\n        ValueError\n            Data type not supported\n        "
        if self.is_created_in_previous_exp(data['parameter_id']):
            return
        if 'value' in data:
            data['value'] = nni.load(data['value'])
        if data['type'] == MetricType.REQUEST_PARAMETER:
            assert multi_phase_enabled()
            assert data['trial_job_id'] is not None
            assert data['parameter_index'] is not None
            assert data['trial_job_id'] in self.job_id_para_id_map
            self._handle_trial_end(self.job_id_para_id_map[data['trial_job_id']])
            ret = self._get_one_trial_job()
            if data['trial_job_id'] is not None:
                ret['trial_job_id'] = data['trial_job_id']
            if data['parameter_index'] is not None:
                ret['parameter_index'] = data['parameter_index']
            self.job_id_para_id_map[data['trial_job_id']] = ret['parameter_id']
            self.send(CommandType.SendTrialJobParameter, nni.dump(ret))
        else:
            value = extract_scalar_reward(data['value'])
            (bracket_id, i, _) = data['parameter_id'].split('_')
            if data['trial_job_id'] in self.job_id_para_id_map:
                assert self.job_id_para_id_map[data['trial_job_id']] == data['parameter_id']
            else:
                self.job_id_para_id_map[data['trial_job_id']] = data['parameter_id']
            if data['type'] == MetricType.FINAL:
                self.brackets[bracket_id].set_config_perf(int(i), data['parameter_id'], sys.maxsize, value)
                self.completed_hyper_configs.append(data)
            elif data['type'] == MetricType.PERIODICAL:
                self.brackets[bracket_id].set_config_perf(int(i), data['parameter_id'], data['sequence'], value)
            else:
                raise ValueError('Data type not supported: {}'.format(data['type']))

    def handle_add_customized_trial(self, data):
        if False:
            i = 10
            return i + 15
        global _next_parameter_id
        previous_max_param_id = self.recover_parameter_id(data)
        _next_parameter_id = previous_max_param_id + 1

    def handle_import_data(self, data):
        if False:
            while True:
                i = 10
        pass
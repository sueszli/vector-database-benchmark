"""
evolution_tuner.py
"""
from __future__ import annotations
import copy
import random
import logging
from collections import deque
import numpy as np
from schema import Schema, Optional
import nni
from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward, split_index, json2parameter, json2space
logger = logging.getLogger(__name__)

class Individual:
    """
    Individual class to store the indv info.

    Parameters
    ----------
    config : str, default = None
        Search space.
    info : str, default = None
        The str to save information of individual.
    result : float, None = None
        The final metric of a individual.
    """

    def __init__(self, config=None, info=None, result=None):
        if False:
            print('Hello World!')
        self.config = config
        self.result = result
        self.info = info

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'info: ' + str(self.info) + ', config :' + str(self.config) + ', result: ' + str(self.result)

class EvolutionClassArgsValidator(ClassArgsValidator):

    def validate_class_args(self, **kwargs):
        if False:
            i = 10
            return i + 15
        Schema({'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'), Optional('population_size'): self.range('population_size', int, 0, 99999)}).validate(kwargs)

class EvolutionTuner(Tuner):
    """
    Naive Evolution comes from `Large-Scale Evolution of Image Classifiers <https://arxiv.org/pdf/1703.01041.pdf>`__
    It randomly initializes a population based on the search space.
    For each generation, it chooses better ones and does some mutation.
    (e.g., changes a hyperparameter, adds/removes one layer, etc.) on them to get the next generation.
    Naive Evolution requires many trials to works but it’s very simple and it’s easily expanded with new features.

    Examples
    --------

    .. code-block::

        config.tuner.name = 'Evolution'
        config.tuner.class_args = {
                'optimize_mode': 'maximize',
                'population_size': 100
        }

    Parameters
    ----------
    optimize_mode: str
        Optimize mode, 'maximize' or 'minimize'.
        If 'maximize', the tuner will try to maximize metrics. If 'minimize', the tuner will try to minimize metrics.
    population_size: int
        The initial size of the population (trial num) in the evolution tuner(default=32).
        The larger population size, the better evolution performance.
        It's suggested that ``population_size`` be much larger than ``concurrency`` so users can get the most out of the algorithm.
        And at least ``concurrency``, or the tuner will fail on its first generation of parameters.
    """

    def __init__(self, optimize_mode='maximize', population_size=32):
        if False:
            i = 10
            return i + 15
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.population_size = population_size
        self.searchspace_json = None
        self.running_trials = {}
        self.num_running_trials = 0
        self.random_state = None
        self.population = None
        self.space = None
        self.credit = 0
        self.send_trial_callback = None
        self.param_ids = deque()

    def update_search_space(self, search_space):
        if False:
            print('Hello World!')
        '\n        Update search space.\n        Search_space contains the information that user pre-defined.\n\n        Parameters\n        ----------\n\n        search_space : dict\n        '
        self.searchspace_json = search_space
        self.space = json2space(self.searchspace_json)
        self.random_state = np.random.RandomState()
        self.population = []
        for _ in range(self.population_size):
            self._random_generate_individual()

    def trial_end(self, parameter_id, success, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        To deal with trial failure. If a trial fails,\n        random generate the parameters and add into the population.\n\n        Parameters\n        ----------\n\n        parameter_id : int\n            Unique identifier for hyper-parameters used by this trial.\n        success : bool\n            True if the trial successfully completed; False if failed or terminated.\n        **kwargs\n            Not used\n        '
        self.num_running_trials -= 1
        logger.info('trial (%d) end', parameter_id)
        if not success:
            self.running_trials.pop(parameter_id)
            self._random_generate_individual()
        if self.credit > 1:
            param_id = self.param_ids.popleft()
            config = self._generate_individual(param_id)
            logger.debug('Send new trial (%d, %s) for reducing credit', param_id, config)
            self.send_trial_callback(param_id, config)
            self.credit -= 1
            self.num_running_trials += 1

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.\n\n        Parameters\n        ----------\n\n        parameter_id_list : list of int\n            Unique identifiers for each set of requested hyper-parameters.\n        **kwargs\n            Not used\n\n        Returns\n        -------\n        list\n            A list of newly generated configurations\n        '
        result = []
        if 'st_callback' in kwargs:
            self.send_trial_callback = kwargs['st_callback']
        else:
            logger.warning('Send trial callback is not found in kwargs. Evolution tuner might not work properly.')
        for parameter_id in parameter_id_list:
            had_exception = False
            try:
                logger.debug('generating param for %s', parameter_id)
                res = self.generate_parameters(parameter_id, **kwargs)
                self.num_running_trials += 1
            except nni.NoMoreTrialError:
                had_exception = True
            if not had_exception:
                result.append(res)
        return result

    def _random_generate_individual(self):
        if False:
            for i in range(10):
                print('nop')
        is_rand = dict()
        for item in self.space:
            is_rand[item] = True
        config = json2parameter(self.searchspace_json, is_rand, self.random_state)
        self.population.append(Individual(config=config))

    def _generate_individual(self, parameter_id):
        if False:
            print('Hello World!')
        '\n        This function will generate the config for a trial.\n        If at the first generation, randomly generates individuals to satisfy self.population_size.\n        Otherwise, random choose a pair of individuals and compare their fitnesses.\n        The worst of the pair will be removed. Copy the best of the pair and mutate it to generate a new individual.\n\n        Parameters\n        ----------\n\n        parameter_id : int\n\n        Returns\n        -------\n        dict\n            A group of candidate parameters that evolution tuner generated.\n        '
        pos = -1
        for i in range(len(self.population)):
            if self.population[i].result is None:
                pos = i
                break
        if pos != -1:
            indiv = copy.deepcopy(self.population[pos])
            self.population.pop(pos)
        else:
            random.shuffle(self.population)
            if len(self.population) > 1 and self.population[0].result < self.population[1].result:
                self.population[0] = self.population[1]
            space = json2space(self.searchspace_json, self.population[0].config)
            is_rand = dict()
            mutation_pos = space[random.randint(0, len(space) - 1)]
            for i in range(len(self.space)):
                is_rand[self.space[i]] = self.space[i] == mutation_pos
            config = json2parameter(self.searchspace_json, is_rand, self.random_state, self.population[0].config)
            if len(self.population) > 1:
                self.population.pop(1)
            indiv = Individual(config=config)
        self.running_trials[parameter_id] = indiv
        config = split_index(indiv.config)
        return config

    def generate_parameters(self, parameter_id, **kwargs):
        if False:
            print('Hello World!')
        '\n        This function will returns a dict of trial (hyper-)parameters.\n        If no trial configration for now, self.credit plus 1 to send the config later\n\n        Parameters\n        ----------\n\n        parameter_id : int\n\n        Returns\n        -------\n\n        dict\n            One newly generated configuration.\n        '
        if not self.population:
            raise RuntimeError('The population is empty')
        if self.num_running_trials >= self.population_size:
            logger.warning('No enough trial config, population_size is suggested to be larger than trialConcurrency')
            self.credit += 1
            self.param_ids.append(parameter_id)
            raise nni.NoMoreTrialError('no more parameters now.')
        return self._generate_individual(parameter_id)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        if False:
            return 10
        '\n        Record the result from a trial\n\n        Parameters\n        ----------\n\n        parameter_id : int\n        parameters : dict\n        value : dict/float\n            if value is dict, it should have "default" key.\n            value is final metrics of the trial.\n        '
        reward = extract_scalar_reward(value)
        if parameter_id not in self.running_trials:
            raise RuntimeError('Received parameter_id %s not in running_trials.', parameter_id)
        config = self.running_trials[parameter_id].config
        self.running_trials.pop(parameter_id)
        if self.optimize_mode == OptimizeMode.Minimize:
            reward = -reward
        indiv = Individual(config=config, result=reward)
        self.population.append(indiv)

    def import_data(self, data):
        if False:
            while True:
                i = 10
        pass
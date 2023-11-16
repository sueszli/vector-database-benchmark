import glob
import json
import logging
import os
import random
import shutil
import sys
from collections import deque
from unittest import TestCase, main
from nni.algorithms.hpo.batch_tuner import BatchTuner
from nni.algorithms.hpo.dngo_tuner import DNGOTuner
from nni.algorithms.hpo.evolution_tuner import EvolutionTuner
from nni.algorithms.hpo.gp_tuner import GPTuner
from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.algorithms.hpo.metis_tuner import MetisTuner
from nni.algorithms.hpo.pbt_tuner import PBTTuner
from nni.algorithms.hpo.random_tuner import RandomTuner
from nni.algorithms.hpo.regularized_evolution_tuner import RegularizedEvolutionTuner
from nni.algorithms.hpo.tpe_tuner import TpeTuner
from nni.runtime.msg_dispatcher import _pack_parameter, MsgDispatcher
smac_imported = False
if sys.platform != 'win32' and sys.version_info < (3, 9):
    from nni.algorithms.hpo.smac_tuner import SMACTuner
    smac_imported = True
from nni.tuner import Tuner
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_tuner')

class BuiltinTunersTestCase(TestCase):
    """
    Targeted at testing functions of built-in tuners, including
        - [ ] load_checkpoint
        - [ ] save_checkpoint
        - [X] update_search_space
        - [X] generate_multiple_parameters
        - [X] import_data
        - [ ] trial_end
        - [x] receive_trial_result
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_round = 3
        self.params_each_round = 50
        self.exhaustive = False

    def send_trial_callback(self, param_queue):
        if False:
            while True:
                i = 10

        def receive(*args):
            if False:
                while True:
                    i = 10
            param_queue.append(tuple(args))
        return receive

    def send_trial_result(self, tuner, parameter_id, parameters, metrics):
        if False:
            for i in range(10):
                print('nop')
        if parameter_id % 2 == 1:
            metrics = {'default': metrics, 'extra': 'hello'}
        tuner.receive_trial_result(parameter_id, parameters, metrics)
        tuner.trial_end(parameter_id, True)

    def search_space_test_one(self, tuner_factory, search_space, nas=False):
        if False:
            return 10
        tuner = tuner_factory()
        self.assertIsInstance(tuner, Tuner)
        tuner.update_search_space(search_space)
        for i in range(self.test_round):
            queue = deque()
            parameters = tuner.generate_multiple_parameters(list(range(i * self.params_each_round, (i + 1) * self.params_each_round)), st_callback=self.send_trial_callback(queue))
            logger.debug(parameters)
            check_range = lambda parameters, search_space: self.nas_check_range(parameters, search_space) if nas else self.check_range(parameters, search_space)
            check_range(parameters, search_space)
            for k in range(min(len(parameters), self.params_each_round)):
                self.send_trial_result(tuner, self.params_each_round * i + k, parameters[k], random.uniform(-100, 100))
            while queue:
                (id_, params) = queue.popleft()
                check_range([params], search_space)
                self.send_trial_result(tuner, id_, params, random.uniform(-100, 100))
            if not parameters and (not self.exhaustive):
                raise ValueError('No parameters generated')

    def check_range(self, generated_params, search_space):
        if False:
            while True:
                i = 10
        EPS = 1e-06
        for param in generated_params:
            if self._testMethodName == 'test_batch':
                param = {list(search_space.keys())[0]: param}
            for (k, v) in param.items():
                if k == 'load_checkpoint_dir' or k == 'save_checkpoint_dir':
                    self.assertIsInstance(v, str)
                    continue
                if k.startswith('_mutable_layer'):
                    (_, block, layer, choice) = k.split('/')
                    cand = search_space[block]['_value'][layer].get(choice)
                    if choice == 'layer_choice':
                        self.assertIn(v, cand)
                    if choice == 'optional_input_size':
                        if isinstance(cand, int):
                            self.assertEqual(v, cand)
                        else:
                            self.assertGreaterEqual(v, cand[0])
                            self.assertLessEqual(v, cand[1])
                    if choice == 'optional_inputs':
                        pass
                    continue
                item = search_space[k]
                if item['_type'] == 'choice':
                    self.assertIn(v, item['_value'])
                if item['_type'] == 'randint':
                    self.assertIsInstance(v, int)
                if item['_type'] == 'uniform':
                    self.assertIsInstance(v, float)
                if item['_type'] in ('randint', 'uniform', 'quniform', 'loguniform', 'qloguniform'):
                    self.assertGreaterEqual(v, item['_value'][0])
                    self.assertLessEqual(v, item['_value'][1])
                if item['_type'].startswith('q'):
                    multiple = v / item['_value'][2]
                    print(k, v, multiple, item)
                    if item['_value'][0] + EPS < v < item['_value'][1] - EPS:
                        self.assertAlmostEqual(int(round(multiple)), multiple)
                if item['_type'] in ('qlognormal', 'lognormal'):
                    self.assertGreaterEqual(v, 0)
                if item['_type'] == 'mutable_layer':
                    for layer_name in item['_value'].keys():
                        self.assertIn(v[layer_name]['chosen_layer'], item['layer_choice'])

    def nas_check_range(self, generated_params, search_space):
        if False:
            i = 10
            return i + 15
        for params in generated_params:
            for k in params:
                v = params[k]
                items = search_space[k]
                if items['_type'] == 'layer_choice':
                    self.assertIn(v['_value'], items['_value'])
                elif items['_type'] == 'input_choice':
                    for choice in v['_value']:
                        self.assertIn(choice, items['_value']['candidates'])
                else:
                    raise KeyError

    def search_space_test_all(self, tuner_factory, supported_types=None, ignore_types=None, fail_types=None):
        if False:
            for i in range(10):
                print('nop')
        with open(os.path.join(os.path.dirname(__file__), 'assets/search_space.json'), 'r') as fp:
            search_space_all = json.load(fp)
        if supported_types is None:
            supported_types = ['choice', 'randint', 'uniform', 'quniform', 'loguniform', 'qloguniform', 'normal', 'qnormal', 'lognormal', 'qlognormal']
        if fail_types is None:
            fail_types = []
        if ignore_types is None:
            ignore_types = []
        full_supported_search_space = dict()
        for single in search_space_all:
            space = search_space_all[single]
            if any((single.startswith(t) for t in ignore_types)):
                continue
            expected_fail = not any((single.startswith(t) for t in supported_types)) or any((single.startswith(t) for t in fail_types)) or 'fail' in single
            single_search_space = {single: space}
            if not expected_fail:
                self.search_space_test_one(tuner_factory, single_search_space)
                full_supported_search_space.update(single_search_space)
            else:
                with self.assertRaises(Exception, msg='Testing {}'.format(single)) as cm:
                    self.search_space_test_one(tuner_factory, single_search_space)
                logger.info('%s %s %s', tuner_factory, single, cm.exception)
        if not any((t in self._testMethodName for t in ['batch', 'grid_search'])):
            logger.info('Full supported search space: %s', full_supported_search_space)
            self.search_space_test_one(tuner_factory, full_supported_search_space)

    def nas_search_space_test_all(self, tuner_factory):
        if False:
            return 10
        with open(os.path.join(os.path.dirname(__file__), 'assets/classic_nas_search_space.json'), 'r') as fp:
            search_space_all = json.load(fp)
        full_supported_search_space = dict()
        for single in search_space_all:
            space = search_space_all[single]
            single_search_space = {single: space}
            self.search_space_test_one(tuner_factory, single_search_space, nas=True)
            full_supported_search_space.update(single_search_space)
        logger.info('Full supported search space: %s', full_supported_search_space)
        self.search_space_test_one(tuner_factory, full_supported_search_space, nas=True)

    def import_data_test_for_pbt(self):
        if False:
            return 10
        '\n        test1: import data with complete epoch\n        test2: import data with incomplete epoch\n        '
        search_space = {'choice_str': {'_type': 'choice', '_value': ['cat', 'dog', 'elephant', 'cow', 'sheep', 'panda']}}
        all_checkpoint_dir = os.path.expanduser('~/nni/checkpoint/test/')
        population_size = 4
        tuner = PBTTuner(all_checkpoint_dir=all_checkpoint_dir, population_size=population_size)
        self.assertIsInstance(tuner, Tuner)
        tuner.update_search_space(search_space)
        save_dirs = [os.path.join(all_checkpoint_dir, str(i), str(0)) for i in range(population_size)]
        for save_dir in save_dirs:
            os.makedirs(save_dir, exist_ok=True)
        data = [{'parameter': {'choice_str': 'cat', 'save_checkpoint_dir': save_dirs[0]}, 'value': 1.1}, {'parameter': {'choice_str': 'dog', 'save_checkpoint_dir': save_dirs[1]}, 'value': {'default': 1.2, 'tmp': 2}}, {'parameter': {'choice_str': 'cat', 'save_checkpoint_dir': save_dirs[2]}, 'value': 11}, {'parameter': {'choice_str': 'cat', 'save_checkpoint_dir': save_dirs[3]}, 'value': 7}]
        epoch = tuner.import_data(data)
        self.assertEqual(epoch, 1)
        logger.info('Imported data successfully at the beginning')
        shutil.rmtree(all_checkpoint_dir)
        tuner = PBTTuner(all_checkpoint_dir=all_checkpoint_dir, population_size=population_size)
        self.assertIsInstance(tuner, Tuner)
        tuner.update_search_space(search_space)
        for i in range(population_size - 1):
            save_dirs.append(os.path.join(all_checkpoint_dir, str(i), str(1)))
        for save_dir in save_dirs:
            os.makedirs(save_dir, exist_ok=True)
        data = [{'parameter': {'choice_str': 'cat', 'save_checkpoint_dir': save_dirs[0]}, 'value': 1.1}, {'parameter': {'choice_str': 'dog', 'save_checkpoint_dir': save_dirs[1]}, 'value': {'default': 1.2, 'tmp': 2}}, {'parameter': {'choice_str': 'cat', 'save_checkpoint_dir': save_dirs[2]}, 'value': 11}, {'parameter': {'choice_str': 'cat', 'save_checkpoint_dir': save_dirs[3]}, 'value': 7}, {'parameter': {'choice_str': 'cat', 'save_checkpoint_dir': save_dirs[4]}, 'value': 1.1}, {'parameter': {'choice_str': 'dog', 'save_checkpoint_dir': save_dirs[5]}, 'value': {'default': 1.2, 'tmp': 2}}, {'parameter': {'choice_str': 'cat', 'save_checkpoint_dir': save_dirs[6]}, 'value': 11}]
        epoch = tuner.import_data(data)
        self.assertEqual(epoch, 1)
        logger.info('Imported data successfully at the beginning with incomplete epoch')
        shutil.rmtree(all_checkpoint_dir)

    def import_data_test(self, tuner_factory, stype='choice_str', support_middle=True):
        if False:
            while True:
                i = 10
        '\n        import data at the beginning with number value and dict value\n        import data in the middle also with number value and dict value, and duplicate data record\n        generate parameters after data import\n\n        Parameters\n        ----------\n        tuner_factory : lambda\n            a lambda for instantiate a tuner\n        stype : str\n            the value type of hp choice, support "choice_str" and "choice_num"\n        '
        if stype == 'choice_str':
            search_space = {'choice_str': {'_type': 'choice', '_value': ['cat', 'dog', 'elephant', 'cow', 'sheep', 'panda', 'tiger']}}
        elif stype == 'choice_num':
            search_space = {'choice_num': {'_type': 'choice', '_value': [10, 20, 30, 40, 50, 60]}}
        else:
            raise RuntimeError('Unexpected stype')
        tuner = tuner_factory()
        self.assertIsInstance(tuner, Tuner)
        tuner.update_search_space(search_space)
        if stype == 'choice_str':
            data = [{'parameter': {'choice_str': 'cat'}, 'value': 1.1}, {'parameter': {'choice_str': 'dog'}, 'value': {'default': 1.2, 'tmp': 2}}]
        else:
            data = [{'parameter': {'choice_num': 20}, 'value': 1.1}, {'parameter': {'choice_num': 60}, 'value': {'default': 1.2, 'tmp': 2}}]
        tuner.import_data(data)
        logger.info('Imported data successfully at the beginning')
        parameters = tuner.generate_multiple_parameters(list(range(3)))
        for i in range(3):
            tuner.receive_trial_result(i, parameters[i], random.uniform(-100, 100))
        if not support_middle:
            return
        if stype == 'choice_str':
            data = [{'parameter': {'choice_str': 'cat'}, 'value': 1.1}, {'parameter': {'choice_str': 'dog'}, 'value': {'default': 1.2, 'tmp': 2}}, {'parameter': {'choice_str': 'cow'}, 'value': 1.3}]
        else:
            data = [{'parameter': {'choice_num': 20}, 'value': 1.1}, {'parameter': {'choice_num': 60}, 'value': {'default': 1.2, 'tmp': 2}}, {'parameter': {'choice_num': 50}, 'value': 1.3}]
        tuner.import_data(data)
        logger.info('Imported data successfully in the middle')
        parameters = tuner.generate_multiple_parameters([3])
        tuner.receive_trial_result(3, parameters[0], random.uniform(-100, 100))

    def test_grid_search(self):
        if False:
            for i in range(10):
                print('nop')
        self.exhaustive = True
        tuner_fn = lambda : GridSearchTuner()
        self.search_space_test_all(tuner_fn)
        self.import_data_test(tuner_fn, support_middle=False)

    def test_tpe(self):
        if False:
            return 10
        self.exhaustive = True
        tuner_fn = TpeTuner
        self.search_space_test_all(TpeTuner)
        self.import_data_test(tuner_fn)

    def test_random_search(self):
        if False:
            while True:
                i = 10
        self.exhaustive = True
        tuner_fn = RandomTuner
        self.search_space_test_all(tuner_fn)
        self.import_data_test(tuner_fn)

    def test_anneal(self):
        if False:
            return 10
        tuner_fn = lambda : HyperoptTuner('anneal')
        self.search_space_test_all(tuner_fn)
        self.import_data_test(tuner_fn)

    def test_smac(self):
        if False:
            return 10
        if not smac_imported:
            return
        tuner_fn = lambda : SMACTuner()
        self.search_space_test_all(tuner_fn, supported_types=['choice', 'randint', 'uniform', 'quniform', 'loguniform'])
        self.import_data_test(tuner_fn)

    def test_batch(self):
        if False:
            for i in range(10):
                print('nop')
        self.exhaustive = True
        tuner_fn = lambda : BatchTuner()
        self.search_space_test_all(tuner_fn, supported_types=['choice'])
        self.import_data_test(tuner_fn)

    def test_evolution(self):
        if False:
            for i in range(10):
                print('nop')
        tuner_fn = lambda : EvolutionTuner(population_size=100)
        self.search_space_test_all(tuner_fn)
        self.import_data_test(tuner_fn)

    def test_gp(self):
        if False:
            while True:
                i = 10
        self.test_round = 1
        tuner_fn = lambda : GPTuner()
        self.search_space_test_all(tuner_fn, supported_types=['choice', 'randint', 'uniform', 'quniform', 'loguniform', 'qloguniform'], ignore_types=['normal', 'lognormal', 'qnormal', 'qlognormal'], fail_types=['choice_str', 'choice_mixed'])
        self.import_data_test(tuner_fn, 'choice_num')

    def test_metis(self):
        if False:
            i = 10
            return i + 15
        self.test_round = 1
        tuner_fn = lambda : MetisTuner()
        self.search_space_test_all(tuner_fn, supported_types=['choice', 'randint', 'uniform', 'quniform'], fail_types=['choice_str', 'choice_mixed'])
        self.import_data_test(tuner_fn, 'choice_num')

    def test_networkmorphism(self):
        if False:
            print('Hello World!')
        pass

    def test_ppo(self):
        if False:
            print('Hello World!')
        pass

    def test_pbt(self):
        if False:
            print('Hello World!')
        self.search_space_test_all(lambda : PBTTuner(all_checkpoint_dir=os.path.expanduser('~/nni/checkpoint/test/'), population_size=12))
        self.search_space_test_all(lambda : PBTTuner(all_checkpoint_dir=os.path.expanduser('~/nni/checkpoint/test/'), population_size=100))
        self.import_data_test_for_pbt()

    def test_dngo(self):
        if False:
            i = 10
            return i + 15
        tuner_fn = lambda : DNGOTuner(trials_per_update=100, num_epochs_per_training=1)
        self.search_space_test_all(tuner_fn, fail_types=['choice_str', 'choice_mixed', 'normal', 'lognormal', 'qnormal', 'qlognormal'])
        self.import_data_test(tuner_fn, stype='choice_num')

    def test_regularized_evolution_tuner(self):
        if False:
            print('Hello World!')
        tuner_fn = lambda : RegularizedEvolutionTuner()
        self.nas_search_space_test_all(tuner_fn)

    def tearDown(self):
        if False:
            print('Hello World!')
        file_list = glob.glob('smac3*') + ['param_config_space.pcs', 'scenario.txt', 'model_path']
        for file in file_list:
            if os.path.exists(file):
                if os.path.isdir(file):
                    shutil.rmtree(file)
                else:
                    os.remove(file)
if __name__ == '__main__':
    main()
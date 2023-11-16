import pytest
from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.orca.automl.search.tensorboardlogger import TensorboardLogger
import numpy as np
import random
import os.path

class TestTensorboardLogger(ZooTestCase):

    def setup_method(self, method):
        if False:
            return 10
        pass

    def teardown_method(self, method):
        if False:
            return 10
        pass

    def test_tblogger_valid_type(self):
        if False:
            while True:
                i = 10
        trail_num = 100
        test_config = {}
        test_metric = {}
        for i in range(trail_num):
            test_config['run_{}'.format(i)] = {'config_good': random.randint(8, 96), 'config_unstable': None if random.random() < 0.5 else 1, 'config_bad': None}
            test_metric['run_{}'.format(i)] = {'matrix_good': random.randint(0, 100) / 100, 'matrix_unstable': np.nan if random.random() < 0.5 else 1, 'matrix_bad': np.nan}
        logger = TensorboardLogger(os.path.abspath(os.path.expanduser('~/test_tbxlogger')))
        logger.run(test_config, test_metric)
        logger.close()

    def test_tblogger_keys(self):
        if False:
            print('Hello World!')
        test_config = {'run1': {'lr': 0.01}}
        test_metric = {'run2': {'lr': 0.02}}
        logger = TensorboardLogger(os.path.abspath(os.path.expanduser('~/test_tbxlogger')))
        with pytest.raises(Exception):
            logger.run(test_config, test_metric)
        logger.close()
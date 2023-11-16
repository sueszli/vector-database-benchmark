import random
import pytest
from easydict import EasyDict
from ditk import logging
from ding.utils.log_helper import build_logger, pretty_print
from ding.utils.file_helper import remove_file
cfg = EasyDict({'env': {}, 'env_num': 4, 'common': {'save_path': './summary_log', 'load_path': '', 'name': 'fakeLog', 'only_evaluate': False}, 'logger': {'print_freq': 10, 'save_freq': 200, 'eval_freq': 200}, 'data': {'train': {}, 'eval': {}}, 'learner': {'log_freq': 100}})

@pytest.mark.unittest
class TestLogger:

    def test_pretty_print(self):
        if False:
            while True:
                i = 10
        pretty_print(cfg)

    def test_logger(self):
        if False:
            print('Hello World!')
        (logger, tb_logger) = build_logger(cfg.common.save_path, name='fake_test', need_tb=True, text_level=logging.DEBUG)
        variables = {'aa': 3.0, 'bb': 4, 'cc': 30000.0}
        logger.info("I'm an info")
        logger.debug("I'm a bug")
        logger.error("I'm an error")
        logger.info(logger.get_tabulate_vars(variables))
        for i in range(10):
            new_vars = {k: v * (i + random.random()) for (k, v) in variables.items()}
            for (k, v) in new_vars.items():
                tb_logger.add_scalar(k, v, i)
        remove_file(cfg.common.save_path)
        tb_logger.close()
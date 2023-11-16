import logging
import unittest
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.log_util import logger

class TestFleetLog(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        fleet.init(log_level='DEBUG')

    def test_log_level(self):
        if False:
            for i in range(10):
                print('nop')
        assert fleet.get_log_level_code() == logging._nameToLevel['DEBUG']
        assert logger.getEffectiveLevel() == logging._nameToLevel['DEBUG']
        fleet.set_log_level('WARNING')
        debug1 = fleet.get_log_level_code()
        debug2 = logging._nameToLevel['WARNING']
        assert debug1 == debug2
        fleet.set_log_level(debug2)
        assert logger.getEffectiveLevel() == logging._nameToLevel['WARNING']
import pytest
from unittest import TestCase
from bigdl.chronos.aiops.config_generator import ConfigGenerator, triggerbyclock
import time
from .. import op_diff_set_all

class TestConfigGenerator(TestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        pass

    def teardown_method(self, method):
        if False:
            while True:
                i = 10
        pass

    @op_diff_set_all
    def test_triggerbyclock(self):
        if False:
            for i in range(10):
                print('nop')

        class MyConfigGenerator(ConfigGenerator):

            def __init__(self, sweetpoint):
                if False:
                    for i in range(10):
                        print('nop')
                self.sweetpoint = sweetpoint
                super().__init__()

            def genConfig(self):
                if False:
                    i = 10
                    return i + 15
                return self.sweetpoint

            @triggerbyclock(2)
            def update_sweetpoint(self):
                if False:
                    print('Hello World!')
                self.sweetpoint += 1
        mycg = MyConfigGenerator(5)
        time.sleep(4)
        assert mycg.genConfig() > 5
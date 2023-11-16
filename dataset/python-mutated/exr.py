from ..test_config_base import TestConfigBase

class TestConfig(TestConfigBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(task_settings='exr')
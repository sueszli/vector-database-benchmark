from ..test_config_base import TestConfigBase

class TestConfig(TestConfigBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(task_settings='jpeg')
from os import path
from ..util import LanghostTest

class RuntimeSettingsTest(LanghostTest):

    def test_runtime_settings(self):
        if False:
            while True:
                i = 10
        self.run_test(program=path.join(self.base_path(), 'runtime_settings'), organization='myorg', project='myproject', stack='mystack', config={'test:known': 'knownkey', 'test:lowercase_true': 'true', 'test:uppercase_true': 'True', 'test:lowercase_false': 'false', 'test:uppercase_false': 'False', 'test:not_a_bool': 'DBBool'}, expected_resource_count=0)
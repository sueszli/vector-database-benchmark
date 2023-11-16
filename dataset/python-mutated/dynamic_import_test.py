from helpers import LuigiTestCase, temporary_unloaded_module
import luigi
import luigi.interface
CONTENTS = b'\nimport luigi\n\nclass FooTask(luigi.Task):\n    x = luigi.IntParameter()\n\n    def run(self):\n        luigi._testing_glob_var = self.x\n'

class CmdlineTest(LuigiTestCase):

    def test_dynamic_loading(self):
        if False:
            while True:
                i = 10
        with temporary_unloaded_module(CONTENTS) as temp_module_name:
            luigi.interface.run(['--module', temp_module_name, 'FooTask', '--x', '123', '--local-scheduler', '--no-lock'])
            self.assertEqual(luigi._testing_glob_var, 123)
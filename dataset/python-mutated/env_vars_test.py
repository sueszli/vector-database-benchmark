import os
import unittest
from google.appengine.ext import testbed

class EnvVarsTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.testbed = testbed.Testbed()
        self.testbed.activate()
        self.testbed.setup_env(app_id='your-app-id', my_config_setting='example', overwrite=True)

    def tearDown(self):
        if False:
            return 10
        self.testbed.deactivate()

    def testEnvVars(self):
        if False:
            return 10
        self.assertEqual(os.environ['APPLICATION_ID'], 'your-app-id')
        self.assertEqual(os.environ['MY_CONFIG_SETTING'], 'example')
if __name__ == '__main__':
    unittest.main()
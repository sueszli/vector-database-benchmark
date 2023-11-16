from twisted.trial import unittest
from buildbot.configurators import ConfiguratorBase
from buildbot.test.util import configurators

class ConfiguratorBaseTests(configurators.ConfiguratorMixin, unittest.SynchronousTestCase):
    ConfiguratorClass = ConfiguratorBase

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        self.setupConfigurator()
        self.assertEqual(self.config_dict, {'schedulers': [], 'protocols': {}, 'workers': [], 'builders': []})
        self.assertEqual(self.configurator.workers, [])
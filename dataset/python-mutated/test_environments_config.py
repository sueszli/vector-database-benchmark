from golem.environments.environment import Environment
from golem.environments.environmentsconfig import EnvironmentsConfig
from golem.environments.environmentsmanager import EnvironmentsManager
from golem.tools.testdirfixture import TestDirFixture

class TestEnvironmentsConfig(TestDirFixture):

    def test_load_config(self):
        if False:
            return 10
        envs = {'test-env': ('Test Env', True)}
        config = EnvironmentsConfig.load_config(envs, self.path)
        assert config

    def test_load_config_empty(self):
        if False:
            while True:
                i = 10
        envs = {}
        config = EnvironmentsConfig.load_config(envs, self.path)
        assert config

    def test_load_config_manager(self):
        if False:
            for i in range(10):
                print('nop')
        mgr = EnvironmentsManager()
        env = Environment()
        mgr.environments[env.get_id()] = env
        mgr.load_config(self.path)
        assert mgr.env_config

    def test_load_config_manager_empty(self):
        if False:
            while True:
                i = 10
        mgr = EnvironmentsManager()
        mgr.load_config(self.path)
        assert mgr.env_config
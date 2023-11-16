import synapse.app.homeserver
from synapse.config._base import ConfigError
from tests.config.utils import ConfigFileTestCase

class HomeserverAppStartTestCase(ConfigFileTestCase):

    def test_wrong_start_caught(self) -> None:
        if False:
            print('Hello World!')
        self.generate_config()
        self.add_lines_to_config(['  '])
        self.add_lines_to_config(['worker_app: test_worker_app'])
        self.add_lines_to_config(['worker_log_config: /data/logconfig.config'])
        self.add_lines_to_config(['instance_map:'])
        self.add_lines_to_config(['  main:', '    host: 127.0.0.1', '    port: 1234'])
        with self.assertRaises(ConfigError):
            synapse.app.homeserver.setup(['-c', self.config_file])
from golem.core.simpleconfig import SimpleConfig, ConfigEntry
import logging
from os import path
ENV_VERSION = 1.01
CONFIG_FILENAME = 'environments.ini'
logger = logging.getLogger(__name__)

class NodeConfig(object):

    def __init__(self, environments):
        if False:
            print('Hello World!')
        self._section = 'Node'
        for (env_id, (env_name, supported)) in environments.items():
            ConfigEntry.create_property(self.section(), env_id.lower(), int(supported), self, env_name)

    def section(self):
        if False:
            i = 10
            return i + 15
        return self._section

class EnvironmentsConfig(object):
    """Manage config file describing whether user want to compute tasks from given environment or not."""

    @classmethod
    def load_config(cls, environments, datadir) -> 'EnvironmentsConfig':
        if False:
            for i in range(10):
                print('nop')
        cfg_file = path.join(datadir, CONFIG_FILENAME)
        cfg = SimpleConfig(NodeConfig(environments), cfg_file, refresh=False)
        return EnvironmentsConfig(cfg, cfg_file)

    def __init__(self, cfg, cfg_file):
        if False:
            while True:
                i = 10
        self._cfg = cfg
        self.cfg_file = cfg_file

    def get_config_entries(self):
        if False:
            print('Hello World!')
        return self._cfg.get_node_config()

    def change_config(self) -> 'EnvironmentsConfig':
        if False:
            i = 10
            return i + 15
        return EnvironmentsConfig(SimpleConfig(self._cfg.get_node_config(), self.cfg_file, refresh=True), self.cfg_file)
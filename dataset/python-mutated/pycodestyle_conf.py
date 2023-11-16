import pycodestyle
from pylsp._utils import find_parents
from .source import ConfigSource
CONFIG_KEY = 'pycodestyle'
USER_CONFIGS = [pycodestyle.USER_CONFIG] if pycodestyle.USER_CONFIG else []
PROJECT_CONFIGS = ['pycodestyle.cfg', 'setup.cfg', 'tox.ini']
OPTIONS = [('exclude', 'plugins.pycodestyle.exclude', list), ('filename', 'plugins.pycodestyle.filename', list), ('hang-closing', 'plugins.pycodestyle.hangClosing', bool), ('ignore', 'plugins.pycodestyle.ignore', list), ('max-line-length', 'plugins.pycodestyle.maxLineLength', int), ('indent-size', 'plugins.pycodestyle.indentSize', int), ('select', 'plugins.pycodestyle.select', list), ('aggressive', 'plugins.pycodestyle.aggressive', int)]

class PyCodeStyleConfig(ConfigSource):

    def user_config(self):
        if False:
            print('Hello World!')
        config = self.read_config_from_files(USER_CONFIGS)
        return self.parse_config(config, CONFIG_KEY, OPTIONS)

    def project_config(self, document_path):
        if False:
            for i in range(10):
                print('nop')
        files = find_parents(self.root_path, document_path, PROJECT_CONFIGS)
        config = self.read_config_from_files(files)
        return self.parse_config(config, CONFIG_KEY, OPTIONS)
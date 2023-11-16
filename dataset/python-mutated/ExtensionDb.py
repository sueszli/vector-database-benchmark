from ulauncher.config import PATHS
from ulauncher.utils.json_conf import JsonConf

class ExtensionRecord(JsonConf):
    id = ''
    url = ''
    updated_at = ''
    last_commit = ''
    last_commit_time = ''
    is_enabled = True

class ExtensionDb(JsonConf):

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        super().__setitem__(key, ExtensionRecord(value))

    @classmethod
    def load(cls):
        if False:
            return 10
        return super().load(f'{PATHS.CONFIG}/extensions.json')
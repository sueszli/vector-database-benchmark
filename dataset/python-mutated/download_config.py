import base64
from typing import Dict, Optional
from configobj import ConfigObj
from validate import Validator
from tribler.core.components.libtorrent.settings import DownloadDefaultsSettings, get_default_download_dir
from tribler.core.components.libtorrent.utils.libtorrent_helper import libtorrent as lt
from tribler.core.exceptions import InvalidConfigException
from tribler.core.utilities.install_dir import get_lib_path
from tribler.core.utilities.path_util import Path
from tribler.core.utilities.utilities import bdecode_compat
SPEC_FILENAME = 'download_config.spec'
CONFIG_SPEC_PATH = get_lib_path() / 'components/libtorrent/download_manager' / SPEC_FILENAME
NONPERSISTENT_DEFAULTS = {}

def _from_dict(value: Dict) -> str:
    if False:
        i = 10
        return i + 15
    binary = lt.bencode(value)
    base64_bytes = base64.b64encode(binary)
    return base64_bytes.decode('utf-8')

def _to_dict(value: str) -> Optional[Dict]:
    if False:
        print('Hello World!')
    binary = value.encode('utf-8')
    base64_bytes = base64.b64decode(binary + b'==')
    return bdecode_compat(base64_bytes)

class DownloadConfig:

    def __init__(self, config=None, state_dir=None):
        if False:
            i = 10
            return i + 15
        self.config = config or ConfigObj(configspec=str(CONFIG_SPEC_PATH), default_encoding='utf8')
        self.nonpersistent = NONPERSISTENT_DEFAULTS.copy()
        self.state_dir = state_dir
        self.validate()

    def validate(self):
        if False:
            print('Hello World!')
        "\n        Validate the ConfigObj using Validator.\n\n        Note that `validate()` returns `True` if the ConfigObj is correct and a dictionary with `True` and `False`\n        values for keys who's validation failed if at least one key was found to be incorrect.\n        "
        validator = Validator()
        validation_result = self.config.validate(validator)
        if validation_result is not True:
            raise InvalidConfigException(f'DownloadConfig is invalid: {str(validation_result)}')

    @staticmethod
    def load(config_path=None):
        if False:
            for i in range(10):
                print('nop')
        return DownloadConfig(ConfigObj(infile=Path.fix_win_long_file(config_path), file_error=True, configspec=str(CONFIG_SPEC_PATH), default_encoding='utf-8'))

    @staticmethod
    def from_defaults(settings: DownloadDefaultsSettings, state_dir=None):
        if False:
            return 10
        config = DownloadConfig(state_dir=state_dir)
        config.set_hops(settings.number_hops)
        config.set_safe_seeding(settings.safeseeding_enabled)
        config.set_dest_dir(settings.saveas)
        return config

    def copy(self):
        if False:
            while True:
                i = 10
        return DownloadConfig(ConfigObj(self.config, configspec=str(CONFIG_SPEC_PATH), default_encoding='utf-8'), state_dir=self.state_dir)

    def write(self, filename):
        if False:
            while True:
                i = 10
        self.config.filename = Path.fix_win_long_file(filename)
        self.config.write()

    def set_dest_dir(self, path):
        if False:
            i = 10
            return i + 15
        ' Sets the directory where to save this Download.\n        @param path A path of a directory.\n        '
        path = Path(path).normalize_to(self.state_dir)
        self.config['download_defaults']['saveas'] = str(path)

    def get_dest_dir(self):
        if False:
            while True:
                i = 10
        ' Gets the directory where to save this Download.\n        '
        dest_dir = self.config['download_defaults']['saveas']
        if not dest_dir:
            dest_dir = get_default_download_dir()
            self.set_dest_dir(dest_dir)
        if not Path(dest_dir).is_absolute():
            dest_dir = self.state_dir / dest_dir
        return Path(dest_dir)

    def set_hops(self, hops):
        if False:
            while True:
                i = 10
        self.config['download_defaults']['hops'] = hops

    def get_hops(self):
        if False:
            i = 10
            return i + 15
        return self.config['download_defaults']['hops']

    def set_safe_seeding(self, value):
        if False:
            i = 10
            return i + 15
        self.config['download_defaults']['safe_seeding'] = value

    def get_safe_seeding(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config['download_defaults']['safe_seeding']

    def set_user_stopped(self, value):
        if False:
            while True:
                i = 10
        self.config['download_defaults']['user_stopped'] = value

    def get_user_stopped(self):
        if False:
            while True:
                i = 10
        return self.config['download_defaults']['user_stopped']

    def set_share_mode(self, value):
        if False:
            i = 10
            return i + 15
        self.config['download_defaults']['share_mode'] = value

    def get_share_mode(self):
        if False:
            while True:
                i = 10
        return self.config['download_defaults']['share_mode']

    def set_upload_mode(self, value):
        if False:
            while True:
                i = 10
        self.config['download_defaults']['upload_mode'] = value

    def get_upload_mode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config['download_defaults']['upload_mode']

    def set_time_added(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.config['download_defaults']['time_added'] = value

    def get_time_added(self):
        if False:
            return 10
        return self.config['download_defaults']['time_added']

    def set_selected_files(self, file_indexes):
        if False:
            i = 10
            return i + 15
        ' Select which files in the torrent to download.\n        @param file_indexes List of file indexes as ordered in the torrent (e.g. [0,1])\n        '
        self.config['download_defaults']['selected_file_indexes'] = file_indexes

    def get_selected_files(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns the list of files selected for download.\n        @return A list of file indexes. '
        return self.config['download_defaults']['selected_file_indexes']

    def set_channel_download(self, value):
        if False:
            i = 10
            return i + 15
        self.config['download_defaults']['channel_download'] = value

    def get_channel_download(self):
        if False:
            i = 10
            return i + 15
        return bool(self.config['download_defaults']['channel_download'])

    def set_add_to_channel(self, value):
        if False:
            while True:
                i = 10
        self.config['download_defaults']['add_to_channel'] = value

    def get_add_to_channel(self):
        if False:
            while True:
                i = 10
        return bool(self.config['download_defaults']['add_to_channel'])

    def set_bootstrap_download(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.config['download_defaults']['bootstrap_download'] = value

    def get_bootstrap_download(self):
        if False:
            i = 10
            return i + 15
        return self.config['download_defaults']['bootstrap_download']

    def set_metainfo(self, metainfo: Dict):
        if False:
            for i in range(10):
                print('nop')
        self.config['state']['metainfo'] = _from_dict(metainfo)

    def get_metainfo(self) -> Optional[Dict]:
        if False:
            print('Hello World!')
        return _to_dict(self.config['state']['metainfo'])

    def set_engineresumedata(self, engineresumedata: Dict):
        if False:
            i = 10
            return i + 15
        self.config['state']['engineresumedata'] = _from_dict(engineresumedata)

    def get_engineresumedata(self) -> Optional[Dict]:
        if False:
            while True:
                i = 10
        return _to_dict(self.config['state']['engineresumedata'])
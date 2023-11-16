import binascii
import codecs
from datetime import datetime
import json
import logging
import os
from typing import Optional
from esphome import const
from esphome.core import CORE
from esphome.helpers import write_file_if_changed
from esphome.const import CONF_MDNS, CONF_DISABLED
from esphome.types import CoreType
_LOGGER = logging.getLogger(__name__)

def storage_path() -> str:
    if False:
        i = 10
        return i + 15
    return os.path.join(CORE.data_dir, 'storage', f'{CORE.config_filename}.json')

def ext_storage_path(config_filename: str) -> str:
    if False:
        return 10
    return os.path.join(CORE.data_dir, 'storage', f'{config_filename}.json')

def esphome_storage_path() -> str:
    if False:
        print('Hello World!')
    return os.path.join(CORE.data_dir, 'esphome.json')

def trash_storage_path() -> str:
    if False:
        return 10
    return CORE.relative_config_path('trash')

class StorageJSON:

    def __init__(self, storage_version, name, friendly_name, comment, esphome_version, src_version, address, web_port, target_platform, build_path, firmware_bin_path, loaded_integrations, no_mdns):
        if False:
            return 10
        assert storage_version is None or isinstance(storage_version, int)
        self.storage_version: int = storage_version
        self.name: str = name
        self.friendly_name: str = friendly_name
        self.comment: str = comment
        self.esphome_version: str = esphome_version
        assert src_version is None or isinstance(src_version, int)
        self.src_version: int = src_version
        self.address: str = address
        assert web_port is None or isinstance(web_port, int)
        self.web_port: int = web_port
        self.target_platform: str = target_platform
        self.build_path: str = build_path
        self.firmware_bin_path: str = firmware_bin_path
        self.loaded_integrations: list[str] = loaded_integrations
        self.loaded_integrations.sort()
        self.no_mdns = no_mdns

    def as_dict(self):
        if False:
            return 10
        return {'storage_version': self.storage_version, 'name': self.name, 'friendly_name': self.friendly_name, 'comment': self.comment, 'esphome_version': self.esphome_version, 'src_version': self.src_version, 'address': self.address, 'web_port': self.web_port, 'esp_platform': self.target_platform, 'build_path': self.build_path, 'firmware_bin_path': self.firmware_bin_path, 'loaded_integrations': self.loaded_integrations, 'no_mdns': self.no_mdns}

    def to_json(self):
        if False:
            print('Hello World!')
        return f'{json.dumps(self.as_dict(), indent=2)}\n'

    def save(self, path):
        if False:
            for i in range(10):
                print('nop')
        write_file_if_changed(path, self.to_json())

    @staticmethod
    def from_esphome_core(esph: CoreType, old: Optional['StorageJSON']) -> 'StorageJSON':
        if False:
            i = 10
            return i + 15
        hardware = esph.target_platform.upper()
        if esph.is_esp32:
            from esphome.components import esp32
            hardware = esp32.get_esp32_variant(esph)
        return StorageJSON(storage_version=1, name=esph.name, friendly_name=esph.friendly_name, comment=esph.comment, esphome_version=const.__version__, src_version=1, address=esph.address, web_port=esph.web_port, target_platform=hardware, build_path=esph.build_path, firmware_bin_path=esph.firmware_bin, loaded_integrations=list(esph.loaded_integrations), no_mdns=CONF_MDNS in esph.config and CONF_DISABLED in esph.config[CONF_MDNS] and (esph.config[CONF_MDNS][CONF_DISABLED] is True))

    @staticmethod
    def from_wizard(name: str, friendly_name: str, address: str, platform: str) -> 'StorageJSON':
        if False:
            return 10
        return StorageJSON(storage_version=1, name=name, friendly_name=friendly_name, comment=None, esphome_version=None, src_version=1, address=address, web_port=None, target_platform=platform, build_path=None, firmware_bin_path=None, loaded_integrations=[], no_mdns=False)

    @staticmethod
    def _load_impl(path: str) -> Optional['StorageJSON']:
        if False:
            while True:
                i = 10
        with codecs.open(path, 'r', encoding='utf-8') as f_handle:
            storage = json.load(f_handle)
        storage_version = storage['storage_version']
        name = storage.get('name')
        friendly_name = storage.get('friendly_name')
        comment = storage.get('comment')
        esphome_version = storage.get('esphome_version', storage.get('esphomeyaml_version'))
        src_version = storage.get('src_version')
        address = storage.get('address')
        web_port = storage.get('web_port')
        esp_platform = storage.get('esp_platform')
        build_path = storage.get('build_path')
        firmware_bin_path = storage.get('firmware_bin_path')
        loaded_integrations = storage.get('loaded_integrations', [])
        no_mdns = storage.get('no_mdns', False)
        return StorageJSON(storage_version, name, friendly_name, comment, esphome_version, src_version, address, web_port, esp_platform, build_path, firmware_bin_path, loaded_integrations, no_mdns)

    @staticmethod
    def load(path: str) -> Optional['StorageJSON']:
        if False:
            print('Hello World!')
        try:
            return StorageJSON._load_impl(path)
        except Exception:
            return None

    def __eq__(self, o) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(o, StorageJSON) and self.as_dict() == o.as_dict()

class EsphomeStorageJSON:

    def __init__(self, storage_version, cookie_secret, last_update_check, remote_version):
        if False:
            print('Hello World!')
        assert storage_version is None or isinstance(storage_version, int)
        self.storage_version: int = storage_version
        self.cookie_secret: str = cookie_secret
        self.last_update_check_str: str = last_update_check
        self.remote_version: Optional[str] = remote_version

    def as_dict(self) -> dict:
        if False:
            print('Hello World!')
        return {'storage_version': self.storage_version, 'cookie_secret': self.cookie_secret, 'last_update_check': self.last_update_check_str, 'remote_version': self.remote_version}

    @property
    def last_update_check(self) -> Optional[datetime]:
        if False:
            for i in range(10):
                print('nop')
        try:
            return datetime.strptime(self.last_update_check_str, '%Y-%m-%dT%H:%M:%S')
        except Exception:
            return None

    @last_update_check.setter
    def last_update_check(self, new: datetime) -> None:
        if False:
            print('Hello World!')
        self.last_update_check_str = new.strftime('%Y-%m-%dT%H:%M:%S')

    def to_json(self) -> dict:
        if False:
            i = 10
            return i + 15
        return f'{json.dumps(self.as_dict(), indent=2)}\n'

    def save(self, path: str) -> None:
        if False:
            return 10
        write_file_if_changed(path, self.to_json())

    @staticmethod
    def _load_impl(path: str) -> Optional['EsphomeStorageJSON']:
        if False:
            return 10
        with codecs.open(path, 'r', encoding='utf-8') as f_handle:
            storage = json.load(f_handle)
        storage_version = storage['storage_version']
        cookie_secret = storage.get('cookie_secret')
        last_update_check = storage.get('last_update_check')
        remote_version = storage.get('remote_version')
        return EsphomeStorageJSON(storage_version, cookie_secret, last_update_check, remote_version)

    @staticmethod
    def load(path: str) -> Optional['EsphomeStorageJSON']:
        if False:
            return 10
        try:
            return EsphomeStorageJSON._load_impl(path)
        except Exception:
            return None

    @staticmethod
    def get_default() -> 'EsphomeStorageJSON':
        if False:
            while True:
                i = 10
        return EsphomeStorageJSON(storage_version=1, cookie_secret=binascii.hexlify(os.urandom(64)).decode(), last_update_check=None, remote_version=None)

    def __eq__(self, o) -> bool:
        if False:
            print('Hello World!')
        return isinstance(o, EsphomeStorageJSON) and self.as_dict() == o.as_dict()
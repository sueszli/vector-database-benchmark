import tarfile
import shutil
from enum import Enum
from pathlib import Path
from tempfile import mkdtemp
from urllib.request import urlretrieve
import inspect
import sys
from os.path import realpath
from cryptography.x509 import ObjectIdentifier
from sslyze.plugins.certificate_info.trust_stores.trust_store import TrustStore
from typing import List

class TrustStoreEnum(Enum):
    APPLE = 1
    GOOGLE_AOSP = 2
    MICROSOFT_WINDOWS = 3
    MOZILLA_NSS = 4
    ORACLE_JAVA = 5

def _get_script_dir(follow_symlinks: bool=True) -> Path:
    if False:
        i = 10
        return i + 15
    if getattr(sys, 'frozen', False):
        path = Path(sys.executable).absolute()
    else:
        path = Path(inspect.getabsfile(_get_script_dir))
    if follow_symlinks:
        path = Path(realpath(path))
    return path.parent

class TrustStoresRepository:
    """The list of default trust stores used by SSLyze for certificate validation.

    By default, SSLyze packages the following trust stores: Mozilla, Microsoft, Apple, Android and Java.
    """
    _DEFAULT_TRUST_STORES_PATH = _get_script_dir() / 'pem_files'
    _DEFAULT_REPOSITORY = None
    _STORE_PRETTY_NAMES = {TrustStoreEnum.APPLE: 'Apple', TrustStoreEnum.GOOGLE_AOSP: 'Android', TrustStoreEnum.MICROSOFT_WINDOWS: 'Windows', TrustStoreEnum.MOZILLA_NSS: 'Mozilla', TrustStoreEnum.ORACLE_JAVA: 'Java'}

    def __init__(self, repository_path: Path) -> None:
        if False:
            i = 10
            return i + 15
        available_stores = {}
        for store_enum in TrustStoreEnum:
            store_yaml_path = repository_path / f'{store_enum.name.lower()}.yaml'
            store_yaml = store_yaml_path.read_text()
            if store_enum in [TrustStoreEnum.MICROSOFT_WINDOWS, TrustStoreEnum.MOZILLA_NSS]:
                store_version = store_yaml.split('date_fetched: ', 1)[1].split('\n', 1)[0].strip()
            else:
                store_version = store_yaml.split('version: ', 1)[1].split('\n', 1)[0].strip(" '")
            store_pem_path = repository_path / f'{store_enum.name.lower()}.pem'
            if not store_pem_path.exists():
                raise ValueError(f'Could not find trust store at {store_pem_path}')
            available_stores[store_enum] = TrustStore(path=store_pem_path, name=self._STORE_PRETTY_NAMES[store_enum], version=store_version, ev_oids=[ObjectIdentifier(oid) for oid in _MOZILLA_EV_OIDS] if store_enum == TrustStoreEnum.MOZILLA_NSS else None)
        self._available_stores = available_stores

    def get_all_stores(self) -> List[TrustStore]:
        if False:
            for i in range(10):
                print('nop')
        return list(self._available_stores.values())

    def get_main_store(self) -> TrustStore:
        if False:
            print('Hello World!')
        return self._available_stores[TrustStoreEnum.MOZILLA_NSS]

    @classmethod
    def get_default(cls) -> 'TrustStoresRepository':
        if False:
            for i in range(10):
                print('nop')
        if cls._DEFAULT_REPOSITORY is None:
            cls._DEFAULT_REPOSITORY = cls(cls._DEFAULT_TRUST_STORES_PATH)
        return cls._DEFAULT_REPOSITORY
    _UPDATE_URL = 'https://nabla-c0d3.github.io/trust_stores_observatory/trust_stores_as_pem.tar.gz'

    @classmethod
    def update_default(cls) -> 'TrustStoresRepository':
        if False:
            i = 10
            return i + 15
        'Update the default trust stores used by SSLyze.\n\n        The latest stores will be downloaded from https://github.com/nabla-c0d3/trust_stores_observatory.\n        '
        temp_path = Path(mkdtemp())
        try:
            archive_path = temp_path / 'trust_stores_as_pem.tar.gz'
            urlretrieve(cls._UPDATE_URL, archive_path)
            extract_path = temp_path / 'extracted'
            tarfile.open(archive_path).extractall(extract_path)
            shutil.rmtree(cls._DEFAULT_TRUST_STORES_PATH)
            shutil.copytree(extract_path, cls._DEFAULT_TRUST_STORES_PATH)
        finally:
            shutil.rmtree(temp_path)
        cls._DEFAULT_REPOSITORY = cls(cls._DEFAULT_TRUST_STORES_PATH)
        return cls._DEFAULT_REPOSITORY
_MOZILLA_EV_OIDS = ['1.2.276.0.44.1.1.1.4', '1.2.392.200091.100.721.1', '1.2.40.0.17.1.22', '1.2.616.1.113527.2.5.1.1', '1.3.159.1.17.1', '1.3.6.1.4.1.13177.10.1.3.10', '1.3.6.1.4.1.14370.1.6', '1.3.6.1.4.1.14777.6.1.1', '1.3.6.1.4.1.14777.6.1.2', '1.3.6.1.4.1.17326.10.14.2.1.2', '1.3.6.1.4.1.17326.10.14.2.2.2', '1.3.6.1.4.1.17326.10.8.12.1.2', '1.3.6.1.4.1.17326.10.8.12.2.2', '1.3.6.1.4.1.22234.2.5.2.3.1', '1.3.6.1.4.1.23223.1.1.1', '1.3.6.1.4.1.29836.1.10', '1.3.6.1.4.1.34697.2.1', '1.3.6.1.4.1.34697.2.2', '1.3.6.1.4.1.34697.2.3', '1.3.6.1.4.1.34697.2.4', '1.3.6.1.4.1.36305.2', '1.3.6.1.4.1.40869.1.1.22.3', '1.3.6.1.4.1.4146.1.1', '1.3.6.1.4.1.4788.2.202.1', '1.3.6.1.4.1.6334.1.100.1', '1.3.6.1.4.1.6449.1.2.1.5.1', '1.3.6.1.4.1.782.1.2.1.8.1', '1.3.6.1.4.1.7879.13.24.1', '1.3.6.1.4.1.8024.0.2.100.1.2', '2.16.156.112554.3', '2.16.528.1.1003.1.2.7', '2.16.578.1.26.1.3.3', '2.16.756.1.83.21.0', '2.16.756.1.89.1.2.1.1', '2.16.792.3.0.3.1.1.5', '2.16.792.3.0.4.1.1.4', '2.16.840.1.113733.1.7.23.6', '2.16.840.1.113733.1.7.48.1', '2.16.840.1.114028.10.1.2', '2.16.840.1.114171.500.9', '2.16.840.1.114404.1.1.2.4.1', '2.16.840.1.114412.2.1', '2.16.840.1.114413.1.7.23.3', '2.16.840.1.114414.1.7.23.3', '2.16.840.1.114414.1.7.24.3']
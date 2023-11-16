import os
import sys
import tempfile
from importlib import reload
from unittest.mock import patch
import pytest
from feast import Entity, RepoConfig
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig

@pytest.fixture(scope='function')
def dummy_exporter():
    if False:
        i = 10
        return i + 15
    event_log = []
    with patch('feast.usage._export', new=event_log.append):
        yield event_log

@pytest.fixture(scope='function')
def enabling_toggle():
    if False:
        while True:
            i = 10
    with patch('feast.usage._is_enabled') as p:
        p.__bool__.return_value = True
        yield p
    _reload_feast()

@pytest.mark.integration
def test_usage_on(dummy_exporter, enabling_toggle):
    if False:
        for i in range(10):
            print('nop')
    _reload_feast()
    from feast.feature_store import FeatureStore
    with tempfile.TemporaryDirectory() as temp_dir:
        test_feature_store = FeatureStore(config=RepoConfig(registry=os.path.join(temp_dir, 'registry.db'), project='fake_project', provider='local', online_store=SqliteOnlineStoreConfig(path=os.path.join(temp_dir, 'online.db')), entity_key_serialization_version=2))
        entity = Entity(name='driver_car_id', description='Car driver id', tags={'team': 'matchmaking'})
        test_feature_store.apply([entity])
        assert len(dummy_exporter) == 3
        assert {'entrypoint': 'feast.infra.registry.file.FileRegistryStore.get_registry_proto'}.items() <= dummy_exporter[0].items()
        assert {'entrypoint': 'feast.infra.registry.file.FileRegistryStore.update_registry_proto'}.items() <= dummy_exporter[1].items()
        assert {'entrypoint': 'feast.feature_store.FeatureStore.apply'}.items() <= dummy_exporter[2].items()

@pytest.mark.integration
def test_usage_off(dummy_exporter, enabling_toggle):
    if False:
        return 10
    enabling_toggle.__bool__.return_value = False
    _reload_feast()
    from feast.feature_store import FeatureStore
    with tempfile.TemporaryDirectory() as temp_dir:
        test_feature_store = FeatureStore(config=RepoConfig(registry=os.path.join(temp_dir, 'registry.db'), project='fake_project', provider='local', online_store=SqliteOnlineStoreConfig(path=os.path.join(temp_dir, 'online.db')), entity_key_serialization_version=2))
        entity = Entity(name='driver_car_id', description='Car driver id', tags={'team': 'matchmaking'})
        test_feature_store.apply([entity])
        assert not dummy_exporter

@pytest.mark.integration
def test_exception_usage_on(dummy_exporter, enabling_toggle):
    if False:
        for i in range(10):
            print('nop')
    _reload_feast()
    from feast.feature_store import FeatureStore
    with pytest.raises(OSError):
        FeatureStore('/tmp/non_existent_directory')
    assert len(dummy_exporter) == 1
    assert {'entrypoint': 'feast.feature_store.FeatureStore.__init__', 'exception': repr(FileNotFoundError(2, 'No such file or directory'))}.items() <= dummy_exporter[0].items()

@pytest.mark.integration
def test_exception_usage_off(dummy_exporter, enabling_toggle):
    if False:
        print('Hello World!')
    enabling_toggle.__bool__.return_value = False
    _reload_feast()
    from feast.feature_store import FeatureStore
    with pytest.raises(OSError):
        FeatureStore('/tmp/non_existent_directory')
    assert not dummy_exporter

def _reload_feast():
    if False:
        while True:
            i = 10
    'After changing environment need to reload modules and rerun usage decorators'
    modules = ('feast.infra.registry.file', 'feast.infra.online_stores.sqlite', 'feast.feature_store')
    for mod in modules:
        if mod in sys.modules:
            reload(sys.modules[mod])
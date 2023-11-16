import json
from typing import Any, Mapping
from airbyte_cdk.models import OrchestratorType, Type
from airbyte_cdk.sources import Source
from source_google_search_console.config_migrations import MigrateCustomReports
from source_google_search_console.source import SourceGoogleSearchConsole
CMD = 'check'
TEST_CONFIG_PATH = 'unit_tests/test_migrations/test_config.json'
NEW_TEST_CONFIG_PATH = 'unit_tests/test_migrations/test_new_config.json'
SOURCE_INPUT_ARGS = [CMD, '--config', TEST_CONFIG_PATH]
SOURCE: Source = SourceGoogleSearchConsole()

def load_config(config_path: str=TEST_CONFIG_PATH) -> Mapping[str, Any]:
    if False:
        while True:
            i = 10
    with open(config_path, 'r') as config:
        return json.load(config)

def revert_migration(config_path: str=TEST_CONFIG_PATH) -> None:
    if False:
        i = 10
        return i + 15
    with open(config_path, 'r') as test_config:
        config = json.load(test_config)
        config.pop('custom_reports_array')
        with open(config_path, 'w') as updated_config:
            config = json.dumps(config)
            updated_config.write(config)

def test_migrate_config():
    if False:
        print('Hello World!')
    migration_instance = MigrateCustomReports()
    original_config = load_config()
    migration_instance.migrate(SOURCE_INPUT_ARGS, SOURCE)
    test_migrated_config = load_config()
    assert 'custom_reports_array' in test_migrated_config
    assert isinstance(test_migrated_config['custom_reports_array'], list)
    assert 'custom_reports' in test_migrated_config
    assert isinstance(test_migrated_config['custom_reports'], str)
    assert not migration_instance.should_migrate(test_migrated_config)
    assert json.loads(original_config['custom_reports']) == test_migrated_config['custom_reports_array']
    control_msg = migration_instance.message_repository._message_queue[0]
    assert control_msg.type == Type.CONTROL
    assert control_msg.control.type == OrchestratorType.CONNECTOR_CONFIG
    assert isinstance(control_msg.control.connectorConfig.config['custom_reports'], str)
    assert isinstance(control_msg.control.connectorConfig.config['custom_reports_array'], list)
    assert control_msg.control.connectorConfig.config['custom_reports_array'][0]['name'] == 'custom_dimensions'
    assert control_msg.control.connectorConfig.config['custom_reports_array'][0]['dimensions'] == ['date', 'country', 'device']
    revert_migration()

def test_config_is_reverted():
    if False:
        for i in range(10):
            print('nop')
    test_config = load_config()
    assert 'custom_reports_array' not in test_config
    assert 'custom_reports' in test_config
    assert isinstance(test_config['custom_reports'], str)

def test_should_not_migrate_new_config():
    if False:
        for i in range(10):
            print('nop')
    new_config = load_config(NEW_TEST_CONFIG_PATH)
    migration_instance = MigrateCustomReports()
    assert not migration_instance.should_migrate(new_config)
import json
import os
import pathlib
import shutil
import pytest
from integration_tests.dbt_integration_test import DbtIntegrationTest
from integration_tests.utils import generate_dbt_models, run_destination_process, setup_test_dir
from normalization import DestinationType
temporary_folders = set()
dbt_test_utils = DbtIntegrationTest()

@pytest.fixture(scope='module', autouse=True)
def before_all_tests(request):
    if False:
        for i in range(10):
            print('nop')
    destinations_to_test = dbt_test_utils.get_test_targets()
    clean_up_args = {'destination_type': [d for d in DestinationType if d.value in destinations_to_test], 'test_type': 'test_reset_scd_overwrite', 'tmp_folders': temporary_folders}
    dbt_test_utils.set_target_schema('test_reset_scd_overwrite')
    dbt_test_utils.change_current_test_dir(request)
    dbt_test_utils.setup_db(destinations_to_test)
    os.environ['PATH'] = os.path.abspath('../.venv/bin/') + ':' + os.environ['PATH']
    yield
    dbt_test_utils.clean_tmp_tables(**clean_up_args)
    dbt_test_utils.tear_down_db()
    for folder in temporary_folders:
        print(f'Deleting temporary test folder {folder}')
        shutil.rmtree(folder, ignore_errors=True)

@pytest.fixture
def setup_test_path(request):
    if False:
        print('Hello World!')
    dbt_test_utils.change_current_test_dir(request)
    print(f'Running from: {pathlib.Path().absolute()}')
    print(f"Current PATH is: {os.environ['PATH']}")
    yield
    os.chdir(request.config.invocation_dir)

@pytest.mark.parametrize('destination_type', DestinationType.testable_destinations())
def test_reset_scd_on_overwrite(destination_type: DestinationType, setup_test_path):
    if False:
        for i in range(10):
            print('nop')
    if destination_type.value not in dbt_test_utils.get_test_targets():
        pytest.skip(f'Destinations {destination_type} is not in NORMALIZATION_TEST_TARGET env variable')
    if destination_type.value in [DestinationType.ORACLE.value, DestinationType.TIDB.value]:
        pytest.skip(f'{destination_type} does not support incremental sync with schema change yet')
    elif destination_type.value == DestinationType.REDSHIFT.value:
        dbt_test_utils.set_target_schema(dbt_test_utils.generate_random_string('test_reset_scd_'))
    test_resource_name = 'test_reset_scd_overwrite'
    target_schema = dbt_test_utils.target_schema
    try:
        print(f'Testing resetting SCD tables on overwrite with {destination_type} in schema {target_schema}')
        run_reset_scd_on_overwrite_test(destination_type, test_resource_name)
    finally:
        dbt_test_utils.set_target_schema(target_schema)

def run_reset_scd_on_overwrite_test(destination_type: DestinationType, test_resource_name: str):
    if False:
        for i in range(10):
            print('nop')
    integration_type = destination_type.value
    test_root_dir = setup_test_dir(integration_type, temporary_folders)
    destination_config = dbt_test_utils.generate_profile_yaml_file(destination_type, test_root_dir)
    test_directory = os.path.join(test_root_dir, 'models/generated')
    shutil.rmtree(test_directory, ignore_errors=True)
    config_file = os.path.join(test_root_dir, 'destination_config.json')
    with open(config_file, 'w') as f:
        f.write(json.dumps(destination_config))
    dbt_test_utils.dbt_check(destination_type, test_root_dir)
    original_catalog_file = os.path.join('resources', test_resource_name, 'data_input', 'test_drop_scd_catalog.json')
    dbt_test_utils.copy_replace(original_catalog_file, os.path.join(test_root_dir, 'initial_reset_catalog.json'), pattern='"destination_sync_mode": ".*"', replace_value='"destination_sync_mode": "overwrite"')
    assert run_destination_process(destination_type, test_root_dir, '', 'initial_reset_catalog.json', dbt_test_utils)
    generate_dbt_models(destination_type, test_resource_name, test_root_dir, 'models', 'test_drop_scd_catalog_reset.json', dbt_test_utils)
    dbt_test_utils.dbt_run(destination_type, test_root_dir, force_full_refresh=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_incremental')
    shutil.rmtree(test_directory, ignore_errors=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_views')
    shutil.rmtree(test_directory, ignore_errors=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_ctes')
    shutil.rmtree(test_directory, ignore_errors=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_tables')
    shutil.rmtree(test_directory, ignore_errors=True)
    dbt_test_utils.copy_replace(original_catalog_file, os.path.join(test_root_dir, 'destination_catalog.json'))
    message_file = os.path.join('resources', test_resource_name, 'data_input', 'test_drop_scd_messages.txt')
    assert run_destination_process(destination_type, test_root_dir, message_file, 'destination_catalog.json', dbt_test_utils)
    generate_dbt_models(destination_type, test_resource_name, test_root_dir, 'models', 'test_drop_scd_catalog.json', dbt_test_utils)
    dbt_test_utils.dbt_run(destination_type, test_root_dir, force_full_refresh=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_incremental')
    shutil.rmtree(test_directory, ignore_errors=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_views')
    shutil.rmtree(test_directory, ignore_errors=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_ctes')
    shutil.rmtree(test_directory, ignore_errors=True)
    reset_catalog_file = os.path.join('resources', test_resource_name, 'data_input', 'test_drop_scd_catalog_reset.json')
    dbt_test_utils.copy_replace(reset_catalog_file, os.path.join(test_root_dir, 'reset_catalog.json'))
    assert run_destination_process(destination_type, test_root_dir, '', 'reset_catalog.json', dbt_test_utils)
    generate_dbt_models(destination_type, test_resource_name, test_root_dir, 'models', 'test_drop_scd_catalog_reset.json', dbt_test_utils)
    dbt_test_utils.dbt_run(destination_type, test_root_dir, force_full_refresh=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_incremental')
    shutil.rmtree(test_directory, ignore_errors=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_views')
    shutil.rmtree(test_directory, ignore_errors=True)
    test_directory = os.path.join(test_root_dir, 'models/generated/airbyte_ctes')
    shutil.rmtree(test_directory, ignore_errors=True)
    modified_catalog_file = os.path.join('resources', test_resource_name, 'data_input', 'test_drop_scd_catalog_incremental.json')
    dbt_test_utils.copy_replace(modified_catalog_file, os.path.join(test_root_dir, 'destination_catalog.json'))
    message_file = os.path.join('resources', test_resource_name, 'data_input', 'test_scd_reset_messages_incremental.txt')
    assert run_destination_process(destination_type, test_root_dir, message_file, 'destination_catalog.json', dbt_test_utils)
    generate_dbt_models(destination_type, test_resource_name, test_root_dir, 'models', 'test_drop_scd_catalog_reset.json', dbt_test_utils)
    dbt_test_utils.dbt_run(destination_type, test_root_dir)
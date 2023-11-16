from __future__ import annotations
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.microsoft.azure.transfers.sftp_to_wasb import SftpFile, SFTPToWasbOperator
TASK_ID = 'test-gcs-to-sftp-operator'
WASB_CONN_ID = 'wasb_default'
SFTP_CONN_ID = 'ssh_default'
CONTAINER_NAME = 'test-container'
WILDCARD_PATH = 'main_dir/*'
WILDCARD_FILE_NAME = 'main_dir/test_object*.json'
SOURCE_PATH_NO_WILDCARD = 'main_dir/'
SOURCE_OBJECT_MULTIPLE_WILDCARDS = 'main_dir/csv/*/test_*.csv'
BLOB_PREFIX = 'sponge-bob'
EXPECTED_BLOB_NAME = 'test_object3.json'
EXPECTED_FILES = [SOURCE_PATH_NO_WILDCARD + EXPECTED_BLOB_NAME]

class TestSFTPToWasbOperator:

    def test_init(self):
        if False:
            print('Hello World!')
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=SOURCE_PATH_NO_WILDCARD, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, blob_prefix=BLOB_PREFIX, wasb_conn_id=WASB_CONN_ID, move_object=False)
        assert operator.sftp_source_path == SOURCE_PATH_NO_WILDCARD
        assert operator.sftp_conn_id == SFTP_CONN_ID
        assert operator.container_name == CONTAINER_NAME
        assert operator.wasb_conn_id == WASB_CONN_ID
        assert operator.blob_prefix == BLOB_PREFIX
        assert operator.create_container is False

    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.WasbHook', autospec=True)
    def test_execute_more_than_one_wildcard_exception(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=SOURCE_OBJECT_MULTIPLE_WILDCARDS, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, blob_prefix=BLOB_PREFIX, wasb_conn_id=WASB_CONN_ID, move_object=False)
        with pytest.raises(AirflowException) as err:
            operator.check_wildcards_limit()
        assert "Only one wildcard '*' is allowed" in str(err.value)

    def test_get_sftp_tree_behavior(self):
        if False:
            print('Hello World!')
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=WILDCARD_PATH, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, wasb_conn_id=WASB_CONN_ID, move_object=False)
        (sftp_complete_path, prefix, delimiter) = operator.get_tree_behavior()
        assert sftp_complete_path == 'main_dir', 'not matched at expected complete path'
        assert prefix == 'main_dir/', 'Prefix must be EQUAL TO wildcard'
        assert delimiter == '', 'Delimiter must be empty'

    def test_get_sftp_tree_behavior_without_wildcard(self):
        if False:
            print('Hello World!')
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=SOURCE_PATH_NO_WILDCARD, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, wasb_conn_id=WASB_CONN_ID, move_object=False)
        (sftp_complete_path, prefix, delimiter) = operator.get_tree_behavior()
        assert sftp_complete_path == 'main_dir/', 'not matched at expected complete path'
        assert prefix is None, 'Prefix must be NONE when no wildcard'
        assert delimiter is None, 'Delimiter must be none'

    def test_source_path_contains_wildcard(self):
        if False:
            return 10
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=WILDCARD_PATH, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, wasb_conn_id=WASB_CONN_ID, move_object=False)
        output = operator.source_path_contains_wildcard
        assert output is True, 'This path contains a wildpath'

    def test_source_path_not_contains_wildcard(self):
        if False:
            return 10
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=SOURCE_PATH_NO_WILDCARD, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, wasb_conn_id=WASB_CONN_ID, move_object=False)
        output = operator.source_path_contains_wildcard
        assert output is False, 'This path does not contains a wildpath'

    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.WasbHook')
    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.SFTPHook')
    def test_get_sftp_files_map_no_wildcard(self, sftp_hook, mock_hook):
        if False:
            while True:
                i = 10
        sftp_hook.return_value.get_tree_map.return_value = [EXPECTED_FILES, [], []]
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=SOURCE_PATH_NO_WILDCARD, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, wasb_conn_id=WASB_CONN_ID, move_object=True)
        files = operator.get_sftp_files_map()
        assert len(files) == 1, 'no matched at expected found files'
        assert files[0].blob_name == EXPECTED_BLOB_NAME, 'expected blob name not matched'

    @pytest.mark.parametrize(argnames='create_container', argvalues=[True, False])
    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.WasbHook')
    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.SFTPHook')
    def test_copy_files_to_wasb(self, sftp_hook, mock_hook, create_container):
        if False:
            for i in range(10):
                print('nop')
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=SOURCE_PATH_NO_WILDCARD, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, wasb_conn_id=WASB_CONN_ID, move_object=True, create_container=create_container)
        sftp_files = [SftpFile(EXPECTED_FILES[0], EXPECTED_BLOB_NAME)]
        files = operator.copy_files_to_wasb(sftp_files)
        operator.sftp_hook.retrieve_file.assert_has_calls([mock.call('main_dir/test_object3.json', mock.ANY)])
        mock_hook.return_value.load_file.assert_called_once_with(mock.ANY, CONTAINER_NAME, EXPECTED_BLOB_NAME, create_container, overwrite=False)
        assert len(files) == 1, 'no matched at expected uploaded files'

    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.SFTPHook')
    def test_delete_files(self, sftp_hook):
        if False:
            return 10
        sftp_mock = sftp_hook.return_value
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=SOURCE_PATH_NO_WILDCARD, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, wasb_conn_id=WASB_CONN_ID, move_object=True)
        sftp_file_paths = EXPECTED_FILES
        operator.delete_files(sftp_file_paths)
        sftp_mock.delete_file.assert_has_calls([mock.call(EXPECTED_FILES[0])])

    @pytest.mark.parametrize(argnames='create_container', argvalues=[True, False])
    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.WasbHook')
    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.SFTPHook')
    def test_execute(self, sftp_hook, mock_hook, create_container):
        if False:
            while True:
                i = 10
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=WILDCARD_FILE_NAME, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, wasb_conn_id=WASB_CONN_ID, move_object=False, create_container=create_container)
        sftp_hook.return_value.get_tree_map.return_value = [['main_dir/test_object.json'], [], []]
        operator.execute(None)
        sftp_hook.return_value.get_tree_map.assert_called_with('main_dir', prefix='main_dir/test_object', delimiter='.json')
        sftp_hook.return_value.retrieve_file.assert_has_calls([mock.call('main_dir/test_object.json', mock.ANY)])
        mock_hook.return_value.load_file.assert_called_once_with(mock.ANY, CONTAINER_NAME, 'test_object.json', create_container, overwrite=False)
        sftp_hook.return_value.delete_file.assert_not_called()

    @pytest.mark.parametrize(argnames='create_container', argvalues=[True, False])
    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.WasbHook')
    @mock.patch('airflow.providers.microsoft.azure.transfers.sftp_to_wasb.SFTPHook')
    def test_execute_moved_files(self, sftp_hook, mock_hook, create_container):
        if False:
            return 10
        operator = SFTPToWasbOperator(task_id=TASK_ID, sftp_source_path=WILDCARD_FILE_NAME, sftp_conn_id=SFTP_CONN_ID, container_name=CONTAINER_NAME, wasb_conn_id=WASB_CONN_ID, move_object=True, blob_prefix=BLOB_PREFIX, create_container=create_container)
        sftp_hook.return_value.get_tree_map.return_value = [['main_dir/test_object.json'], [], []]
        operator.execute(None)
        sftp_hook.return_value.get_tree_map.assert_called_with('main_dir', prefix='main_dir/test_object', delimiter='.json')
        sftp_hook.return_value.retrieve_file.assert_has_calls([mock.call('main_dir/test_object.json', mock.ANY)])
        mock_hook.return_value.load_file.assert_called_once_with(mock.ANY, CONTAINER_NAME, BLOB_PREFIX + 'test_object.json', create_container, overwrite=False)
        assert sftp_hook.return_value.delete_file.called is True, 'File must be moved'
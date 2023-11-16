import os
from datetime import datetime, timedelta
from unittest.mock import patch
from mage_ai.data_preparation.logging.logger_manager import LoggerManager
from mage_ai.data_preparation.storage.local_storage import LocalStorage
from mage_ai.tests.base_test import TestCase

class LoggerManagerTest(TestCase):

    def test_delete_old_logs_no_retention_period(self):
        if False:
            print('Hello World!')
        mock_pipeline_uuid = 'pipeline_uuid_1'
        logger_manager = LoggerManager(pipeline_uuid=mock_pipeline_uuid)
        logger_manager.logging_config.retention_period = None
        logger_manager.storage = MockStorage()
        self.__create_log_dir(logger_manager, mock_pipeline_uuid)
        logger_manager.delete_old_logs()
        self.assertEqual(logger_manager.storage.remove_dir_calls, [])

    def test_delete_old_logs_with_retention_period(self):
        if False:
            print('Hello World!')
        mock_pipeline_uuid = 'pipeline_uuid_2'
        logger_manager = LoggerManager(pipeline_uuid=mock_pipeline_uuid)
        retention_period = '7d'
        logger_manager.logging_config.retention_period = retention_period
        logger_manager.storage = MockStorage()
        mock_old_log_folder = self.__create_log_dir(logger_manager, mock_pipeline_uuid)
        logger_manager.delete_old_logs()
        self.assertIn(mock_old_log_folder, logger_manager.storage.remove_dir_calls)

    @patch('mage_ai.data_preparation.models.pipeline.Pipeline.get_all_pipelines')
    def test_delete_old_logs_with_no_pipeline_uuid(self, mock_get_all_pipelines):
        if False:
            while True:
                i = 10
        logger_manager = LoggerManager()
        retention_period = '7d'
        logger_manager.logging_config.retention_period = retention_period
        logger_manager.pipeline_uuid = None
        logger_manager.storage = MockStorage()
        mock_pipeline_uuids = ['pipeline_uuid_3', 'pipeline_uuid_4']
        mock_get_all_pipelines.return_value = mock_pipeline_uuids
        mock_pipeline_configs = [dict(pipeline_uuid='pipeline_uuid_3', days_ago=10, trigger_id=1), dict(pipeline_uuid='pipeline_uuid_4', days_ago=5, trigger_id=2), dict(pipeline_uuid='pipeline_uuid_4', days_ago=15, trigger_id=3)]
        for mock_pipeline_config in mock_pipeline_configs:
            mock_log_folder = self.__create_log_dir(logger_manager, mock_pipeline_config['pipeline_uuid'], days_ago=mock_pipeline_config['days_ago'], trigger_id=mock_pipeline_config['trigger_id'])
            mock_pipeline_config['log_folder'] = mock_log_folder
        logger_manager.delete_old_logs()
        for mock_pipeline_config in mock_pipeline_configs:
            if mock_pipeline_config['days_ago'] >= 7:
                self.assertIn(mock_pipeline_config['log_folder'], logger_manager.storage.remove_dir_calls)
            else:
                self.assertNotIn(mock_pipeline_config['log_folder'], logger_manager.storage.remove_dir_calls)

    def __create_log_dir(self, logger_manager: LoggerManager, pipeline_uuid: str, days_ago: int=10, trigger_id: int=1) -> str:
        if False:
            while True:
                i = 10
        mock_log_path_prefix = logger_manager.get_log_filepath_prefix(pipeline_uuid=pipeline_uuid)
        mock_old_log_date = (datetime.utcnow() - timedelta(days=days_ago)).strftime(format='%Y%m%dT%H%M%S')
        mock_log_folder = os.path.join(mock_log_path_prefix, str(trigger_id), mock_old_log_date)
        logger_manager.storage.makedirs(mock_log_folder)
        return mock_log_folder

class MockStorage(LocalStorage):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.remove_dir_calls = []

    def remove_dir(self, path):
        if False:
            for i in range(10):
                print('nop')
        super().remove_dir(path)
        self.remove_dir_calls.append(path)
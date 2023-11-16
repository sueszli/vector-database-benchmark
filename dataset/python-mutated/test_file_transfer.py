from __future__ import annotations
from unittest import mock
from airflow.providers.common.io.operators.file_transfer import FileTransferOperator

def test_file_transfer_copy():
    if False:
        while True:
            i = 10
    with mock.patch('airflow.providers.common.io.operators.file_transfer.ObjectStoragePath') as mock_object_storage_path:
        source_path = mock.MagicMock()
        target_path = mock.MagicMock()
        mock_object_storage_path.side_effect = [source_path, target_path]
        source_path.exists.return_value = True
        target_path.exists.return_value = False
        operator = FileTransferOperator(task_id='test_common_io_file_transfer_task', src='test_source', dst='test_target')
        operator.execute(context={})
        mock_object_storage_path.assert_has_calls([mock.call('test_source', conn_id=None), mock.call('test_target', conn_id=None)])
        source_path.copy.assert_called_once_with(target_path)
        target_path.copy.assert_not_called()
from __future__ import annotations
from unittest.mock import patch
import pytest
from sqlalchemy import or_
from airflow import models
from airflow.providers.mysql.transfers.s3_to_mysql import S3ToMySqlOperator
from airflow.utils import db
from airflow.utils.session import create_session
pytestmark = pytest.mark.db_test

class TestS3ToMySqlTransfer:

    def setup_method(self):
        if False:
            while True:
                i = 10
        db.merge_conn(models.Connection(conn_id='s3_test', conn_type='s3', schema='test', extra='{"aws_access_key_id": "aws_access_key_id", "aws_secret_access_key": "aws_secret_access_key"}'))
        db.merge_conn(models.Connection(conn_id='mysql_test', conn_type='mysql', host='some.host.com', schema='test_db', login='user', password='password'))
        self.s3_to_mysql_transfer_kwargs = {'aws_conn_id': 's3_test', 'mysql_conn_id': 'mysql_test', 's3_source_key': 'test/s3_to_mysql_test.csv', 'mysql_table': 'mysql_table', 'mysql_duplicate_key_handling': 'IGNORE', 'mysql_extra_options': "\n                FIELDS TERMINATED BY ','\n                IGNORE 1 LINES\n            ", 'mysql_local_infile': False, 'task_id': 'task_id', 'dag': None}

    @patch('airflow.providers.mysql.transfers.s3_to_mysql.S3Hook.download_file')
    @patch('airflow.providers.mysql.transfers.s3_to_mysql.MySqlHook.bulk_load_custom')
    @patch('airflow.providers.mysql.transfers.s3_to_mysql.os.remove')
    def test_execute(self, mock_remove, mock_bulk_load_custom, mock_download_file):
        if False:
            print('Hello World!')
        S3ToMySqlOperator(**self.s3_to_mysql_transfer_kwargs).execute({})
        mock_download_file.assert_called_once_with(key=self.s3_to_mysql_transfer_kwargs['s3_source_key'])
        mock_bulk_load_custom.assert_called_once_with(table=self.s3_to_mysql_transfer_kwargs['mysql_table'], tmp_file=mock_download_file.return_value, duplicate_key_handling=self.s3_to_mysql_transfer_kwargs['mysql_duplicate_key_handling'], extra_options=self.s3_to_mysql_transfer_kwargs['mysql_extra_options'])
        mock_remove.assert_called_once_with(mock_download_file.return_value)

    @patch('airflow.providers.mysql.transfers.s3_to_mysql.S3Hook.download_file')
    @patch('airflow.providers.mysql.transfers.s3_to_mysql.MySqlHook.bulk_load_custom')
    @patch('airflow.providers.mysql.transfers.s3_to_mysql.os.remove')
    def test_execute_exception(self, mock_remove, mock_bulk_load_custom, mock_download_file):
        if False:
            return 10
        mock_bulk_load_custom.side_effect = Exception
        with pytest.raises(Exception):
            S3ToMySqlOperator(**self.s3_to_mysql_transfer_kwargs).execute({})
        mock_download_file.assert_called_once_with(key=self.s3_to_mysql_transfer_kwargs['s3_source_key'])
        mock_bulk_load_custom.assert_called_once_with(table=self.s3_to_mysql_transfer_kwargs['mysql_table'], tmp_file=mock_download_file.return_value, duplicate_key_handling=self.s3_to_mysql_transfer_kwargs['mysql_duplicate_key_handling'], extra_options=self.s3_to_mysql_transfer_kwargs['mysql_extra_options'])
        mock_remove.assert_called_once_with(mock_download_file.return_value)

    def teardown_method(self):
        if False:
            return 10
        with create_session() as session:
            session.query(models.Connection).filter(or_(models.Connection.conn_id == 's3_test', models.Connection.conn_id == 'mysql_test')).delete()
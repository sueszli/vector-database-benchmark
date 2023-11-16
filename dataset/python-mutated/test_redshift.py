from __future__ import annotations
from unittest import mock
from boto3.session import Session
from airflow.providers.amazon.aws.utils.redshift import build_credentials_block

class TestS3ToRedshiftTransfer:

    @mock.patch('boto3.session.Session')
    def test_build_credentials_block(self, mock_session):
        if False:
            return 10
        access_key = 'aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        token = 'aws_secret_token'
        mock_session.return_value = Session(access_key, secret_key)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = None
        credentials_block = build_credentials_block(mock_session.return_value)
        assert access_key in credentials_block
        assert secret_key in credentials_block
        assert token not in credentials_block

    @mock.patch('boto3.session.Session')
    def test_build_credentials_block_sts(self, mock_session):
        if False:
            while True:
                i = 10
        access_key = 'ASIA_aws_access_key_id'
        secret_key = 'aws_secret_access_key'
        token = 'aws_secret_token'
        mock_session.return_value = Session(access_key, secret_key)
        mock_session.return_value.access_key = access_key
        mock_session.return_value.secret_key = secret_key
        mock_session.return_value.token = token
        credentials_block = build_credentials_block(mock_session.return_value)
        assert access_key in credentials_block
        assert secret_key in credentials_block
        assert token in credentials_block
"""Unit tests."""
import re
import mock
import pandas
import pytest
from google.api_core import exceptions
from google.auth.credentials import AnonymousCredentials
from google.cloud import automl_v1beta1
PROJECT = 'project'

class TestGcsClient(object):

    def gcs_client(self, bucket_name=None, client_attrs={}):
        if False:
            for i in range(10):
                print('nop')
        client_mock = mock.Mock(**client_attrs)
        return automl_v1beta1.GcsClient(bucket_name=bucket_name, client=client_mock)

    def test_init_with_project_and_credentials(self):
        if False:
            i = 10
            return i + 15

        class FakeStorageClient:

            def __init__(self, project=None, credentials=None):
                if False:
                    return 10
                self.project = project
                self.credentials = credentials
        patch = mock.patch('google.cloud.storage.Client', new=FakeStorageClient)
        with patch:
            credentials = AnonymousCredentials()
            gcs_client = automl_v1beta1.GcsClient(project=PROJECT, credentials=credentials)
            assert isinstance(gcs_client.client, FakeStorageClient)
            assert gcs_client.client.project == PROJECT
            assert gcs_client.client.credentials == credentials

    def test_ensure_bucket_exists(self):
        if False:
            return 10
        mock_bucket = mock.Mock()
        gcs_client = self.gcs_client(client_attrs={'get_bucket.side_effect': exceptions.NotFound('err'), 'bucket.return_value': mock_bucket})
        returned_bucket_name = gcs_client.ensure_bucket_exists('my-project', 'us-central1')
        gcs_client.client.get_bucket.assert_called_with('my-project-automl-tables-staging')
        gcs_client.client.bucket.assert_called_with('my-project-automl-tables-staging')
        mock_bucket.create.assert_called_with(project='my-project', location='us-central1')
        assert returned_bucket_name == 'my-project-automl-tables-staging'

    def test_ensure_bucket_exists_bucket_already_exists_in_different_project(self):
        if False:
            while True:
                i = 10
        mock_bucket = mock.Mock()
        gcs_client = self.gcs_client(client_attrs={'get_bucket.side_effect': exceptions.Forbidden('err'), 'bucket.return_value': mock_bucket})
        returned_bucket_name = gcs_client.ensure_bucket_exists('my-project', 'us-central1')
        gcs_client.client.get_bucket.assert_called_with('my-project-automl-tables-staging')
        gcs_client.client.bucket.assert_called_with(returned_bucket_name)
        mock_bucket.create.assert_called_with(project='my-project', location='us-central1')
        assert re.match('^my-project-automl-tables-staging-[0-9]*$', returned_bucket_name)

    def test_ensure_bucket_exists_bucket_already_exists_in_current_project(self):
        if False:
            i = 10
            return i + 15
        gcs_client = self.gcs_client()
        returned_bucket_name = gcs_client.ensure_bucket_exists('my-project', 'us-central1')
        gcs_client.client.get_bucket.assert_called_with('my-project-automl-tables-staging')
        gcs_client.client.bucket.assert_not_called()
        assert returned_bucket_name == 'my-project-automl-tables-staging'

    def test_ensure_bucket_exists_custom_bucket_name(self):
        if False:
            while True:
                i = 10
        mock_bucket = mock.Mock()
        gcs_client = self.gcs_client(bucket_name='my-bucket', client_attrs={'get_bucket.side_effect': exceptions.NotFound('err'), 'bucket.return_value': mock_bucket})
        returned_bucket_name = gcs_client.ensure_bucket_exists('my-project', 'us-central1')
        gcs_client.client.get_bucket.assert_called_with('my-bucket')
        gcs_client.client.bucket.assert_called_with('my-bucket')
        mock_bucket.create.assert_called_with(project='my-project', location='us-central1')
        assert returned_bucket_name == 'my-bucket'

    def test_upload_pandas_dataframe(self):
        if False:
            for i in range(10):
                print('nop')
        mock_blob = mock.Mock()
        mock_bucket = mock.Mock(**{'blob.return_value': mock_blob})
        gcs_client = self.gcs_client(bucket_name='my-bucket', client_attrs={'get_bucket.return_value': mock_bucket})
        dataframe = pandas.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        gcs_uri = gcs_client.upload_pandas_dataframe(dataframe, 'my-file.csv')
        gcs_client.client.get_bucket.assert_called_with('my-bucket')
        mock_bucket.blob.assert_called_with('my-file.csv')
        mock_blob.upload_from_string.assert_called_with('col1,col2\n1,3\n2,4\n')
        assert gcs_uri == 'gs://my-bucket/my-file.csv'

    def test_upload_pandas_dataframe_no_csv_name(self):
        if False:
            i = 10
            return i + 15
        mock_blob = mock.Mock()
        mock_bucket = mock.Mock(**{'blob.return_value': mock_blob})
        gcs_client = self.gcs_client(bucket_name='my-bucket', client_attrs={'get_bucket.return_value': mock_bucket})
        dataframe = pandas.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        gcs_uri = gcs_client.upload_pandas_dataframe(dataframe)
        generated_csv_name = gcs_uri.split('/')[-1]
        gcs_client.client.get_bucket.assert_called_with('my-bucket')
        mock_bucket.blob.assert_called_with(generated_csv_name)
        mock_blob.upload_from_string.assert_called_with('col1,col2\n1,3\n2,4\n')
        assert re.match('^gs://my-bucket/automl-tables-dataframe-[0-9]*.csv$', gcs_uri)

    def test_upload_pandas_dataframe_not_type_dataframe(self):
        if False:
            while True:
                i = 10
        gcs_client = self.gcs_client()
        with pytest.raises(ValueError):
            gcs_client.upload_pandas_dataframe('my-dataframe')
        gcs_client.client.upload_pandas_dataframe.assert_not_called()

    def test_upload_pandas_dataframe_bucket_not_exist(self):
        if False:
            i = 10
            return i + 15
        gcs_client = self.gcs_client()
        dataframe = pandas.DataFrame({})
        with pytest.raises(ValueError):
            gcs_client.upload_pandas_dataframe(dataframe)
        gcs_client.client.upload_pandas_dataframe.assert_not_called()
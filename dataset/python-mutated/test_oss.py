from __future__ import annotations
from unittest import mock
from airflow.providers.alibaba.cloud.hooks.oss import OSSHook
from tests.providers.alibaba.cloud.utils.oss_mock import mock_oss_hook_default_project_id
OSS_STRING = 'airflow.providers.alibaba.cloud.hooks.oss.{}'
MOCK_OSS_CONN_ID = 'mock_id'
MOCK_BUCKET_NAME = 'mock_bucket_name'
MOCK_KEY = 'mock_key'
MOCK_KEYS = ['mock_key1', 'mock_key2', 'mock_key3']
MOCK_CONTENT = 'mock_content'
MOCK_FILE_PATH = 'mock_file_path'

class TestOSSHook:

    def setup_method(self):
        if False:
            print('Hello World!')
        with mock.patch(OSS_STRING.format('OSSHook.__init__'), new=mock_oss_hook_default_project_id):
            self.hook = OSSHook(oss_conn_id=MOCK_OSS_CONN_ID)

    def test_parse_oss_url(self):
        if False:
            while True:
                i = 10
        parsed = self.hook.parse_oss_url(f'oss://{MOCK_BUCKET_NAME}/this/is/not/a-real-key.txt')
        assert parsed == (MOCK_BUCKET_NAME, 'this/is/not/a-real-key.txt'), 'Incorrect parsing of the oss url'

    def test_parse_oss_object_directory(self):
        if False:
            return 10
        parsed = self.hook.parse_oss_url(f'oss://{MOCK_BUCKET_NAME}/this/is/not/a-real-oss-directory/')
        assert parsed == (MOCK_BUCKET_NAME, 'this/is/not/a-real-oss-directory/'), 'Incorrect parsing of the oss url'

    @mock.patch(OSS_STRING.format('oss2'))
    def test_get_credential(self, mock_oss2):
        if False:
            while True:
                i = 10
        self.hook.get_credential()
        mock_oss2.Auth.assert_called_once_with('mock_access_key_id', 'mock_access_key_secret')

    @mock.patch(OSS_STRING.format('OSSHook.get_credential'))
    @mock.patch(OSS_STRING.format('oss2'))
    def test_get_bucket(self, mock_oss2, mock_get_credential):
        if False:
            while True:
                i = 10
        self.hook.get_bucket('mock_bucket_name')
        mock_get_credential.assert_called_once_with()
        mock_oss2.Bucket.assert_called_once_with(mock_get_credential.return_value, 'https://oss-mock_region.aliyuncs.com', MOCK_BUCKET_NAME)

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_object_exist(self, mock_service):
        if False:
            print('Hello World!')
        mock_bucket = mock_service.return_value
        exists_method = mock_bucket.object_exists
        exists_method.return_value = True
        res = self.hook.object_exists(MOCK_KEY, MOCK_BUCKET_NAME)
        assert res is True
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        exists_method.assert_called_once_with(MOCK_KEY)

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_load_string(self, mock_service):
        if False:
            i = 10
            return i + 15
        self.hook.load_string(MOCK_KEY, MOCK_CONTENT, MOCK_BUCKET_NAME)
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.put_object.assert_called_once_with(MOCK_KEY, MOCK_CONTENT)

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_upload_local_file(self, mock_service):
        if False:
            print('Hello World!')
        self.hook.upload_local_file(MOCK_KEY, MOCK_FILE_PATH, MOCK_BUCKET_NAME)
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.put_object_from_file.assert_called_once_with(MOCK_KEY, MOCK_FILE_PATH)

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_download_file(self, mock_service):
        if False:
            return 10
        self.hook.download_file(MOCK_KEY, MOCK_FILE_PATH, MOCK_BUCKET_NAME)
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.get_object_to_file.assert_called_once_with(MOCK_KEY, MOCK_FILE_PATH)

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_delete_object(self, mock_service):
        if False:
            while True:
                i = 10
        self.hook.delete_object(MOCK_KEY, MOCK_BUCKET_NAME)
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.delete_object.assert_called_once_with(MOCK_KEY)

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_delete_objects(self, mock_service):
        if False:
            while True:
                i = 10
        self.hook.delete_objects(MOCK_KEYS, MOCK_BUCKET_NAME)
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.batch_delete_objects.assert_called_once_with(MOCK_KEYS)

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_delete_bucket(self, mock_service):
        if False:
            return 10
        self.hook.delete_bucket(MOCK_BUCKET_NAME)
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.delete_bucket.assert_called_once_with()

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_create_bucket(self, mock_service):
        if False:
            for i in range(10):
                print('nop')
        self.hook.create_bucket(MOCK_BUCKET_NAME)
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.create_bucket.assert_called_once_with()

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_append_string(self, mock_service):
        if False:
            return 10
        self.hook.append_string(MOCK_BUCKET_NAME, MOCK_CONTENT, MOCK_KEY, 0)
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.append_object.assert_called_once_with(MOCK_KEY, 0, MOCK_CONTENT)

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_read_key(self, mock_service):
        if False:
            return 10
        mock_service.return_value.get_object.return_value.read.return_value.decode.return_value = MOCK_CONTENT
        res = self.hook.read_key(MOCK_BUCKET_NAME, MOCK_KEY)
        assert res == MOCK_CONTENT
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.get_object.assert_called_once_with(MOCK_KEY)
        mock_service.return_value.get_object.return_value.read.assert_called_once_with()
        mock_service.return_value.get_object.return_value.read.return_value.decode.assert_called_once_with('utf-8')

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_head_key(self, mock_service):
        if False:
            for i in range(10):
                print('nop')
        self.hook.head_key(MOCK_BUCKET_NAME, MOCK_KEY)
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.head_object.assert_called_once_with(MOCK_KEY)

    @mock.patch(OSS_STRING.format('OSSHook.get_bucket'))
    def test_key_exists(self, mock_service):
        if False:
            for i in range(10):
                print('nop')
        mock_service.return_value.object_exists.return_value = True
        res = self.hook.key_exist(MOCK_BUCKET_NAME, MOCK_KEY)
        assert res is True
        mock_service.assert_called_once_with(MOCK_BUCKET_NAME)
        mock_service.return_value.object_exists.assert_called_once_with(MOCK_KEY)

    def test_get_default_region(self):
        if False:
            while True:
                i = 10
        assert self.hook.get_default_region() == 'mock_region'
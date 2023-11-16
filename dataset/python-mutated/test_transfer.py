import pathlib
from tempfile import NamedTemporaryFile
import pytest
from s3transfer.futures import NonThreadedExecutor
from s3transfer.manager import TransferManager
from boto3.exceptions import RetriesExceededError, S3UploadFailedError
from boto3.s3.transfer import KB, MB, ClientError, OSUtils, ProgressCallbackInvoker, S3Transfer, S3TransferRetriesExceededError, TransferConfig, create_transfer_manager
from tests import mock, unittest

class TestCreateTransferManager(unittest.TestCase):

    def test_create_transfer_manager(self):
        if False:
            i = 10
            return i + 15
        client = object()
        config = TransferConfig()
        osutil = OSUtils()
        with mock.patch('boto3.s3.transfer.TransferManager') as manager:
            create_transfer_manager(client, config, osutil)
            assert manager.call_args == mock.call(client, config, osutil, None)

    def test_create_transfer_manager_with_no_threads(self):
        if False:
            print('Hello World!')
        client = object()
        config = TransferConfig()
        config.use_threads = False
        with mock.patch('boto3.s3.transfer.TransferManager') as manager:
            create_transfer_manager(client, config)
            assert manager.call_args == mock.call(client, config, None, NonThreadedExecutor)

class TestTransferConfig(unittest.TestCase):

    def assert_value_of_actual_and_alias(self, config, actual, alias, ref_value):
        if False:
            print('Hello World!')
        assert getattr(config, actual) == ref_value
        assert getattr(config, alias) == ref_value

    def test_alias_max_concurreny(self):
        if False:
            print('Hello World!')
        ref_value = 10
        config = TransferConfig(max_concurrency=ref_value)
        self.assert_value_of_actual_and_alias(config, 'max_request_concurrency', 'max_concurrency', ref_value)
        new_value = 15
        config.max_concurrency = new_value
        self.assert_value_of_actual_and_alias(config, 'max_request_concurrency', 'max_concurrency', new_value)

    def test_alias_max_io_queue(self):
        if False:
            for i in range(10):
                print('nop')
        ref_value = 10
        config = TransferConfig(max_io_queue=ref_value)
        self.assert_value_of_actual_and_alias(config, 'max_io_queue_size', 'max_io_queue', ref_value)
        new_value = 15
        config.max_io_queue = new_value
        self.assert_value_of_actual_and_alias(config, 'max_io_queue_size', 'max_io_queue', new_value)

    def test_transferconfig_parameters(self):
        if False:
            i = 10
            return i + 15
        config = TransferConfig(multipart_threshold=8 * MB, max_concurrency=10, multipart_chunksize=8 * MB, num_download_attempts=5, max_io_queue=100, io_chunksize=256 * KB, use_threads=True, max_bandwidth=1024 * KB)
        assert config.multipart_threshold == 8 * MB
        assert config.multipart_chunksize == 8 * MB
        assert config.max_request_concurrency == 10
        assert config.num_download_attempts == 5
        assert config.max_io_queue_size == 100
        assert config.io_chunksize == 256 * KB
        assert config.use_threads is True
        assert config.max_bandwidth == 1024 * KB

class TestProgressCallbackInvoker(unittest.TestCase):

    def test_on_progress(self):
        if False:
            for i in range(10):
                print('nop')
        callback = mock.Mock()
        subscriber = ProgressCallbackInvoker(callback)
        subscriber.on_progress(bytes_transferred=1)
        callback.assert_called_with(1)

class TestS3Transfer(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.client = mock.Mock()
        self.manager = mock.Mock(TransferManager(self.client))
        self.transfer = S3Transfer(manager=self.manager)
        self.callback = mock.Mock()
        with NamedTemporaryFile('w') as tmp_file:
            self.file_path_str = tmp_file.name

    def assert_callback_wrapped_in_subscriber(self, call_args):
        if False:
            i = 10
            return i + 15
        subscribers = call_args[0][4]
        assert len(subscribers) == 1
        subscriber = subscribers[0]
        assert isinstance(subscriber, ProgressCallbackInvoker)
        subscriber.on_progress(bytes_transferred=1)
        self.callback.assert_called_with(1)

    def test_upload_file(self):
        if False:
            for i in range(10):
                print('nop')
        extra_args = {'ACL': 'public-read'}
        self.transfer.upload_file('smallfile', 'bucket', 'key', extra_args=extra_args)
        self.manager.upload.assert_called_with('smallfile', 'bucket', 'key', extra_args, None)

    def test_upload_file_via_path(self):
        if False:
            while True:
                i = 10
        extra_args = {'ACL': 'public-read'}
        self.transfer.upload_file(pathlib.Path(self.file_path_str), 'bucket', 'key', extra_args=extra_args)
        self.manager.upload.assert_called_with(self.file_path_str, 'bucket', 'key', extra_args, None)

    def test_upload_file_via_purepath(self):
        if False:
            return 10
        extra_args = {'ACL': 'public-read'}
        self.transfer.upload_file(pathlib.PurePath(self.file_path_str), 'bucket', 'key', extra_args=extra_args)
        self.manager.upload.assert_called_with(self.file_path_str, 'bucket', 'key', extra_args, None)

    def test_download_file(self):
        if False:
            while True:
                i = 10
        extra_args = {'SSECustomerKey': 'foo', 'SSECustomerAlgorithm': 'AES256'}
        self.transfer.download_file('bucket', 'key', self.file_path_str, extra_args=extra_args)
        self.manager.download.assert_called_with('bucket', 'key', self.file_path_str, extra_args, None)

    def test_download_file_via_path(self):
        if False:
            print('Hello World!')
        extra_args = {'SSECustomerKey': 'foo', 'SSECustomerAlgorithm': 'AES256'}
        self.transfer.download_file('bucket', 'key', pathlib.Path(self.file_path_str), extra_args=extra_args)
        self.manager.download.assert_called_with('bucket', 'key', self.file_path_str, extra_args, None)

    def test_upload_wraps_callback(self):
        if False:
            i = 10
            return i + 15
        self.transfer.upload_file('smallfile', 'bucket', 'key', callback=self.callback)
        self.assert_callback_wrapped_in_subscriber(self.manager.upload.call_args)

    def test_download_wraps_callback(self):
        if False:
            return 10
        self.transfer.download_file('bucket', 'key', '/tmp/smallfile', callback=self.callback)
        self.assert_callback_wrapped_in_subscriber(self.manager.download.call_args)

    def test_propogation_of_retry_error(self):
        if False:
            i = 10
            return i + 15
        future = mock.Mock()
        future.result.side_effect = S3TransferRetriesExceededError(Exception())
        self.manager.download.return_value = future
        with pytest.raises(RetriesExceededError):
            self.transfer.download_file('bucket', 'key', '/tmp/smallfile')

    def test_propogation_s3_upload_failed_error(self):
        if False:
            print('Hello World!')
        future = mock.Mock()
        future.result.side_effect = ClientError({'Error': {}}, 'op_name')
        self.manager.upload.return_value = future
        with pytest.raises(S3UploadFailedError):
            self.transfer.upload_file('smallfile', 'bucket', 'key')

    def test_can_create_with_just_client(self):
        if False:
            for i in range(10):
                print('nop')
        transfer = S3Transfer(client=mock.Mock())
        assert isinstance(transfer, S3Transfer)

    def test_can_create_with_extra_configurations(self):
        if False:
            while True:
                i = 10
        transfer = S3Transfer(client=mock.Mock(), config=TransferConfig(), osutil=OSUtils())
        assert isinstance(transfer, S3Transfer)

    def test_client_or_manager_is_required(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError):
            S3Transfer()

    def test_client_and_manager_are_mutually_exclusive(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            S3Transfer(self.client, manager=self.manager)

    def test_config_and_manager_are_mutually_exclusive(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            S3Transfer(config=mock.Mock(), manager=self.manager)

    def test_osutil_and_manager_are_mutually_exclusive(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            S3Transfer(osutil=mock.Mock(), manager=self.manager)

    def test_upload_requires_string_filename(self):
        if False:
            return 10
        transfer = S3Transfer(client=mock.Mock())
        with pytest.raises(ValueError):
            transfer.upload_file(filename=object(), bucket='foo', key='bar')

    def test_download_requires_string_filename(self):
        if False:
            while True:
                i = 10
        transfer = S3Transfer(client=mock.Mock())
        with pytest.raises(ValueError):
            transfer.download_file(bucket='foo', key='bar', filename=object())

    def test_context_manager(self):
        if False:
            return 10
        manager = mock.Mock()
        manager.__exit__ = mock.Mock()
        with S3Transfer(manager=manager):
            pass
        assert manager.__exit__.call_args == mock.call(None, None, None)

    def test_context_manager_with_errors(self):
        if False:
            for i in range(10):
                print('nop')
        manager = mock.Mock()
        manager.__exit__ = mock.Mock()
        raised_exception = ValueError()
        with pytest.raises(type(raised_exception)):
            with S3Transfer(manager=manager):
                raise raised_exception
        assert manager.__exit__.call_args == mock.call(type(raised_exception), raised_exception, mock.ANY)
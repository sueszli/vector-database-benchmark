import datetime
from unittest import mock
from freezegun import freeze_time
from golem.resource.client import ClientOptions
from golem.resource.resourcemanager import ResourceManager, ResourceId
from golem.testutils import TempDirFixture
NOW = 1555555555
TIMEOUT = 10
VALID_TO = 1555565555

class TestResourceManager(TempDirFixture):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.client_options = ClientOptions(client_id='mocked', version=1.0, options={'timeout': TIMEOUT})
        client = mock.Mock()
        client.add_async.return_value = ResourceId('0x0')
        client.get_async.return_value = ['0x0', ['mocked.file']]
        client.resource_async.return_value = {'validTo': VALID_TO}
        client.build_options.return_value = self.client_options
        self.resource_manager = ResourceManager(client)
        self.client = client

    def test_types(self):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(ResourceId('0x0'), str)

    def test_build_client_options(self):
        if False:
            print('Hello World!')
        assert isinstance(self.resource_manager.build_client_options(), ClientOptions)

    def test_share(self):
        if False:
            print('Hello World!')
        sample_path = self.new_path / 'sample.txt'
        response = self.resource_manager.share(sample_path, self.client_options)
        assert isinstance(response.result, str)
        assert self.resource_manager._cache[sample_path] == response.result

    @freeze_time(datetime.datetime.fromtimestamp(NOW))
    def test_share_cached(self):
        if False:
            for i in range(10):
                print('nop')
        sample_path = self.new_path / 'sample.txt'
        self.resource_manager.share(sample_path, self.client_options)
        assert self.client.add_async.call_count == 1
        self.resource_manager.share(sample_path, self.client_options)
        assert self.client.add_async.call_count == 1

    @freeze_time(datetime.datetime.fromtimestamp(NOW))
    def test_share_cache_timed_out(self):
        if False:
            return 10
        sample_path = self.new_path / 'sample.txt'
        self.client.resource_async.return_value = {'validTo': NOW + TIMEOUT - 1}
        self.resource_manager.share(sample_path, self.client_options)
        assert self.client.cancel_async.call_count == 0
        assert self.client.add_async.call_count == 1
        self.resource_manager.share(sample_path, self.client_options)
        assert self.client.cancel_async.call_count == 1
        assert self.client.add_async.call_count == 2

    def test_download(self):
        if False:
            i = 10
            return i + 15
        resource_id = ResourceId('0x0')
        sample_dir = self.new_path / 'directory'
        sample_dir.mkdir(parents=True)
        self.resource_manager.download(resource_id, sample_dir, self.client_options)
        assert self.client.get_async.called

    def test_drop(self):
        if False:
            while True:
                i = 10
        sample_path = self.new_path / 'sample.txt'
        response = self.resource_manager.share(sample_path, self.client_options)
        assert sample_path in self.resource_manager._cache
        self.resource_manager.drop(response.result)
        assert sample_path not in self.resource_manager._cache
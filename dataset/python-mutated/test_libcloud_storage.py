"""
    :codeauthor: :email:`Anthony Shaw <anthonyshaw@apache.org>`
"""
import pytest
import salt.modules.libcloud_storage as libcloud_storage
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.mock import MagicMock, patch
from tests.support.unit import TestCase
try:
    from libcloud.storage.base import BaseDriver, Container, Object
    HAS_LIBCLOUD = True
except ImportError:
    HAS_LIBCLOUD = False
if HAS_LIBCLOUD:

    class MockStorageDriver(BaseDriver):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self._TEST_CONTAINER = Container(name='test_container', extra={}, driver=self)
            self._TEST_OBJECT = Object(name='test_obj', size=1234, hash='123sdfsdf', extra={}, meta_data={'key': 'value'}, container=self._TEST_CONTAINER, driver=self)

        def list_containers(self):
            if False:
                i = 10
                return i + 15
            return [self._TEST_CONTAINER]

        def get_container(self, container_name):
            if False:
                print('Hello World!')
            assert container_name == 'test_container'
            return self._TEST_CONTAINER

        def list_container_objects(self, container):
            if False:
                i = 10
                return i + 15
            assert container.name == 'test_container'
            return [self._TEST_OBJECT]

        def create_container(self, container_name):
            if False:
                return 10
            assert container_name == 'new_test_container'
            return self._TEST_CONTAINER

        def get_container_object(self, container_name, object_name):
            if False:
                while True:
                    i = 10
            assert container_name == 'test_container'
            assert object_name == 'test_obj'
            return self._TEST_OBJECT
else:
    MockStorageDriver = object

def get_mock_driver():
    if False:
        while True:
            i = 10
    return MockStorageDriver()

@pytest.mark.skipif(not HAS_LIBCLOUD, reason='No libcloud installed')
@patch('salt.modules.libcloud_storage._get_driver', MagicMock(return_value=MockStorageDriver()))
class LibcloudStorageModuleTestCase(TestCase, LoaderModuleMockMixin):

    def setup_loader_modules(self):
        if False:
            i = 10
            return i + 15
        module_globals = {'__salt__': {'config.option': MagicMock(return_value={'test': {'driver': 'test', 'key': '2orgk34kgk34g'}})}}
        if libcloud_storage.HAS_LIBCLOUD is False:
            module_globals['sys.modules'] = {'libcloud': MagicMock()}
        return {libcloud_storage: module_globals}

    def test_module_creation(self):
        if False:
            while True:
                i = 10
        client = libcloud_storage._get_driver('test')
        self.assertFalse(client is None)

    def test_list_containers(self):
        if False:
            print('Hello World!')
        containers = libcloud_storage.list_containers('test')
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0]['name'], 'test_container')

    def test_list_container_objects(self):
        if False:
            return 10
        objects = libcloud_storage.list_container_objects('test_container', 'test')
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]['name'], 'test_obj')
        self.assertEqual(objects[0]['size'], 1234)

    def test_create_container(self):
        if False:
            while True:
                i = 10
        container = libcloud_storage.create_container('new_test_container', 'test')
        self.assertEqual(container['name'], 'test_container')

    def test_get_container(self):
        if False:
            print('Hello World!')
        container = libcloud_storage.get_container('test_container', 'test')
        self.assertEqual(container['name'], 'test_container')

    def test_get_container_object(self):
        if False:
            for i in range(10):
                print('nop')
        obj = libcloud_storage.get_container_object('test_container', 'test_obj', 'test')
        self.assertEqual(obj['name'], 'test_obj')
        self.assertEqual(obj['size'], 1234)
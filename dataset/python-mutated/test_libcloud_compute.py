"""
    :codeauthor: :email:`Anthony Shaw <anthonyshaw@apache.org>`
"""
import logging
import pytest
import salt.modules.libcloud_compute as libcloud_compute
from salt.utils.versions import Version
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.mock import MagicMock, patch
from tests.support.unit import TestCase
REQUIRED_LIBCLOUD_VERSION = '2.0.0'
try:
    import libcloud
    from libcloud.compute.base import BaseDriver, KeyPair, Node, NodeImage, NodeLocation, NodeSize, NodeState, StorageVolume, StorageVolumeState, VolumeSnapshot
    if hasattr(libcloud, '__version__') and Version(libcloud.__version__) < Version(REQUIRED_LIBCLOUD_VERSION):
        raise ImportError()
    logging.getLogger('libcloud').setLevel(logging.CRITICAL)
    HAS_LIBCLOUD = True
except ImportError:
    HAS_LIBCLOUD = False
if HAS_LIBCLOUD:

    class MockComputeDriver(BaseDriver):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self._TEST_SIZE = NodeSize(id='test_id', name='test_size', ram=4096, disk=10240, bandwidth=100000, price=0, driver=self)
            self._TEST_NODE = Node(id='test_id', name='test_node', state=NodeState.RUNNING, public_ips=['1.2.3.4'], private_ips=['2.3.4.5'], driver=self, size=self._TEST_SIZE, extra={'ex_key': 'ex_value'})
            self._TEST_LOCATION = NodeLocation(id='test_location', name='location1', country='Australia', driver=self)
            self._TEST_VOLUME = StorageVolume(id='vol1', name='vol_name', size=40960, driver=self, state=StorageVolumeState.AVAILABLE, extra={'ex_key': 'ex_value'})
            self._TEST_VOLUME_SNAPSHOT = VolumeSnapshot(id='snap1', size=80960, driver=self)
            self._TEST_IMAGE = NodeImage(id='image1', name='test_image', extra={'ex_key': 'ex_value'}, driver=self)
            self._TEST_KEY_PAIR = KeyPair(name='test_key', fingerprint='abc123', public_key='pub123', private_key='priv123', driver=self, extra={'ex_key': 'ex_value'})

        def list_nodes(self):
            if False:
                print('Hello World!')
            return [self._TEST_NODE]

        def list_sizes(self, location=None):
            if False:
                for i in range(10):
                    print('nop')
            if location:
                assert location.id == 'test_location'
            return [self._TEST_SIZE]

        def list_locations(self):
            if False:
                print('Hello World!')
            return [self._TEST_LOCATION]

        def reboot_node(self, node):
            if False:
                while True:
                    i = 10
            assert node.id == 'test_id'
            return True

        def destroy_node(self, node):
            if False:
                return 10
            assert node.id == 'test_id'
            return True

        def list_volumes(self):
            if False:
                for i in range(10):
                    print('nop')
            return [self._TEST_VOLUME]

        def list_volume_snapshots(self, volume):
            if False:
                print('Hello World!')
            assert volume.id == 'vol1'
            return [self._TEST_VOLUME_SNAPSHOT]

        def create_volume(self, size, name, location=None, snapshot=None):
            if False:
                for i in range(10):
                    print('nop')
            assert size == 9000
            assert name == 'test_new_volume'
            if location:
                assert location.country == 'Australia'
            return self._TEST_VOLUME

        def create_volume_snapshot(self, volume, name=None):
            if False:
                i = 10
                return i + 15
            assert volume.id == 'vol1'
            if name:
                assert name == 'test_snapshot'
            return self._TEST_VOLUME_SNAPSHOT

        def attach_volume(self, node, volume, device=None):
            if False:
                return 10
            assert node.id == 'test_id'
            assert volume.id == 'vol1'
            if device:
                assert device == '/dev/sdc'
            return True

        def detach_volume(self, volume):
            if False:
                for i in range(10):
                    print('nop')
            assert volume.id == 'vol1'
            return True

        def destroy_volume(self, volume):
            if False:
                i = 10
                return i + 15
            assert volume.id == 'vol1'
            return True

        def destroy_volume_snapshot(self, snapshot):
            if False:
                while True:
                    i = 10
            assert snapshot.id == 'snap1'
            return True

        def list_images(self, location=None):
            if False:
                for i in range(10):
                    print('nop')
            if location:
                assert location.id == 'test_location'
            return [self._TEST_IMAGE]

        def create_image(self, node, name, description=None):
            if False:
                return 10
            assert node.id == 'test_id'
            return self._TEST_IMAGE

        def delete_image(self, node_image):
            if False:
                while True:
                    i = 10
            return True

        def get_image(self, image_id):
            if False:
                return 10
            assert image_id == 'image1'
            return self._TEST_IMAGE

        def copy_image(self, source_region, node_image, name, description=None):
            if False:
                for i in range(10):
                    print('nop')
            assert source_region == 'us-east1'
            assert node_image.id == 'image1'
            assert name == 'copy_test'
            return self._TEST_IMAGE

        def list_key_pairs(self):
            if False:
                while True:
                    i = 10
            return [self._TEST_KEY_PAIR]

        def get_key_pair(self, name):
            if False:
                print('Hello World!')
            assert name == 'test_key'
            return self._TEST_KEY_PAIR

        def create_key_pair(self, name):
            if False:
                while True:
                    i = 10
            assert name == 'test_key'
            return self._TEST_KEY_PAIR

        def import_key_pair_from_string(self, name, key_material):
            if False:
                print('Hello World!')
            assert name == 'test_key'
            assert key_material == 'test_key_value'
            return self._TEST_KEY_PAIR

        def import_key_pair_from_file(self, name, key_file_path):
            if False:
                i = 10
                return i + 15
            assert name == 'test_key'
            assert key_file_path == '/path/to/key'
            return self._TEST_KEY_PAIR

        def delete_key_pair(self, key_pair):
            if False:
                while True:
                    i = 10
            assert key_pair.name == 'test_key'
            return True
else:
    MockComputeDriver = object

@pytest.mark.skipif(not HAS_LIBCLOUD, reason='No libcloud installed')
@patch('salt.modules.libcloud_compute._get_driver', MagicMock(return_value=MockComputeDriver()))
class LibcloudComputeModuleTestCase(TestCase, LoaderModuleMockMixin):

    def setup_loader_modules(self):
        if False:
            for i in range(10):
                print('nop')
        module_globals = {'__salt__': {'config.option': MagicMock(return_value={'test': {'driver': 'test', 'key': '2orgk34kgk34g'}})}}
        if libcloud_compute.HAS_LIBCLOUD is False:
            module_globals['sys.modules'] = {'libcloud': MagicMock()}
        return {libcloud_compute: module_globals}

    def test_module_creation(self):
        if False:
            for i in range(10):
                print('nop')
        client = libcloud_compute._get_driver('test')
        self.assertFalse(client is None)

    def _validate_node(self, node):
        if False:
            return 10
        self.assertEqual(node['name'], 'test_node')
        self.assertEqual(node['id'], 'test_id')
        self.assertEqual(node['private_ips'], ['2.3.4.5'])
        self.assertEqual(node['public_ips'], ['1.2.3.4'])
        self.assertEqual(node['size']['name'], 'test_size')

    def _validate_size(self, size):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(size['id'], 'test_id')
        self.assertEqual(size['name'], 'test_size')
        self.assertEqual(size['ram'], 4096)

    def _validate_location(self, location):
        if False:
            print('Hello World!')
        self.assertEqual(location['id'], 'test_location')
        self.assertEqual(location['name'], 'location1')
        self.assertEqual(location['country'], 'Australia')

    def _validate_volume(self, volume):
        if False:
            print('Hello World!')
        self.assertEqual(volume['id'], 'vol1')
        self.assertEqual(volume['name'], 'vol_name')
        self.assertEqual(volume['size'], 40960)
        self.assertEqual(volume['state'], 'available')
        self.assertEqual(volume['extra'], {'ex_key': 'ex_value'})

    def _validate_volume_snapshot(self, volume):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(volume['id'], 'snap1')
        self.assertEqual(volume['size'], 80960)

    def _validate_image(self, image):
        if False:
            i = 10
            return i + 15
        self.assertEqual(image['id'], 'image1')
        self.assertEqual(image['name'], 'test_image')
        self.assertEqual(image['extra'], {'ex_key': 'ex_value'})

    def _validate_key_pair(self, key):
        if False:
            i = 10
            return i + 15
        self.assertEqual(key['name'], 'test_key')
        self.assertEqual(key['fingerprint'], 'abc123')
        self.assertEqual(key['extra'], {'ex_key': 'ex_value'})

    def test_list_nodes(self):
        if False:
            print('Hello World!')
        nodes = libcloud_compute.list_nodes('test')
        self.assertEqual(len(nodes), 1)
        self._validate_node(nodes[0])

    def test_list_sizes(self):
        if False:
            print('Hello World!')
        sizes = libcloud_compute.list_sizes('test')
        self.assertEqual(len(sizes), 1)
        self._validate_size(sizes[0])

    def test_list_sizes_location(self):
        if False:
            return 10
        sizes = libcloud_compute.list_sizes('test', location_id='test_location')
        self.assertEqual(len(sizes), 1)
        self._validate_size(sizes[0])

    def test_list_locations(self):
        if False:
            while True:
                i = 10
        locations = libcloud_compute.list_locations('test')
        self.assertEqual(len(locations), 1)
        self._validate_location(locations[0])

    def test_reboot_node(self):
        if False:
            while True:
                i = 10
        result = libcloud_compute.reboot_node('test_id', 'test')
        self.assertTrue(result)

    def test_reboot_node_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            libcloud_compute.reboot_node('foo_node', 'test')

    def test_destroy_node(self):
        if False:
            print('Hello World!')
        result = libcloud_compute.destroy_node('test_id', 'test')
        self.assertTrue(result)

    def test_destroy_node_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            libcloud_compute.destroy_node('foo_node', 'test')

    def test_list_volumes(self):
        if False:
            for i in range(10):
                print('nop')
        volumes = libcloud_compute.list_volumes('test')
        self.assertEqual(len(volumes), 1)
        self._validate_volume(volumes[0])

    def test_list_volume_snapshots(self):
        if False:
            while True:
                i = 10
        volumes = libcloud_compute.list_volume_snapshots('vol1', 'test')
        self.assertEqual(len(volumes), 1)
        self._validate_volume_snapshot(volumes[0])

    def test_create_volume(self):
        if False:
            i = 10
            return i + 15
        volume = libcloud_compute.create_volume(9000, 'test_new_volume', 'test')
        self._validate_volume(volume)

    def test_create_volume_in_location(self):
        if False:
            i = 10
            return i + 15
        volume = libcloud_compute.create_volume(9000, 'test_new_volume', 'test', location_id='test_location')
        self._validate_volume(volume)

    def test_create_volume_snapshot(self):
        if False:
            return 10
        snapshot = libcloud_compute.create_volume_snapshot('vol1', 'test')
        self._validate_volume_snapshot(snapshot)

    def test_create_volume_snapshot_named(self):
        if False:
            while True:
                i = 10
        snapshot = libcloud_compute.create_volume_snapshot('vol1', 'test', name='test_snapshot')
        self._validate_volume_snapshot(snapshot)

    def test_attach_volume(self):
        if False:
            print('Hello World!')
        result = libcloud_compute.attach_volume('test_id', 'vol1', 'test')
        self.assertTrue(result)

    def test_detatch_volume(self):
        if False:
            i = 10
            return i + 15
        result = libcloud_compute.detach_volume('vol1', 'test')
        self.assertTrue(result)

    def test_destroy_volume(self):
        if False:
            return 10
        result = libcloud_compute.destroy_volume('vol1', 'test')
        self.assertTrue(result)

    def test_destroy_volume_snapshot(self):
        if False:
            while True:
                i = 10
        result = libcloud_compute.destroy_volume_snapshot('vol1', 'snap1', 'test')
        self.assertTrue(result)

    def test_list_images(self):
        if False:
            return 10
        images = libcloud_compute.list_images('test')
        self.assertEqual(len(images), 1)
        self._validate_image(images[0])

    def test_list_images_in_location(self):
        if False:
            for i in range(10):
                print('nop')
        images = libcloud_compute.list_images('test', location_id='test_location')
        self.assertEqual(len(images), 1)
        self._validate_image(images[0])

    def test_create_image(self):
        if False:
            return 10
        image = libcloud_compute.create_image('test_id', 'new_image', 'test')
        self._validate_image(image)

    def test_delete_image(self):
        if False:
            print('Hello World!')
        result = libcloud_compute.delete_image('image1', 'test')
        self.assertTrue(result)

    def test_get_image(self):
        if False:
            while True:
                i = 10
        image = libcloud_compute.get_image('image1', 'test')
        self._validate_image(image)

    def test_copy_image(self):
        if False:
            while True:
                i = 10
        new_image = libcloud_compute.copy_image('us-east1', 'image1', 'copy_test', 'test')
        self._validate_image(new_image)

    def test_list_key_pairs(self):
        if False:
            while True:
                i = 10
        keys = libcloud_compute.list_key_pairs('test')
        self.assertEqual(len(keys), 1)
        self._validate_key_pair(keys[0])

    def test_get_key_pair(self):
        if False:
            return 10
        key = libcloud_compute.get_key_pair('test_key', 'test')
        self._validate_key_pair(key)

    def test_create_key_pair(self):
        if False:
            print('Hello World!')
        key = libcloud_compute.create_key_pair('test_key', 'test')
        self._validate_key_pair(key)

    def test_import_key_string(self):
        if False:
            return 10
        key = libcloud_compute.import_key_pair('test_key', 'test_key_value', 'test')
        self._validate_key_pair(key)

    def test_import_key_file(self):
        if False:
            return 10
        key = libcloud_compute.import_key_pair('test_key', '/path/to/key', 'test', key_type='FILE')
        self._validate_key_pair(key)

    def test_delete_key_pair(self):
        if False:
            return 10
        result = libcloud_compute.delete_key_pair('test_key', 'test')
        self.assertTrue(result)
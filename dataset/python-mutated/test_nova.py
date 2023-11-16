"""
    :codeauthor: Rahul Handay <rahulha@saltstack.com>
"""
import salt.modules.nova as nova
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.mock import MagicMock, patch
from tests.support.unit import TestCase

class NovaTestCase(TestCase, LoaderModuleMockMixin):
    """
    Test cases for salt.modules.nova
    """

    def setup_loader_modules(self):
        if False:
            for i in range(10):
                print('nop')
        patcher = patch('salt.modules.nova._auth')
        self.mock_auth = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(delattr, self, 'mock_auth')
        return {nova: {}}

    def test_boot(self):
        if False:
            while True:
                i = 10
        '\n        Test for Boot (create) a new instance\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'boot', MagicMock(return_value='A')):
            self.assertTrue(nova.boot('name'))

    def test_volume_list(self):
        if False:
            i = 10
            return i + 15
        '\n        Test for List storage volumes\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'volume_list', MagicMock(return_value='A')):
            self.assertTrue(nova.volume_list())

    def test_volume_show(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test for Create a block storage volume\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'volume_show', MagicMock(return_value='A')):
            self.assertTrue(nova.volume_show('name'))

    def test_volume_create(self):
        if False:
            i = 10
            return i + 15
        '\n        Test for Create a block storage volume\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'volume_create', MagicMock(return_value='A')):
            self.assertTrue(nova.volume_create('name'))

    def test_volume_delete(self):
        if False:
            return 10
        '\n        Test for Destroy the volume\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'volume_delete', MagicMock(return_value='A')):
            self.assertTrue(nova.volume_delete('name'))

    def test_volume_detach(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test for Attach a block storage volume\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'volume_detach', MagicMock(return_value='A')):
            self.assertTrue(nova.volume_detach('name'))

    def test_volume_attach(self):
        if False:
            return 10
        '\n        Test for Attach a block storage volume\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'volume_attach', MagicMock(return_value='A')):
            self.assertTrue(nova.volume_attach('name', 'serv_name'))

    def test_suspend(self):
        if False:
            while True:
                i = 10
        '\n        Test for Suspend an instance\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'suspend', MagicMock(return_value='A')):
            self.assertTrue(nova.suspend('instance_id'))

    def test_resume(self):
        if False:
            i = 10
            return i + 15
        '\n        Test for Resume an instance\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'resume', MagicMock(return_value='A')):
            self.assertTrue(nova.resume('instance_id'))

    def test_lock(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test for Lock an instance\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'lock', MagicMock(return_value='A')):
            self.assertTrue(nova.lock('instance_id'))

    def test_delete(self):
        if False:
            while True:
                i = 10
        '\n        Test for Delete an instance\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'delete', MagicMock(return_value='A')):
            self.assertTrue(nova.delete('instance_id'))

    def test_flavor_list(self):
        if False:
            while True:
                i = 10
        '\n        Test for Return a list of available flavors (nova flavor-list)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'flavor_list', MagicMock(return_value='A')):
            self.assertTrue(nova.flavor_list())

    def test_flavor_create(self):
        if False:
            print('Hello World!')
        '\n        Test for Add a flavor to nova (nova flavor-create)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'flavor_create', MagicMock(return_value='A')):
            self.assertTrue(nova.flavor_create('name'))

    def test_flavor_delete(self):
        if False:
            while True:
                i = 10
        '\n        Test for Delete a flavor from nova by id (nova flavor-delete)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'flavor_delete', MagicMock(return_value='A')):
            self.assertTrue(nova.flavor_delete('flavor_id'))

    def test_keypair_list(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test for Return a list of available keypairs (nova keypair-list)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'keypair_list', MagicMock(return_value='A')):
            self.assertTrue(nova.keypair_list())

    def test_keypair_add(self):
        if False:
            print('Hello World!')
        '\n        Test for Add a keypair to nova (nova keypair-add)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'keypair_add', MagicMock(return_value='A')):
            self.assertTrue(nova.keypair_add('name'))

    def test_keypair_delete(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test for Add a keypair to nova (nova keypair-delete)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'keypair_delete', MagicMock(return_value='A')):
            self.assertTrue(nova.keypair_delete('name'))

    def test_image_list(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test for Return a list of available images\n         (nova images-list + nova image-show)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'image_list', MagicMock(return_value='A')):
            self.assertTrue(nova.image_list())

    def test_image_meta_set(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test for Sets a key=value pair in the\n         metadata for an image (nova image-meta set)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'image_meta_set', MagicMock(return_value='A')):
            self.assertTrue(nova.image_meta_set())

    def test_image_meta_delete(self):
        if False:
            while True:
                i = 10
        '\n        Test for Delete a key=value pair from the metadata for an image\n        (nova image-meta set)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'image_meta_delete', MagicMock(return_value='A')):
            self.assertTrue(nova.image_meta_delete())

    def test_list_(self):
        if False:
            while True:
                i = 10
        '\n        Test for To maintain the feel of the nova command line,\n         this function simply calls\n         the server_list function.\n        '
        with patch.object(nova, 'server_list', return_value=['A']):
            self.assertEqual(nova.list_(), ['A'])

    def test_server_list(self):
        if False:
            i = 10
            return i + 15
        '\n        Test for Return list of active servers\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'server_list', MagicMock(return_value='A')):
            self.assertTrue(nova.server_list())

    def test_show(self):
        if False:
            i = 10
            return i + 15
        '\n        Test for To maintain the feel of the nova command line,\n         this function simply calls\n         the server_show function.\n        '
        with patch.object(nova, 'server_show', return_value=['A']):
            self.assertEqual(nova.show('server_id'), ['A'])

    def test_server_list_detailed(self):
        if False:
            return 10
        '\n        Test for Return detailed list of active servers\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'server_list_detailed', MagicMock(return_value='A')):
            self.assertTrue(nova.server_list_detailed())

    def test_server_show(self):
        if False:
            while True:
                i = 10
        '\n        Test for Return detailed information for an active server\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'server_show', MagicMock(return_value='A')):
            self.assertTrue(nova.server_show('serv_id'))

    def test_secgroup_create(self):
        if False:
            i = 10
            return i + 15
        '\n        Test for Add a secgroup to nova (nova secgroup-create)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'server_list_detailed', MagicMock(return_value='A')):
            self.assertTrue(nova.secgroup_create('name', 'desc'))

    def test_secgroup_delete(self):
        if False:
            print('Hello World!')
        '\n        Test for Delete a secgroup to nova (nova secgroup-delete)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'secgroup_delete', MagicMock(return_value='A')):
            self.assertTrue(nova.secgroup_delete('name'))

    def test_secgroup_list(self):
        if False:
            i = 10
            return i + 15
        '\n        Test for Return a list of available security groups (nova items-list)\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'secgroup_list', MagicMock(return_value='A')):
            self.assertTrue(nova.secgroup_list())

    def test_server_by_name(self):
        if False:
            i = 10
            return i + 15
        '\n        Test for Return information about a server\n        '
        self.mock_auth.side_effect = MagicMock()
        with patch.object(self.mock_auth, 'server_by_name', MagicMock(return_value='A')):
            self.assertTrue(nova.server_by_name('name'))
"""
.. module: security_monkey.tests.watchers.openstack.test_object_container
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.tests.watchers.openstack import OpenStackWatcherTestCase
from cloudaux.openstack.object_container import get_container_metadata
from cloudaux.tests.openstack.mock_object_container import mock_get_container_metadata
import mock
mock.patch('cloudaux.openstack.object_container.get_container_metadata', mock_get_container_metadata).start()

class OpenStackObjectContainerWatcherTestCase(OpenStackWatcherTestCase):

    def pre_test_setup(self):
        if False:
            for i in range(10):
                print('nop')
        super(OpenStackObjectContainerWatcherTestCase, self).pre_test_setup()
        from security_monkey.watchers.openstack.object_store.openstack_object_container import OpenStackObjectContainer
        self.watcher = OpenStackObjectContainer(accounts=[self.account.name])
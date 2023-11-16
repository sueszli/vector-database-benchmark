"""
.. module: security_monkey.tests.watchers.openstack.test_port
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.tests.watchers.openstack import OpenStackWatcherTestCase

class OpenStackPortWatcherTestCase(OpenStackWatcherTestCase):

    def pre_test_setup(self):
        if False:
            print('Hello World!')
        super(OpenStackPortWatcherTestCase, self).pre_test_setup()
        from security_monkey.watchers.openstack.network.openstack_port import OpenStackPort
        self.watcher = OpenStackPort(accounts=[self.account.name])
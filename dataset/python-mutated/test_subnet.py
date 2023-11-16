"""
.. module: security_monkey.tests.watchers.openstack.test_subnet
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.tests.watchers.openstack import OpenStackWatcherTestCase

class OpenStackSubnetWatcherTestCase(OpenStackWatcherTestCase):

    def pre_test_setup(self):
        if False:
            for i in range(10):
                print('nop')
        super(OpenStackSubnetWatcherTestCase, self).pre_test_setup()
        from security_monkey.watchers.openstack.network.openstack_subnet import OpenStackSubnet
        self.watcher = OpenStackSubnet(accounts=[self.account.name])
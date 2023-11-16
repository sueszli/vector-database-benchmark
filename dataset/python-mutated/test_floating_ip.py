"""
.. module: security_monkey.tests.watchers.openstack.test_floating_ip
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.tests.watchers.openstack import OpenStackWatcherTestCase

class OpenStackFloatingIPWatcherTestCase(OpenStackWatcherTestCase):

    def pre_test_setup(self):
        if False:
            return 10
        super(OpenStackFloatingIPWatcherTestCase, self).pre_test_setup()
        from security_monkey.watchers.openstack.network.openstack_floating_ip import OpenStackFloatingIP
        self.watcher = OpenStackFloatingIP(accounts=[self.account.name])
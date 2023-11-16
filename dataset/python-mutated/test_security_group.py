"""
.. module: security_monkey.tests.watchers.openstack.test_security_group
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.tests.watchers.openstack import OpenStackWatcherTestCase

class OpenStackSecurityGroupWatcherTestCase(OpenStackWatcherTestCase):

    def pre_test_setup(self):
        if False:
            return 10
        super(OpenStackSecurityGroupWatcherTestCase, self).pre_test_setup()
        from security_monkey.watchers.openstack.network.openstack_security_group import OpenStackSecurityGroup
        self.watcher = OpenStackSecurityGroup(accounts=[self.account.name])
        self.watcher.detail = 'NONE'

    def test_detail_summary(self):
        if False:
            return 10
        self.watcher.detail = 'SUMMARY'
        super(OpenStackSecurityGroupWatcherTestCase, self).test_slurp()

    def test_detail_full(self):
        if False:
            while True:
                i = 10
        self.watcher.detail = 'FULL'
        super(OpenStackSecurityGroupWatcherTestCase, self).test_slurp()
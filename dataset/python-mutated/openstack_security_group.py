"""
.. module: security_monkey.openstack.auditors.security_group
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.auditors.security_group import SecurityGroupAuditor
from security_monkey.watchers.openstack.network.openstack_security_group import OpenStackSecurityGroup

class OpenStackSecurityGroupAuditor(SecurityGroupAuditor):
    index = OpenStackSecurityGroup.index
    i_am_singular = OpenStackSecurityGroup.i_am_singular
    i_am_plural = OpenStackSecurityGroup.i_am_plural
    network_whitelist = []

    def __init__(self, accounts=None, debug=False):
        if False:
            return 10
        super(OpenStackSecurityGroupAuditor, self).__init__(accounts=accounts, debug=debug)

    def check_securitygroup_ec2_rfc1918(self, sg_item):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _check_internet_cidr(self, cidr):
        if False:
            return 10
        ' some public clouds default to none for any source '
        return not cidr or super(OpenStackSecurityGroupAuditor, self)._check_internet_cidr(cidr)
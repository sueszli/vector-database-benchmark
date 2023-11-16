"""
.. module: security_monkey.openstack.watchers.subnet
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.watchers.openstack.openstack_watcher import OpenStackWatcher

class OpenStackSubnet(OpenStackWatcher):
    index = 'openstack_subnet'
    i_am_singular = 'Subnet'
    i_am_plural = 'Subnets'
    account_type = 'OpenStack'

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(OpenStackSubnet, self).__init__(*args, **kwargs)
        self.item_type = 'subnet'
        self.service = 'network'
        self.generator = 'subnets'
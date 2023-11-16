"""
.. module: security_monkey.openstack.watchers.floating_ip
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.watchers.openstack.openstack_watcher import OpenStackWatcher

class OpenStackFloatingIP(OpenStackWatcher):
    index = 'openstack_floatingip'
    i_am_singular = 'Floating IP'
    i_am_plural = 'Floating IPs'
    account_type = 'OpenStack'

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(OpenStackFloatingIP, self).__init__(*args, **kwargs)
        self.honor_ephemerals = True
        self.item_type = 'floatingip'
        self.service = 'network'
        self.generator = 'ips'
        self.ephemeral_paths = ['updated_at']
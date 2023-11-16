"""
.. module: security_monkey.openstack.watchers.network
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.watchers.openstack.openstack_watcher import OpenStackWatcher

class OpenStackNetwork(OpenStackWatcher):
    index = 'openstack_network'
    i_am_singular = 'Network'
    i_am_plural = 'Network'
    account_type = 'OpenStack'

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(OpenStackNetwork, self).__init__(*args, **kwargs)
        self.honor_ephemerals = True
        self.item_type = 'network'
        self.service = 'network'
        self.generator = 'networks'
        self.ephemeral_paths = ['updated_at']